"""Single-pass replay + grounding labels for BEHAVIOR-2026 sweep.

Combines the official replay (scripts/learning/replay_obs.py -> LeRobot videos) with the
pilot label projector (/root/probes/label_projector.py -> per-frame camera matrix + target
object world positions) in ONE playback, halving sim cost vs two passes.

Reuses replay_obs helpers (task-id mapping, scene file lookup, room instances) by importing
its module. The label callback is chained with LightToggleSynchronizer when the task needs it.

Success contract: judge by the LABELS_DONE line and <labels_out>.done.json file, NEVER the
exit code -- og.shutdown() teardown segfaults AFTER labels+videos are complete are benign.

Usage (behavior conda env, cwd=/root/BEHAVIOR-1K/OmniGibson, taskset -c 0-23, OMP/MKL=16):
  python replay_labeled.py --data_folder /root/replay_root --demo_id 10 \
      --lerobot_root_dir /root/sweep_out/<task> \
      --labels_out /root/sweep_labels/<task>/labels_10.jsonl \
      --targets_json /root/task_targets.json --resume_lerobot
"""

import argparse
import json
import os
import sys

import numpy as np

REPLAY_OBS_DIR = os.environ.get("REPLAY_OBS_DIR", "/root/BEHAVIOR-1K/OmniGibson/scripts/learning")
sys.path.insert(0, REPLAY_OBS_DIR)
import replay_obs as R  # noqa: E402  (sets gm.RENDER_VIEWER_CAMERA etc. at import)

import omnigibson as og  # noqa: E402
from omnigibson.macros import gm  # noqa: E402
from omnigibson.envs import LeRobotPlaybackWrapper  # noqa: E402
from omnigibson.eval.utils.dataset_utils import makedirs_with_mode  # noqa: E402
from omnigibson.eval.utils.eval_utils import PROPRIOCEPTION_INDICES  # noqa: E402
from omnigibson.eval.utils.light_utils import LightToggleSynchronizer  # noqa: E402

# Schema-drift shim from the pilot projector (recordings pre-date controller_groups;
# harmless if fixed upstream).
from omnigibson.robots.robot import Robot as _Robot  # noqa: E402

_orig_load = _Robot._load_state


def _shimmed_load(self, state):
    if "controller_groups" not in state:
        state["controller_groups"] = {}
    return _orig_load(self, state)


_Robot._load_state = _shimmed_load


def quat_to_rot(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def match_label_objects(scene, spec, max_objs):
    """Category-substring match (pilot convention: target_cat in obj.category).

    Targets first, then references, each sorted by name; capped at max_objs total
    (loading_the_car has 293 broad categories -> cap at 20/frame).
    """
    tcats = spec.get("targets") or []
    rcats = spec.get("references") or []

    def matched(o, cats):
        c = getattr(o, "category", "") or ""
        return any(cat in c for cat in cats)

    tgt = sorted((o for o in scene.objects if matched(o, tcats)), key=lambda o: o.name)
    tgt_names = {o.name for o in tgt}
    ref = sorted(
        (o for o in scene.objects if o.name not in tgt_names and matched(o, rcats)),
        key=lambda o: o.name,
    )
    return (tgt + ref)[:max_objs]


def main():
    ap = argparse.ArgumentParser(description="Single-pass replay + labels")
    ap.add_argument("--data_folder", type=str, required=True)
    ap.add_argument("--demo_id", type=int, required=True)
    ap.add_argument("--labels_out", type=str, required=True)
    ap.add_argument("--targets_json", type=str, default="/root/task_targets.json")
    ap.add_argument("--lerobot_root_dir", type=str, required=True)
    ap.add_argument("--lerobot_repo_id", type=str, default=None)
    ap.add_argument("--resume_lerobot", action="store_true")
    ap.add_argument("--max_objs", type=int, default=20)
    ap.add_argument("--flush_every_n_steps", type=int, default=1000)
    args = ap.parse_args()

    task_id = args.demo_id // 10000
    task_name = R._get_task_name_from_task_id(task_id)
    print(f"TASK {task_id} {task_name} demo_id={args.demo_id}", flush=True)

    with open(args.targets_json) as f:
        spec = json.load(f)[task_name]

    gm.ENABLE_TRANSITION_RULES = False

    # --- identical env construction to replay_obs.replay_hdf5_file(output_format="lerobot") ---
    robot_sensor_config = {
        "VisionSensor": {
            "sensor_kwargs": {
                "image_height": 480,
                "image_width": 480,
            },
        },
        "zed_link:Camera:0": {
            "sensor_kwargs": {
                "horizontal_aperture": 40.0,
                "image_height": 720,
                "image_width": 720,
            },
        },
    }
    available_tasks = R._load_challenge_available_tasks()
    if task_name not in available_tasks:
        raise KeyError(f"Task '{task_name}' not found in available challenge task metadata")
    scene_model = available_tasks[task_name][0]["scene_model"]
    full_scene_file = R._find_full_scene_file(task_name=task_name, scene_model=scene_model)
    load_room_instances = R._load_room_instances(task_name=task_name)

    input_path = f"{args.data_folder}/2026-challenge-rawdata/task-{task_id:04d}/episode_{args.demo_id:08d}.hdf5"
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    makedirs_with_mode(args.lerobot_root_dir)
    os.makedirs(os.path.dirname(os.path.abspath(args.labels_out)), exist_ok=True)

    # Isaac boot mutex: concurrent instance STARTUP corrupts the shared shader/appdata
    # cache (segfault at extension load, verified 2026-07-11 with 3 simultaneous boots).
    # Serialize only the boot; episodes run fully parallel afterward.
    import fcntl
    boot_lock = open("/root/isaac_boot.lock", "w")
    fcntl.flock(boot_lock, fcntl.LOCK_EX)
    print("BOOT_LOCK_ACQUIRED", flush=True)

    env = LeRobotPlaybackWrapper.create_from_hdf5(
        input_path=input_path,
        full_scene_file=full_scene_file,
        load_room_instances=load_room_instances,
        robot_sensor_config=robot_sensor_config,
        n_render_iterations=1,
        flush_every_n_steps=args.flush_every_n_steps,
        flush_every_n_traj=1,
        include_robot_control=False,
        robot_proprio_keys=list(PROPRIOCEPTION_INDICES["R1Pro"].keys()),
        robot_obs_modalities=["proprio", "rgb", "depth_linear"],
        include_contacts=False,
        output_path=args.lerobot_repo_id or f"b1k/{task_name}",
        root_dir=args.lerobot_root_dir,
        overwrite=not args.resume_lerobot,
        robot_type="R1Pro",
        task_name=task_name,
        include_task_obs=False,
    )
    env.load_observation_space()

    fcntl.flock(boot_lock, fcntl.LOCK_UN)
    print("BOOT_LOCK_RELEASED", flush=True)

    scene = env.scene
    robot = scene.robots[0]

    # --- label objects (category match, pilot convention) ---
    label_objs = match_label_objects(scene, spec, args.max_objs)
    print("LABEL_OBJS", len(label_objs), [o.name for o in label_objs], flush=True)
    if not label_objs:
        print(f"WARN_NO_TARGETS task={task_name} cats={spec.get('targets')}", flush=True)

    # --- camera machinery (pilot-validated) ---
    # OG sensor wrappers are not initialized in playback mode (touching robot.sensors
    # segfaults natively) -- locate the zed Camera prim on the USD stage, use it ONLY for
    # the static local offset + intrinsics; per-frame pose comes from the physics-side link.
    cam_prim = None
    for prim in og.sim.stage.Traverse():
        path = str(prim.GetPath())
        if prim.GetTypeName() == "Camera" and "zed" in path.lower():
            cam_prim = prim
            break
    assert cam_prim is not None, "no zed Camera prim found on stage"
    print("CAM_PRIM", str(cam_prim.GetPath()), flush=True)

    # USD stage transforms are STALE during playback -- use physics link pose per frame,
    # composed with the camera's static local offset (column-vector convention).
    zed_link = None
    for lname, link in robot.links.items():
        if "zed" in lname.lower():
            zed_link = link
            break
    assert zed_link is not None, f"no zed link among {list(robot.links)[:20]}"
    print("CAM_LINK", zed_link.prim_path, flush=True)

    from pxr import UsdGeom

    M_local_row = np.array(UsdGeom.Xformable(cam_prim).GetLocalTransformation(), dtype=np.float64).reshape(4, 4)
    T_local = M_local_row.T  # Gf row-vector convention -> column-vector homogeneous matrix
    print("CAM_LOCAL_OFFSET", json.dumps([round(float(x), 6) for x in T_local.flatten()]), flush=True)

    # --- demo selection: same as replay_obs default (last demo group) ---
    demo_ids = sorted(int(key.split("_", 1)[1]) for key in env.input_hdf5["data"].keys() if key.startswith("demo_"))
    if not demo_ids:
        raise ValueError(f"No demo groups found in {input_path}")
    episode_id = demo_ids[-1]
    num_samples = int(env.input_hdf5["data"][f"demo_{episode_id}"].attrs["num_samples"])
    print(f"REPLAYING demo_{episode_id} num_samples={num_samples}", flush=True)

    # --- chained post-state-update callback: light sync (if needed) THEN label record ---
    fh = open(args.labels_out, "w", buffering=1)
    ct = {"i": 0}

    def label_cb():
        # NOTE: fires 2x per recorded frame (pilot-verified); consumers use record 2*f for
        # video frame f. Store camera world transform (column-vector homogeneous, p_world =
        # M @ p_cam) + object world positions; projection happens offline.
        i = ct["i"]
        lp, lq = zed_link.get_position_orientation()
        lp = lp.numpy() if hasattr(lp, "numpy") else np.asarray(lp)
        lq = lq.numpy() if hasattr(lq, "numpy") else np.asarray(lq)
        T_link = np.eye(4)
        T_link[:3, :3] = quat_to_rot(lq)
        T_link[:3, 3] = lp
        T_cam = T_link @ T_local
        rec = {"frame": i, "M": [round(float(x), 6) for x in T_cam.flatten()]}
        objs = {}
        for o in label_objs:
            # AABB center = visual center (frame origin sits at object base; pilot-verified
            # origin labels project 40-80px off in overlays)
            try:
                p = o.aabb_center
            except Exception:
                p = o.get_position_orientation()[0]
            p = p.numpy() if hasattr(p, "numpy") else np.asarray(p)
            objs[o.name] = [round(float(x), 5) for x in p]
        rec["objs"] = objs
        fh.write(json.dumps(rec) + "\n")
        ct["i"] += 1

    callbacks = []
    if task_name in R.LIGHT_REPLAY_TASKS:
        light_synchronizer = LightToggleSynchronizer(env.scene)
        callbacks.append(light_synchronizer.sync_from_current_state)
        print("LIGHT_SYNC enabled", flush=True)
    callbacks.append(label_cb)

    if len(callbacks) == 1:
        post_cb = callbacks[0]
    else:
        def post_cb():
            for c in callbacks:
                c()

    env.playback_episode(episode_id=episode_id, record_data=True, post_state_update_callback=post_cb)

    print("Playback complete. Saving data...", flush=True)
    env.save_data()
    fh.close()

    done = {
        "task": task_name,
        "task_id": task_id,
        "demo_id": args.demo_id,
        "episode_id": episode_id,
        "num_samples": num_samples,
        "label_records": ct["i"],
        "frames_est": ct["i"] // 2,
        "n_label_objs": len(label_objs),
        "label_objs": [o.name for o in label_objs],
    }
    with open(args.labels_out + ".done.json", "w") as f:
        json.dump(done, f)
    print("LABELS_DONE", ct["i"], "records ->", args.labels_out, flush=True)

    # Teardown may segfault AFTER this point -- benign, judged by LABELS_DONE/.done.json.
    og.shutdown()


if __name__ == "__main__":
    main()

"""Batched labels-only playback: one env load, many episodes, self-validating.

Why batched
-----------
Measured on a 1957-step radio episode: ~213 s of playback but ~100 s of env setup. Setup is ~32%
of per-episode cost and is pure waste when every instance of a task shares a scene. Reusing one
env across a task's episodes is therefore the single biggest lever available (4x on paper is
wrong; it is ~1.5x, but that is still hours across 2400 episodes).

Why it is safe to reuse
-----------------------
`playback_episode` SETS state every frame (data_wrapper.py), so object placement comes from the
episode being played, not from the env's initial scene. Swapping `self.input_hdf5` between
episodes is therefore sound *provided the object set is identical*. It is not always: task-0001
instances were observed at state dims 667 vs 550, i.e. different object counts.

So this does not assume — it VALIDATES. After each episode we compare the tracked object's
first-frame pose against the published instance JSON
(`2026-challenge-task-instances/.../<task>_0_<instance>_template-tro_state.json`). On the radio
pilot that agreement was 0.0005 m. Any episode exceeding --tol is recorded as FAILED and re-run
in a fresh env, so a composition change degrades throughput rather than silently corrupting labels.

Outputs one JSON per episode: per-frame robot base pose + tracked object poses. That single stream
serves BOTH consumers:
  * point labels   -> target_points = p_obj - p_ee in the base frame
  * odometry truth -> the base trajectory, to arbitrate the 42-52% commanded-vs-achieved gap

Usage:
  python replay_poses_batch.py --data_folder /root/replay_root --task turning_on_radio \
      --episodes 10,20,30 --objects radio_89 --out_dir /root/poses/turning_on_radio
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np

REPLAY_OBS_DIR = os.environ.get(
    "REPLAY_OBS_DIR", "/root/bw/BEHAVIOR-1K/OmniGibson/scripts/learning"
)
sys.path.insert(0, REPLAY_OBS_DIR)
import replay_obs as R  # noqa: E402


class _Truncated(Exception):
    """Raised from the pose callback to stop playback early (see --max_steps)."""


def published_pose(task: str, instance: int, obj_hint: str):
    """First-frame ground truth for `obj_hint` from the published instance JSON, or None."""
    import omnigibson as og  # noqa: F401
    from omnigibson.macros import gm

    root = os.path.join(gm.DATA_PATH, "2026-challenge-task-instances", "scenes")
    pat = os.path.join(root, "*", "json", f"*{task}_instances", f"*_0_{instance}_template-tro_state.json")
    hits = glob.glob(pat)
    if not hits:
        return None
    d = json.load(open(hits[0]))
    for k, v in d.items():
        if isinstance(v, dict) and "root_link" in v and obj_hint.split("_")[0] in k:
            return np.asarray(v["root_link"]["pos"], dtype=float)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_folder", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--episodes", required=True, help="comma-separated demo_ids")
    ap.add_argument("--objects", required=True, help="comma-separated scene object names")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tol", type=float, default=0.01, help="metres; validation threshold")
    ap.add_argument("--max_steps", type=int, default=0,
                    help="stop playback after N simulated steps (0 = full episode). Tier-1 tasks "
                         "run 6k-16k steps vs radio's 1957, and the navigation phase we care "
                         "about is early, so truncation is the biggest available cost lever.")
    a = ap.parse_args()

    demo_ids = [int(x) for x in a.episodes.split(",") if x.strip()]
    want = set(a.objects.split(","))
    os.makedirs(a.out_dir, exist_ok=True)

    import h5py

    import omnigibson as og
    from omnigibson.envs import HDF5PlaybackWrapper
    from omnigibson.macros import gm

    gm.ENABLE_TRANSITION_RULES = False  # asserted by DataPlaybackWrapper; set in replay_obs.main()

    task_id = R._infer_task_id_from_demo_id(demo_ids[0])
    task_name = R._get_task_name_from_task_id(task_id)
    assert task_name == a.task, f"demo_ids map to {task_name!r}, not {a.task!r}"
    scene_model = R._load_challenge_available_tasks()[task_name][0]["scene_model"]

    def path_for(demo_id):
        return f"{a.data_folder}/2026-challenge-rawdata/task-{task_id:04d}/episode_{demo_id:08d}.hdf5"

    common = dict(
        full_scene_file=R._find_full_scene_file(task_name=task_name, scene_model=scene_model),
        load_room_instances=R._load_room_instances(task_name=task_name),
        robot_sensor_config={"VisionSensor": {"sensor_kwargs": {"image_height": 64, "image_width": 64}}},
        n_render_iterations=1,
        flush_every_n_steps=10**9,
        flush_every_n_traj=1,  # wrapper asserts ==1 whenever flush_every_n_steps is set
        include_robot_control=False,
        robot_proprio_keys=list(R.PROPRIOCEPTION_INDICES["R1Pro"].keys()),
        robot_obs_modalities=["proprio"],
        include_contacts=False,
    )

    t_setup = time.time()
    env = HDF5PlaybackWrapper.create_from_hdf5(
        input_path=path_for(demo_ids[0]),
        **common,
        # PID-unique: the wrapper always opens an output hdf5 even with record_data=False, and
        # h5py takes an exclusive lock. Several processes sharing one --out_dir therefore all try
        # to lock the SAME discard file and die with
        #   BlockingIOError: [Errno 11] unable to lock file
        # which killed 5 of 6 workers. Nothing is written here; only the name must be unique.
        output_path=os.path.join(a.out_dir, f".discard_{os.getpid()}.hdf5"),
        compression={"compression": "lzf"},
    )
    env.load_observation_space()
    setup_s = time.time() - t_setup
    print(f"ENV_SETUP {setup_s:.1f}s", flush=True)

    robot = env.env.robots[0]
    # Annotations name objects two different ways: instance-level ('radio_89', which matches the
    # scene name exactly) and category-level ('fridge', 'plate', where the scene name is
    # 'fridge_dszchb_0'). A strict `name in want` matcher silently tracks NOTHING for the
    # category-style tasks — roughly a third of the 24 — so match on exact name OR prefix.
    def matches(name: str) -> bool:
        return name in want or any(name.startswith(w + "_") for w in want)

    tracked = [o for o in env.env.scene.objects if matches(o.name)]
    print(f"tracking {len(tracked)}: {[o.name for o in tracked]}", flush=True)
    assert tracked, f"none of {want} matched any scene object"

    ok, failed = [], []
    for demo_id in demo_ids:
        out = os.path.join(a.out_dir, f"ep{demo_id}.json")
        if os.path.exists(out):
            print(f"SKIP {demo_id} (exists)", flush=True)
            ok.append(demo_id)
            continue

        if env.input_hdf5.filename != path_for(demo_id):
            env.input_hdf5.close()
            env.input_hdf5 = h5py.File(path_for(demo_id), "r")

        frames: list[dict] = []

        # Re-resolve per episode. Sliceable tasks (make_pizza, cook_cabbage, slicing_vegetables)
        # DESTROY prims mid-episode — vidalia_onion_80 becomes half_vidalia_onion_80_{0,1} — and a
        # stale reference then raises "prim view ... is not a valid view", which killed all 100
        # make_pizza episodes. Re-resolving also picks up the halves once they exist.
        ep_tracked = [o for o in env.env.scene.objects if matches(o.name)]

        def log_poses():
            # Truncation: playback_episode() has no step limit, so stop it from the callback.
            # The callback fires twice per simulated step, hence the 2x.
            if a.max_steps and len(frames) >= a.max_steps * 2:
                raise _Truncated
            p, q = robot.get_position_orientation()
            objs = {}
            for o in ep_tracked:
                try:
                    op, oq = o.get_position_orientation()
                except Exception:  # noqa: BLE001 — prim destroyed (sliced/diced); absent is correct
                    objs[o.name] = None
                    continue
                objs[o.name] = {"pos": [float(x) for x in op], "quat": [float(x) for x in oq]}
            frames.append({
                "base_pos": [float(x) for x in p], "base_quat": [float(x) for x in q],
                "objects": objs,
            })

        ids = sorted(int(k.split("_", 1)[1]) for k in env.input_hdf5["data"] if k.startswith("demo_"))
        t0 = time.time()
        truncated = False
        try:
            env.playback_episode(episode_id=ids[-1], record_data=False,
                                 post_state_update_callback=log_poses)
        except _Truncated:
            truncated = True  # expected stop, not a failure
        except Exception as e:  # noqa: BLE001
            print(f"FAILED {demo_id}: {type(e).__name__}: {e}", flush=True)
            failed.append(demo_id)
            continue
        wall = time.time() - t0

        # The callback fires twice per simulated step; keep one sample per step.
        frames = frames[::2]

        # SELF-CHECK: first-frame object pose vs the published instance JSON.
        # demo_id encodes BOTH task and instance: task = demo_id // 10000 (replay_obs.py:39),
        # so the instance is (demo_id % 10000) // 10. Using demo_id // 10 silently worked for
        # turning_on_radio ONLY because its task_id is 0, and produced nonsense (instance 35001)
        # for every other task — which then glob-missed and returned val=None instead of failing.
        instance = (demo_id % 10000) // 10
        err = None
        # Validate against ANY tracked object that has a published counterpart — not just the
        # first. The published JSON keys are BDDL synsets ('radio_receiver.n.01_1'), so matching
        # is by category stem: 'radio_89' -> 'radio' hits, but 'coffee_table_koagbh_0' -> 'coffee'
        # does not appear at all. Picking only the first object made validation silently return
        # None on every episode whose first tracked object happened to be the table.
        ref, gt = None, None
        if frames:
            for name, v in frames[0]["objects"].items():
                if not v:
                    continue
                cand = published_pose(task_name, instance, name)
                if cand is not None:
                    ref, gt = name, cand
                    break
        if gt is None:
            # FAIL LOUD. A silent val=None means episodes are written unverified, which is how
            # bad labels get into training sets.
            print(f"UNVALIDATED {demo_id}: no published pose matched instance {instance} "
                  f"for any of {list(frames[0]['objects']) if frames else []} — writing anyway",
                  flush=True)
        elif frames:
            err = float(np.linalg.norm(np.asarray(frames[0]["objects"][ref]["pos"]) - gt))
            if err > a.tol:
                print(f"FAILED {demo_id}: validation {err:.4f}m > {a.tol}m", flush=True)
                failed.append(demo_id)
                continue

        with open(out, "w") as f:
            json.dump({"task": task_name, "demo_id": demo_id, "instance": instance,
                       "n_frames": len(frames), "wall_seconds": round(wall, 1),
                       "validation_error_m": err, "validated": err is not None, "truncated": truncated,
                       "frames": frames}, f)
        ok.append(demo_id)
        print(f"OK {demo_id} {len(frames)}f {wall:.1f}s val={err}", flush=True)

    print(f"BATCH_DONE ok={len(ok)} failed={len(failed)} failed_ids={failed}", flush=True)
    og.shutdown()


if __name__ == "__main__":
    main()

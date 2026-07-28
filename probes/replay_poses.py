"""Labels-only playback: log per-frame world poses, skip all video encoding.

Why
---
`scripts/learning/replay_obs.py` re-renders and encodes six video streams (rgb + depth_linear ×
3 cameras) — measured at 404 s/episode. We already have those videos in the official LeRobot
dataset. What we do NOT have is the per-frame pose stream:

  * target object world pose  -> point labels (`target_points` = p_obj − p_ee in the BASE frame)
  * robot base world pose     -> odometry ground truth (task #20: integrating `base_qvel`
                                 disagrees with the commanded action by 42–52% and there is
                                 currently nothing to arbitrate)

`playback_episode(record_data=False, post_state_update_callback=...)` sets state per frame and
invokes the callback without recording observations, so this is state PLAYBACK — no physics
divergence, unlike stepping recorded actions.

Env construction is copied from replay_obs.py (full_scene_file + load_room_instances from
B100_task_misc.csv) so the registry matches how the demos were recorded.

Usage (behavior env, cwd=<OmniGibson>):
  python replay_poses.py --data_folder /root/replay_root --demo_id 10 --out /root/poses/ep10.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPLAY_OBS_DIR = os.environ.get(
    "REPLAY_OBS_DIR", "/root/bw/BEHAVIOR-1K/OmniGibson/scripts/learning"
)
sys.path.insert(0, REPLAY_OBS_DIR)
import replay_obs as R  # noqa: E402  (sets gm.* at import time)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_folder", required=True)
    ap.add_argument("--demo_id", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--objects", default="",
                    help="comma-separated scene object names to track, e.g. 'radio_89,"
                         "coffee_table_koagbh_0' — take these from the episode annotation's "
                         "object_id field")
    a = ap.parse_args()

    import omnigibson as og  # noqa: F401  (must follow replay_obs import)
    from omnigibson.envs import HDF5PlaybackWrapper
    from omnigibson.macros import gm

    # DataPlaybackWrapper asserts this (data_wrapper.py:508). replay_obs.py sets it inside main(),
    # which never runs when we import it as a module — so set it ourselves.
    gm.ENABLE_TRANSITION_RULES = False

    task_id = R._infer_task_id_from_demo_id(a.demo_id)
    task_name = R._get_task_name_from_task_id(task_id)
    scene_model = R._load_challenge_available_tasks()[task_name][0]["scene_model"]

    common = dict(
        input_path=f"{a.data_folder}/2026-challenge-rawdata/task-{task_id:04d}/episode_{a.demo_id:08d}.hdf5",
        full_scene_file=R._find_full_scene_file(task_name=task_name, scene_model=scene_model),
        load_room_instances=R._load_room_instances(task_name=task_name),
        # No cameras: we record no observations, so sensors are pure overhead.
        robot_sensor_config={"VisionSensor": {"sensor_kwargs": {"image_height": 64, "image_width": 64}}},
        n_render_iterations=1,
        # Wrapper asserts flush_every_n_traj == 1 whenever flush_every_n_steps is set. We record
        # no data anyway (record_data=False), so these only need to be self-consistent.
        flush_every_n_steps=10**9,
        flush_every_n_traj=1,
        include_robot_control=False,
        robot_proprio_keys=list(R.PROPRIOCEPTION_INDICES["R1Pro"].keys()),
        robot_obs_modalities=["proprio"],
        include_contacts=False,
    )

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    env = HDF5PlaybackWrapper.create_from_hdf5(
        **common, output_path=a.out + ".discard.hdf5", compression={"compression": "lzf"}
    )
    env.load_observation_space()

    demo_ids = sorted(int(k.split("_", 1)[1]) for k in env.input_hdf5["data"] if k.startswith("demo_"))
    episode_id = demo_ids[-1]
    n = env.input_hdf5["data"][f"demo_{episode_id}"].attrs["num_samples"]
    print(f">>> {task_name} demo {a.demo_id}: {n} steps", flush=True)

    robot = env.env.robots[0]
    frames: list[dict] = []

    # Which objects to track. Scene objects are named like 'radio_89' / 'coffee_table_koagbh_0'
    # — NOT BDDL synsets ('radio_receiver.n.01_1'), so a synset filter matches nothing. The
    # annotation's `object_id` / `manipulating_object_id` fields use the scene names exactly,
    # so take the list from there; fall back to every ToggledOn-capable object if not supplied.
    if a.objects:
        want = set(a.objects.split(","))
    else:
        want = {o.name for o in env.env.scene.objects
                if any(getattr(k, "__name__", "") == "ToggledOn" for k in getattr(o, "states", {}))}
    tracked = [o for o in env.env.scene.objects if o.name in want]
    print(f"tracking {len(tracked)} objects: {[o.name for o in tracked][:8]}", flush=True)

    def log_poses():
        p, q = robot.get_position_orientation()
        rec = {"t": len(frames), "base_pos": [float(x) for x in p], "base_quat": [float(x) for x in q],
               "objects": {}}
        for obj in tracked:
            op, oq = obj.get_position_orientation()
            rec["objects"][obj.name] = {"pos": [float(x) for x in op], "quat": [float(x) for x in oq]}
        frames.append(rec)

    t0 = time.time()
    env.playback_episode(episode_id=episode_id, record_data=False, post_state_update_callback=log_poses)
    wall = time.time() - t0

    with open(a.out, "w") as f:
        json.dump({"task": task_name, "demo_id": a.demo_id, "n_frames": len(frames),
                   "wall_seconds": round(wall, 1), "frames": frames}, f)

    # Success contract mirrors replay_labeled.py: judge by this line and the output file, NEVER
    # the exit code — og.shutdown() teardown segfaults AFTER the data is written are benign.
    print(f"POSES_DONE {len(frames)} frames in {wall:.1f}s -> {a.out}", flush=True)
    og.shutdown()


if __name__ == "__main__":
    main()

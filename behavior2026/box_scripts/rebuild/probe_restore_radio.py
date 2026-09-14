"""Verify the 2026-09-14 restore fix: for each demo, restore at t_pre = closure-30 and t_post = closure+6
and report the radio pose vs the demo's recorded frame-0 (sampled) pose and vs the template pose, its tilt,
hand->radio distance, restore wall time, and whether it stays put over 30 hold steps."""
import os, sys, json, time
os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")
import numpy as np, h5py
import omnigibson as og
from omnigibson.macros import gm
gm.HEADLESS = True; gm.ENABLE_TRANSITION_RULES = False
from scipy.spatial.transform import Rotation as R
from omnigibson.envs.hdf5_data_wrapper import HDF5PlaybackWrapper
from reverse_curriculum_collect import EVAL_PROPRIO_KEYS, restore_to_frame
from skill_env_wrapper import _np
TEMPLATE = np.array([3.464, 4.886, 0.534])
for d in [int(x) for x in sys.argv[1:]]:
    path = f"/root/rawdemos/task-0000/episode_{d:08d}.hdf5"
    w = HDF5PlaybackWrapper.create_from_hdf5(input_path=path, output_path=f"/root/probe_tmp_{d}.hdf5",
                                             robot_obs_modalities=("proprio",), robot_proprio_keys=EVAL_PROPRIO_KEYS)
    rob = w.scene.robots[0]; radio = w.scene.object_registry("name", "radio_89")
    with h5py.File(path, "r") as f:
        s0 = f["data/demo_0/state"][0]; acts = f["data/demo_0/action"][:]
    p_samp = s0[484:487].astype(float); q_samp = R.from_quat(s0[487:491] / np.linalg.norm(s0[487:491]))
    closure = json.load(open(f"/root/factory_clips/d{d:03d}_meta.json"))["closure"]
    def radio_pose():
        p, q = radio.get_position_orientation(); return _np(p).astype(float).copy(), R.from_quat(_np(q))
    def hand_p():
        return _np(rob.get_eef_position(arm="right")).astype(float).copy()
    for tag, t in (("t_pre", closure - 30), ("t_post", closure + 6)):
        t0 = time.time(); restore_to_frame(w, 0, t); dt = time.time() - t0
        p, q = radio_pose(); M = q.as_matrix()
        tilt = float(np.degrees(np.arccos(np.clip(M[2, 2], -1, 1))))
        print(f"PROBE d{d} {tag} t={t}: restore {dt:.1f}s replayed={getattr(restore_to_frame, 'last_replayed', '?')} | radio {np.round(p, 3).tolist()} "
              f"|to sampled| {np.linalg.norm(p - p_samp):.3f} m |to template| {np.linalg.norm(p - TEMPLATE):.3f} m "
              f"tilt {tilt:.1f} deg yaw-vs-sampled {float((q * q_samp.inv()).magnitude() * 180 / np.pi):.1f} deg "
              f"hand->radio {np.linalg.norm(hand_p() - p):.3f} m", flush=True)
        if tag == "t_pre":
            hold = np.asarray(acts[t], np.float32).copy(); hold[0:3] = 0.0
            for _ in range(30): w.env.step(hold)
            p2, q2 = radio_pose()
            print(f"PROBE d{d} settle30: radio moved {np.linalg.norm(p2 - p):.4f} m rotated {float((q2 * q.inv()).magnitude() * 180 / np.pi):.1f} deg "
                  f"tilt now {float(np.degrees(np.arccos(np.clip(q2.as_matrix()[2, 2], -1, 1)))):.1f} deg", flush=True)
    os.remove(f"/root/probe_tmp_{d}.hdf5") if os.path.exists(f"/root/probe_tmp_{d}.hdf5") else None
print("PROBE_RESTORE_RADIO_DONE", flush=True)
os._exit(0)

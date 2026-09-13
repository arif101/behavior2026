"""Recorded base travel in a demo: restore frames and read the base position; compare the recorded
displacement over 40 strong-base frames with the action replay displacement; also test a large
command (1.0) and 400 steps at 0.3 to get the steady rate."""
import os, sys
os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")
import numpy as np, h5py
import omnigibson as og
from omnigibson.macros import gm
gm.HEADLESS = True; gm.ENABLE_TRANSITION_RULES = False
from omnigibson.envs.hdf5_data_wrapper import HDF5PlaybackWrapper
from reverse_curriculum_collect import EVAL_PROPRIO_KEYS, restore_to_frame
from skill_env_wrapper import _np
d = int(sys.argv[1]) if len(sys.argv) > 1 else 10
w = HDF5PlaybackWrapper.create_from_hdf5(input_path=f"/root/rawdemos/task-0000/episode_{d:08d}.hdf5", output_path=f"/root/bprobe2_tmp_{d}.hdf5",
                                         robot_obs_modalities=("proprio",), robot_proprio_keys=EVAL_PROPRIO_KEYS)
rob = w.scene.robots[0]
with h5py.File(f"/root/rawdemos/task-0000/episode_{d:08d}.hdf5", "r") as f:
    key = sorted(k for k in f["data"].keys() if k.startswith("demo_"))[0]; acts = f[f"data/{key}/action"][:]
n = len(acts)
def pos(): return _np(rob.get_position_orientation()[0]).copy()
frames = [0, 100, 200, 400, 600, 800, 1000, 1200, 1400, min(n - 2, 1800)]
P = {}
for t in frames:
    restore_to_frame(w, 0, t); P[t] = pos()
print("RECORDED base xy per frame:", {t: np.round(P[t][:2], 3).tolist() for t in frames}, flush=True)
print("RECORDED total path (chord sum):", round(sum(float(np.linalg.norm(P[frames[i+1]] - P[frames[i]])) for i in range(len(frames) - 1)), 3), flush=True)
mag = np.linalg.norm(acts[:, :2], axis=1); t0 = int(np.argmax(mag > 0.2))
restore_to_frame(w, 0, t0); pa = pos(); restore_to_frame(w, 0, t0 + 40); pb = pos()
print(f"RECORDED displacement frames {t0}->{t0+40}: {np.linalg.norm(pb - pa):.4f} m; mean |cmd xy| {np.abs(acts[t0:t0+40,:2]).mean():.3f}", flush=True)
restore_to_frame(w, 0, t0); p0 = pos()
for t in range(t0, t0 + 40): w.env.step(np.asarray(acts[t], np.float32))
print(f"REPLAY displacement same frames: {np.linalg.norm(pos() - p0):.4f} m", flush=True)
hold = np.asarray(acts[t0], np.float32).copy(); hold[0:3] = 0.0
for val in (0.3, 1.0, 3.0):
    restore_to_frame(w, 0, t0)
    for _ in range(10): w.env.step(hold)
    p0 = pos(); cmd = hold.copy(); cmd[0] = val
    for k in range(400): w.env.step(cmd)
    print(f"CONST x={val} x400: moved {np.linalg.norm(pos() - p0):.4f} m ({np.linalg.norm(pos()-p0)/400:.5f} m/step)", flush=True)
print("BASE_PROBE2_DONE", flush=True)
os._exit(0)

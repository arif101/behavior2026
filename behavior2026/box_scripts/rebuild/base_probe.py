"""Base command semantics probe: (1) replay 40 demo steps with strong base action and measure base
displacement; (2) apply a constant [0.3,0,0] / [0,0.3,0] / [0,0,0.3] for 40 steps each and measure;
(3) print the base controller's config. Demo 10."""
import os, sys, json
os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")
import numpy as np, h5py, torch as th
import omnigibson as og
from omnigibson.macros import gm
gm.HEADLESS = True; gm.ENABLE_TRANSITION_RULES = False
from omnigibson.envs.hdf5_data_wrapper import HDF5PlaybackWrapper
from reverse_curriculum_collect import EVAL_PROPRIO_KEYS, restore_to_frame
from skill_env_wrapper import _np
d = int(sys.argv[1]) if len(sys.argv) > 1 else 10
w = HDF5PlaybackWrapper.create_from_hdf5(input_path=f"/root/rawdemos/task-0000/episode_{d:08d}.hdf5", output_path=f"/root/bprobe_tmp_{d}.hdf5",
                                         robot_obs_modalities=("proprio",), robot_proprio_keys=EVAL_PROPRIO_KEYS)
rob = w.scene.robots[0]
with h5py.File(f"/root/rawdemos/task-0000/episode_{d:08d}.hdf5", "r") as f:
    key = sorted(k for k in f["data"].keys() if k.startswith("demo_"))[0]; acts = f[f"data/{key}/action"][:]
bc = rob.controllers.get("base")
print("BASE_CTRL", type(bc).__name__, "in_limits", getattr(bc, "_command_input_limits", None), "out_limits", getattr(bc, "_command_output_limits", None),
      "mode", getattr(bc, "_motor_type", None), "control_freq", getattr(bc, "_control_freq", None), flush=True)
print("action_dim", rob.action_dim, "base dof idx", _np(rob.base_control_idx) if hasattr(rob, "base_control_idx") else None, flush=True)
mag = np.linalg.norm(acts[:, :2], axis=1); t0 = int(np.argmax(mag > 0.2)); print("demo strong-base frame", t0, "cmd", acts[t0, :3].round(3), flush=True)
restore_to_frame(w, 0, t0)
def pos(): return _np(rob.get_position_orientation()[0]).copy()
p0 = pos()
for t in range(t0, t0 + 40):
    w.env.step(np.asarray(acts[t], np.float32))
p1 = pos(); print(f"REPLAY 40 demo steps: base moved {np.linalg.norm(p1 - p0):.4f} m ({np.linalg.norm((p1-p0))/40:.5f} m/step); mean |cmd xy| {np.abs(acts[t0:t0+40,:2]).mean():.3f}", flush=True)
hold = np.asarray(acts[t0], np.float32).copy(); hold[0:3] = 0.0
for ch, name in ((0, "x"), (1, "y"), (2, "rz")):
    restore_to_frame(w, 0, t0)
    for _ in range(10): w.env.step(hold)
    p0 = pos(); q0 = _np(rob.get_position_orientation()[1]).copy()
    cmd = hold.copy(); cmd[ch] = 0.3
    for _ in range(40): w.env.step(cmd)
    p1 = pos(); q1 = _np(rob.get_position_orientation()[1]).copy()
    print(f"CONST {name}=0.3 x40: moved {np.linalg.norm(p1 - p0):.4f} m ({np.linalg.norm(p1-p0)/40:.5f} m/step) dxy={np.round(p1[:2]-p0[:2],4).tolist()} dquat={np.round(q1-q0,3).tolist()}", flush=True)
# ramp test: constant x=0.3 for 200 steps, print every 40
restore_to_frame(w, 0, t0)
for _ in range(10): w.env.step(hold)
p0 = pos(); cmd = hold.copy(); cmd[0] = 0.3
for k in range(1, 201):
    w.env.step(cmd)
    if k % 40 == 0: print(f"RAMP x=0.3 k={k}: moved {np.linalg.norm(pos() - p0):.4f} m", flush=True)
print("BASE_PROBE_DONE", flush=True)
os._exit(0)

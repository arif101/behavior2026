"""FK precompute (ARCH_4D_ATTENTION_SPEC A1): per training frame, the three camera poses in the robot BASE frame
[px py pz qx qy qz qw] x 3 (zed, left_realsense, right_realsense) from the 18 proprio joints (trunk 53:57, left arm
3:10, right arm 28:35), using the same robot + sensors as the eval harness (HDF5PlaybackWrapper env of a demo) and the
same relative-pose computation as the serving wrapper (rel(sensor)). Output /root/fk/cam_pose.npy [N, 21] aligned with
proprio.npy / index.npy. Unique joint configurations are cached (rounded to 2e-3 rad)."""
import os, sys, time
os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")
import numpy as np, torch as th
import omnigibson as og
from omnigibson.macros import gm
gm.HEADLESS = True; gm.ENABLE_TRANSITION_RULES = False
from omnigibson.envs.hdf5_data_wrapper import HDF5PlaybackWrapper
from reverse_curriculum_collect import EVAL_PROPRIO_KEYS
from skill_env_wrapper import _np, P as PS

def q2r(q):
    x, y, z, w = [float(v) for v in q]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
def r2q(R):
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    return np.array([(R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w), w]) if w > 1e-6 else np.array([0, 0, 0, 1.0])

demo = int(sys.argv[1]) if len(sys.argv) > 1 else 10
w = HDF5PlaybackWrapper.create_from_hdf5(input_path=f"/root/rawdemos/task-0000/episode_{demo:08d}.hdf5", output_path=f"/root/fk_tmp_{demo}.hdf5",
                                         robot_obs_modalities=("proprio",), robot_proprio_keys=EVAL_PROPRIO_KEYS)
rob = w.scene.robots[0]
sensors = {}
for sname, sensor in rob.sensors.items():
    for key in ("zed", "left_realsense", "right_realsense"):
        if key in sname: sensors[key] = sensor
assert len(sensors) == 3, list(rob.sensors.keys())
order = ["zed", "left_realsense", "right_realsense"]
idx_trunk = np.asarray(_np(rob.trunk_control_idx), int); idx_l = np.asarray(_np(rob.arm_control_idx["left"]), int); idx_r = np.asarray(_np(rob.arm_control_idx["right"]), int)
idx = np.concatenate([idx_trunk, idx_l, idx_r]); print("dof idx trunk/l/r", idx_trunk, idx_l, idx_r, flush=True)

def rel_poses():
    bp, bq = rob.get_position_orientation(); bp = _np(bp); Rb = q2r(_np(bq)); out = []
    for key in order:
        sp, sq = sensors[key].get_position_orientation(); sp = _np(sp); Rs = q2r(_np(sq))
        out.append(np.concatenate([Rb.T @ (sp - bp), r2q(Rb.T @ Rs)]))
    return np.concatenate(out).astype(np.float32)

prop = np.load("/root/fk/proprio.npy"); N = len(prop); out = np.zeros((N, 21), np.float32)
Q = np.concatenate([prop[:, PS["trunk_qpos"]], prop[:, PS["left"]["arm_qpos"]], prop[:, PS["right"]["arm_qpos"]]], axis=1)   # [N, 18]
keys = np.round(Q / 2e-3).astype(np.int64); cache = {}; t0 = time.time(); hits = 0
for n in range(N):
    k = keys[n].tobytes()
    if k in cache: out[n] = cache[k]; hits += 1; continue
    rob.set_joint_positions(th.as_tensor(Q[n], dtype=th.float32), indices=th.as_tensor(idx), drive=False)
    og.sim.step(render=False)
    r = rel_poses(); cache[k] = r; out[n] = r
    if n % 5000 == 0: print(f"{n}/{N} cache={len(cache)} hits={hits} {time.time()-t0:.0f}s", flush=True); np.save("/root/fk/cam_pose.npy", out)
np.save("/root/fk/cam_pose.npy", out); print("FK_DONE", out.shape, "unique", len(cache), flush=True)
os._exit(0)

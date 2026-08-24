"""Replay round 11: radio at the SETTLED pulled POSE (position AND orientation).

R9 verdict: in the RECORDING the assist pulls the radio ~5cm horizontally into
the finger cage at closure+3 (the ball comes to the hand). Replayed fingers
arrive within 1.8mm of the pulled box, so: pre-place the radio at the pull
destination (legal instance variation), state-replay the hand, NATIVE trigger
only. GT pass measures the pull vector from per-frame radio positions.

Trials: pull*1.0, pull +/-1cm along pull axis, +/-1cm lateral. Filmed; on AG,
demo lift + still-15.

Run (training paused):  OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root \
    OMNIGIBSON_HEADLESS=1 python -u rt_replay_test10.py --demo 30
"""
import argparse
import json
import os

import numpy as np

os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")

OUT = "/root/rt11_film"


def lookat_quat(pos, target):
    f = np.asarray(target, float) - np.asarray(pos, float); f /= np.linalg.norm(f)
    up = np.array([0.0, 0.0, 1.0])
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    m = np.stack([r, u, -f], axis=1)
    w = np.sqrt(max(1e-9, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    return np.array([(m[2, 1] - m[1, 2]) / (4 * w), (m[0, 2] - m[2, 0]) / (4 * w),
                     (m[1, 0] - m[0, 1]) / (4 * w), w])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", type=int, default=30)
    ap.add_argument("--back", type=int, default=5)
    ap.add_argument("--tol", type=float, default=0.006)
    ap.add_argument("--settle", type=int, default=40)
    a = ap.parse_args()

    import h5py
    import imageio
    import torch as th
    import omnigibson as og
    from omnigibson.macros import gm
    gm.HEADLESS = True; gm.ENABLE_TRANSITION_RULES = False
    from omnigibson.envs.hdf5_data_wrapper import HDF5PlaybackWrapper
    from reverse_curriculum_collect import EVAL_PROPRIO_KEYS, restore_to_frame
    from skill_env_wrapper import _np, P, A_TORSO

    os.makedirs(OUT, exist_ok=True)
    man = json.load(open(f"/root/snapshot_bank_v21/manifest_d{a.demo}.json"))
    G = next(r for r in man["entries"] if str(r["stage"]) == "G")
    closure = G["grasp_closure_frame"]
    lift_z = G["lift_z"]
    f0, f_end = closure - a.back, closure + 22

    wrapper = HDF5PlaybackWrapper.create_from_hdf5(
        input_path=f"/root/rawdemos/task-0000/episode_{a.demo:08d}.hdf5",
        output_path="/root/rtt11_tmp.hdf5",
        robot_obs_modalities=("proprio",), robot_proprio_keys=EVAL_PROPRIO_KEYS)
    radio = [o for o in wrapper.scene.objects if "radio" in o.name.lower()][0]
    rob = wrapper.scene.robots[0]
    cam = og.sim.viewer_camera

    with h5py.File(f"/root/rawdemos/task-0000/episode_{a.demo:08d}.hdf5", "r") as f:
        key = sorted(k for k in f["data"].keys() if k.startswith("demo_"))[0]
        acts = f[f"data/{key}/action"][:]

    def q61():
        obs = wrapper.env.get_obs()[0]
        def find(node, sub):
            if isinstance(node, dict):
                for k, v in node.items():
                    r = find(v, sub)
                    if r is not None:
                        return r
                    if sub in str(k):
                        return v
            return None
        return _np(find(obs, "proprio")).reshape(-1)

    def contact():
        try:
            cs, _ = rob._find_gripper_contacts(arm="right")
            return any(radio.name in c for c in cs)
        except Exception:  # noqa: BLE001
            return False

    def native_ag():
        return rob._ag_obj_constraint_params.get("right") is not None

    def step_cmd(cmd, settle, tol):
        q = q61()
        for _ in range(settle):
            wrapper.env.step(cmd)
            q = q61()
            if (np.abs(q[P["left"]["arm_qpos"]] - cmd[7:14]).max() < tol
                    and np.abs(q[P["right"]["arm_qpos"]] - cmd[15:22]).max() < tol
                    and np.abs(q[P["trunk_qpos"]] - cmd[A_TORSO]).max() < tol):
                break

    def links():
        return [_np(l.get_position_orientation()[0])
                for l in rob.finger_links["right"]]

    def grab(target):
        pos = np.asarray(target) + np.array([0.32, -0.30, 0.20])
        cam.set_position_orientation(
            th.as_tensor(pos, dtype=th.float32),
            th.as_tensor(lookat_quat(pos, target), dtype=th.float32))
        for _ in range(2):
            og.sim.render()
        obs, _ = cam.get_obs()
        return np.asarray(obs["rgb"])[..., :3].astype(np.uint8)

    # GT pass: joint states + radio positions -> pull vector
    qL_gt, qR_gt, tr_gt, rp_gt, rq_gt = {}, {}, {}, {}, {}
    for t in range(f0, f_end + 1):
        restore_to_frame(wrapper, 0, t)
        q = q61()
        qL_gt[t] = q[P["left"]["arm_qpos"]].copy()
        qR_gt[t] = q[P["right"]["arm_qpos"]].copy()
        tr_gt[t] = q[P["trunk_qpos"]].copy()
        rp, rq = radio.get_position_orientation()
        rp_gt[t], rq_gt[t] = _np(rp).copy(), _np(rq).copy()
    json.dump({t: [list(map(float, rp_gt[t])), list(map(float, rq_gt[t]))]
               for t in rp_gt},
              open(f"{OUT}/radio_pose_gt_d{a.demo}.json", "w"))
    pull = rp_gt[closure + 15] - rp_gt[f0]
    pull[2] = 0.0
    pdir = pull / (np.linalg.norm(pull) + 1e-9)
    print("PULL vector (+15):", [round(float(x), 4) for x in pull],
          "norm", round(float(np.linalg.norm(pull)), 4), flush=True)

    def state_cmd(t):
        cmd = np.asarray(acts[t], np.float32).copy()
        cmd[7:14] = qL_gt[t]
        cmd[15:22] = qR_gt[t]
        cmd[A_TORSO] = tr_gt[t]
        return cmd

    dest_t = closure + 15
    TRIALS = [("settled", 0.0), ("settled_p05", 0.005), ("settled_p15", 0.015),
              ("settled_m05", -0.005)]
    all_s = []
    for name, extra in TRIALS:
        restore_to_frame(wrapper, 0, f0)
        pos = rp_gt[dest_t].copy() + extra * pdir
        pos[2] = rp_gt[f0][2]
        radio.set_position_orientation(
            position=th.as_tensor(pos, dtype=th.float32),
            orientation=th.as_tensor(rq_gt[dest_t], dtype=th.float32))
        frames, ag_frame, ncontact, min_gap = [], None, 0, 9.9
        for t in range(f0, f_end + 1):
            step_cmd(state_cmd(t), a.settle, a.tol)
            lo, hi = (_np(x) for x in radio.aabb)
            g = min(float(np.linalg.norm(
                np.maximum(np.maximum(lo - p, 0.0), p - hi))) for p in links())
            min_gap = min(min_gap, g)
            c = contact()
            ncontact += int(c)
            frames.append(grab(0.5 * (np.mean(links(), axis=0)
                                      + _np(radio.get_position_orientation()[0]))))
            if native_ag() and ag_frame is None:
                ag_frame = t
                print(f"AG_FIRED trial={name} rel={t - closure}", flush=True)
            print(f"RP11 {name} rel={t - closure:+d} gap={g:.4f} contact={c} "
                  f"ag={native_ag()}", flush=True)
        lifted = still = False
        if ag_frame is not None:
            tp_prev, dwell = None, 0
            for t in range(f_end + 1, min(closure + 90, len(acts))):
                cmd = np.asarray(acts[t], np.float32).copy()
                cmd[22] = -1.0
                step_cmd(cmd, a.settle, a.tol)
                tp = _np(radio.get_position_orientation()[0])
                lifted = lifted or tp[2] > lift_z + 0.05
                stillnow = (tp_prev is not None and
                            float(np.linalg.norm(tp - tp_prev)) < 0.005)
                tp_prev = tp.copy()
                dwell = dwell + 1 if (lifted and stillnow and native_ag()) else 0
                if t % 2 == 0:
                    frames.append(grab(0.5 * (np.mean(links(), axis=0) + tp)))
                if dwell >= 15:
                    still = True
                    break
        s = dict(trial=name, extra=extra,
                 ag=ag_frame is not None,
                 ag_rel=None if ag_frame is None else ag_frame - closure,
                 contact_frames=ncontact, min_gap=round(min_gap, 4),
                 lifted=lifted, still15=still)
        all_s.append(s)
        print("SUMMARY", json.dumps(s), flush=True)
        imageio.mimsave(f"{OUT}/rt11_{name}.mp4", frames, fps=4)
    json.dump(all_s, open(f"{OUT}/rt11_d{a.demo}.json", "w"), indent=1)
    print("DONE", flush=True)
    os._exit(0)


main()

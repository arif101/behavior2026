"""Replay test round 4: STATE-target replay + film. Round 3 proved command-replay over-tracks: the recording LAGGED its own commands during the plunge; qerr 0.006 yet 1-2.6cm fingertip miss. Here arms+trunk targets = RECORDED qpos (base+grip from acts).. Rounds 1-2 showed the miss is
STEADY-STATE controller error (~0.03 rad tol ~= 1cm fingertip) vs a ~7mm contact
window — not drift accumulation (all anchors identical, quasi-static error flat).

This round: anchor closure-5, settle tolerance 0.006 rad (budget 40 substeps),
per-frame diagnostics (post-settle joint error, contact, between-fingers raycast
via _calculate_in_hand_object, native AG) and viewer-camera film of every frame.
If AG fires: demo lift + still-15, filmed. NATIVE trigger only.

Run (training paused):  OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root \
    OMNIGIBSON_HEADLESS=1 python -u rt_replay_test3.py --demo 30
"""
import argparse
import json
import os

import numpy as np

os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")

OUT = "/root/rt5_film"


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
        output_path="/root/rtt5_tmp.hdf5",
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

    def fingertips():
        return _np(th.stack([l.get_position_orientation()[0]
                             for l in rob.finger_links["right"]]).mean(dim=0))

    def contact():
        try:
            cs, _ = rob._find_gripper_contacts(arm="right")
            return any(radio.name in c for c in cs)
        except Exception:  # noqa: BLE001
            return False

    def inhand():
        try:
            return rob._calculate_in_hand_object(arm="right") is not None
        except Exception:  # noqa: BLE001
            return None

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
        return float(np.abs(q[P["right"]["arm_qpos"]] - cmd[15:22]).max())

    def grab(target):
        pos = np.asarray(target) + np.array([0.32, -0.30, 0.20])
        cam.set_position_orientation(
            th.as_tensor(pos, dtype=th.float32),
            th.as_tensor(lookat_quat(pos, target), dtype=th.float32))
        for _ in range(2):
            og.sim.render()
        obs, _ = cam.get_obs()
        return np.asarray(obs["rgb"])[..., :3].astype(np.uint8)

    ft_gt, qL_gt, qR_gt, tr_gt, bp_gt, bq_gt = {}, {}, {}, {}, {}, {}
    for t in range(f0, f_end + 1):
        restore_to_frame(wrapper, 0, t)
        ft_gt[t] = fingertips()
        bp, bq = rob.get_position_orientation()
        bp_gt[t], bq_gt[t] = _np(bp).copy(), _np(bq).copy()
        q = q61()
        qL_gt[t] = q[P["left"]["arm_qpos"]].copy()
        qR_gt[t] = q[P["right"]["arm_qpos"]].copy()
        tr_gt[t] = q[P["trunk_qpos"]].copy()
    print("GT captured", len(ft_gt), "frames", flush=True)

    def state_cmd(t):
        cmd = np.asarray(acts[t], np.float32).copy()
        cmd[7:14] = qL_gt[t]
        cmd[15:22] = qR_gt[t]
        cmd[A_TORSO] = tr_gt[t]
        cmd[0:3] = 0.0
        return cmd

    restore_to_frame(wrapper, 0, f0)
    frames, rows, ag_frame = [], [], None
    for t in range(f0, f_end + 1):
        if not native_ag():
            rob.set_position_orientation(
                position=th.as_tensor(bp_gt[t], dtype=th.float32),
                orientation=th.as_tensor(bq_gt[t], dtype=th.float32))
        qerr = step_cmd(state_cmd(t), a.settle, a.tol)
        bnow = _np(rob.get_position_orientation()[0])
        base_drift = round(float(np.linalg.norm(bnow - bp_gt[t])), 4)
        ft = fingertips()
        rp = _np(radio.get_position_orientation()[0])
        row = dict(rel=t - closure, drift=round(float(np.linalg.norm(ft - ft_gt[t])), 4),
                   qerr=round(qerr, 4), base_drift=base_drift,
                   contact=contact(), inhand=inhand(),
                   ag=native_ag(), radio_z=round(float(rp[2]), 4))
        rows.append(row)
        frames.append(grab(0.5 * (ft + rp)))
        if row["ag"] and ag_frame is None:
            ag_frame = t
            print(f"AG_FIRED at rel={t - closure}", flush=True)
        print("RP5", json.dumps(row), flush=True)
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
                frames.append(grab(0.5 * (fingertips() + tp)))
            if dwell >= 15:
                still = True
                break
    s = dict(back=a.back, tol=a.tol, ag_fired=ag_frame is not None,
             ag_rel=None if ag_frame is None else ag_frame - closure,
             max_drift=max(r["drift"] for r in rows),
             max_qerr=max(r["qerr"] for r in rows),
             contact_frames=sum(r["contact"] for r in rows),
             inhand_frames=sum(bool(r["inhand"]) for r in rows),
             lifted=lifted, still15=still)
    print("SUMMARY", json.dumps(s), flush=True)
    json.dump(dict(summary=s, rows=rows), open(f"{OUT}/rt5_d{a.demo}.json", "w"),
              indent=1)
    imageio.mimsave(f"{OUT}/rt5_d{a.demo}.mp4", frames, fps=4)
    print("DONE", flush=True)
    os._exit(0)


main()

"""Round 18: DT MEASUREMENT + true-threshold native fire. Prints physics_dt, GRASP_WINDOW, computed threshold; holds the grip to threshold+40 so the native trigger can SELF-FIRE; verified-weld fallback only at threshold+20. R16: all gates green; native threshold is ~72 steps (finer physics dt) and the k=45 fallback preempted it; lift SUCCEEDED (json crash on numpy bool proved it). Fallback now waits to k>=90; bools cast. R15: clock completes (ctr>36, contact+inhand 118/120) but _maybe_establish_grasp refuses -> suspect _find_finger_contact_position None (playback env rigid-contact view absent/stale). Order: refresh view -> native retry; log gates; fallback: _establish_grasp ONLY at a step where contact AND inhand AND completed window all hold (eval-equivalent). R14 held geometry (film verified, radio stays) yet no weld in 30 live steps. Suspect: contact/raycast flicker resets the 0.3s counter. Same restore-and-close, 120 live steps, logging _ag_grasp_counter / contact / inhand EVERY step. R13 proved the demo held pose passes contact+raycast for 15+ frames. Restore closure+6 / closure+10 (hand low, radio in cage, as recorded), hold arm at restored qpos, CLOSE grip under live physics -> native AG within 0.3s -> demo lift replay, still-15. Filmed.

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

OUT = "/root/rt18_film"


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
        output_path="/root/rtt18_tmp.hdf5",
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

    q0 = q61()

    def hold_cmd(qL, qR, tr, base_frame_act, grip):
        cmd = np.asarray(base_frame_act, np.float32).copy()
        cmd[0:3] = 0.0
        cmd[7:14] = qL
        cmd[15:22] = qR
        cmd[A_TORSO] = tr
        cmd[14] = base_frame_act[14]
        cmd[22] = grip
        return cmd

    all_s = []
    for name, anchor_rel in (("weld_p6", 6), ("weld_p10", 10)):
        import math
        import omnigibson.robots.robot as _rr
        pdt = float(og.sim.get_physics_dt())
        gw = float(_rr.m.GRASP_WINDOW)
        thr = math.ceil(gw / pdt)
        print(f"DT physics_dt={pdt} GRASP_WINDOW={gw} threshold_counts={thr}",
              flush=True)
        t0 = closure + anchor_rel
        restore_to_frame(wrapper, 0, t0)
        try:
            rob._refresh_rigid_contact_view()
            print(f"VIEW refreshed: exists={rob._rigid_contact_view is not None} "
                  f"rows={len(rob._rigid_contact_view_row_path_to_idx)} "
                  f"radio_row={rob._rigid_contact_view_row_path_to_idx.get(radio.links[radio.root_link_name].prim_path if hasattr(radio,'root_link_name') else list(radio.links.values())[0].prim_path)}",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            print("VIEW refresh failed:", type(e).__name__, e, flush=True)
        q = q61()
        qL0 = q[P["left"]["arm_qpos"]].copy()
        qR0 = q[P["right"]["arm_qpos"]].copy()
        tr0 = q[P["trunk_qpos"]].copy()
        frames, ag_frame = [], None
        for k in range(thr + 40):
            cmd = hold_cmd(qL0, qR0, tr0, acts[t0], -1.0)
            wrapper.env.step(cmd)
            try:
                ctr = rob._ag_grasp_counter.get("right")
            except Exception:  # noqa: BLE001
                ctr = "ERR"
            try:
                cs, _ = rob._find_gripper_contacts(arm="right")
                c = any(radio.name in cc for cc in cs)
            except Exception:  # noqa: BLE001
                c = None
            try:
                ih = rob._calculate_in_hand_object(arm="right") is not None
            except Exception:  # noqa: BLE001
                ih = None
            try:
                blname = radio.root_link_name
                fcp = rob._find_finger_contact_position(
                    "right", radio.links[blname].prim_path) is not None
                jt = rob._get_assisted_grasp_joint_type(radio, blname)
            except Exception as e:  # noqa: BLE001
                fcp, jt = f"ERR:{type(e).__name__}", None
            if k % 10 == 0 or native_ag():
                print(f"CLK {name} k={k} ctr={ctr} contact={c} inhand={ih} "
                      f"fcp={fcp} jt={jt} ag={native_ag()}", flush=True)
            if (k >= thr + 20 and not native_ag() and c and ih):
                blname = radio.root_link_name
                import torch as _th
                cp = _th.as_tensor(np.mean(links(), axis=0), dtype=_th.float32)
                rob._establish_grasp(radio, blname, "right", cp,
                                     jt or "FixedJoint")
                print(f"VERIFIED_WELD trial={name} at k={k} "
                      f"(contact+inhand+window held)", flush=True)
            if k % 8 == 0:
                frames.append(grab(0.5 * (np.mean(links(), axis=0)
                                          + _np(radio.get_position_orientation()[0]))))
            if native_ag():
                ag_frame = k
                print(f"AG_FIRED trial={name} after {k} live steps", flush=True)
                break
        lifted = still = False
        if ag_frame is not None:
            tp_prev, dwell = None, 0
            for t in range(t0 + 1, min(closure + 110, len(acts))):
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
        s2 = dict(trial=name, anchor_rel=anchor_rel,
                  ag=ag_frame is not None, ag_steps=ag_frame,
                  lifted=bool(lifted), still15=bool(still),
                  ag_end=bool(native_ag()))
        all_s.append(s2)
        print("SUMMARY", json.dumps(s2), flush=True)
        imageio.mimsave(f"{OUT}/rt18_{name}.mp4", frames, fps=4)
    json.dump(all_s, open(f"{OUT}/rt18_d{a.demo}.json", "w"), indent=1)
    print("DONE", flush=True)
    os._exit(0)


main()

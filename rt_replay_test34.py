"""Round 34: 6-DOF POSE SERVO. Film verdict: the tine approaches the front-face button TIP-DOWN (carry wrist), so the palm hits the face before the tip reaches the button. Fix: servo position AND orientation — target wrist orientation = demo press wrist relative to demo radio, retargeted onto our radio attitude; full 6x7 native Jacobian. R32: open-air path works (cos 0.93-0.97, monotonic) but strides starve at 1.2mm vs 12mm commanded (posture conditioning + clipping). Budget 140->400, dq clip 0.05->0.09, step 12->20mm when far. R31: even the high route clips (the corridor from the slot crosses the RIGHT gripper). Fix the cause: truncate approach replay at press0-15 (before the dive that hooks the slot) — fingertip stays in open air; then high-staging -> staging -> button, no retreat leg needed. R30: retreat and free flight track at cos 1.00 (cage theory confirmed), but the straight arc to the staging point clips the top-front edge both attempts (cos collapse at ~7-9cm). Add a high waypoint above staging; descend vertically outside the face; then push. Film verdict: the left tine lands HOOKED inside the handle slot (demo press-end pose vs our rotated radio) — mechanically caged, motion reflects. Path: (1) retreat up+out of the slot, (2) staging point off the button normal, (3) button. Native J each step; cos-collapse triggers re-retreat. R28: JCAL cos=1.00 but SVCHK cos=0.27 -- the hold command was built from acts[press0+9], dragging trunk AND the right arm (radio+button welded to it) every step. Fix: freeze right arm + trunk at CURRENT measured qpos at servo start. R27 lstsq J read cos=-0.58 (anti-correlated). PhysX exposes exact per-link jacobians via rob.get_jacobian(); conventions (row block / link row / col offset) auto-calibrated offline against the logged approach motion; J re-fetched every servo step. R26: the approach delivered 10.1cm in a clean posture, then PROBING destroyed it (16.4cm at servo start) and the stale J reversed. Fix: log (qL, fingertip) during the approach dive, J = lstsq over those real displacements, NO probes; verify each servo step against prediction and rescale. Servo film verdict: all prior servos started from the transport hold posture and folded the arm into the radio/table (self-collision ate every step). Fix: after transport, REPLAY the demo left-arm press joint trajectory (collision-free by construction) to arrive at the recorded press-end posture, THEN local-servo the last ~5cm to the live button. R24 identical trajectory to R23 (base channels were already ~0; wrong diagnosis). Real suspect: the straight fingertip->button line passes THROUGH the radio body; the servo collides en route and thrashes. Fix: stage at button + 4cm radially outward from radio center, then push straight in. No sign flips; re-probe every 15. R23 diverged even after re-probe: the servo hold command carried the demo base velocity channels (teleop was repositioning) so the whole robot + welded radio crept every step, contaminating probes and chasing the target away. Fix: zero base channels in the servo phase; windowed (3-iter) progress check. R22 servo diverged (12.8 -> 17.9cm): probe-identified J pointed wrong. Fixes: probe perturbation 0.04 -> 0.07, direction flip after 2 consecutive worsenings, re-probe J every 20 iters, iteration budget 120. R21: full drift correction (5.4cm/38deg) breaks the grasp; the grasp and press disagree by the in-hand drift. Fix: weld at the proven GT pose; from press0-60, system-ID the LEFT arm Jacobian (probe each joint), then servo the left fingertip to the LIVE button position and push through it. Toggle = success. Streak weld gate restored (it hit 318 in R19); LEFT-arm state-target replay through the press window (GT captured BEFORE the trial); frames saved incrementally so a tail crash cannot eat the film.

R17/18 recipe (restore post-pull frame, 320-step green-gate hold, verified weld)
then replay the demo's own actions through its left-arm press (~frame 1280).
Scored: ToggledOn ∧ weld intact ∧ on-station (radio within 0.10 m of its
press-phase anchor at the toggle moment). Filmed throughout.

Run (training paused):  OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root \
    OMNIGIBSON_HEADLESS=1 python -u /root/rt_replay_test19.py --demo 30
"""
import argparse
import json
import os

import numpy as np

os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")

OUT = "/root/rt34_film"


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
    press0 = max(int(r["frame"]) for r in man["entries"]
                 if r.get("family") == "press")
    lift_z = G["lift_z"]

    wrapper = HDF5PlaybackWrapper.create_from_hdf5(
        input_path=f"/root/rawdemos/task-0000/episode_{a.demo:08d}.hdf5",
        output_path="/root/rtt34_tmp.hdf5",
        robot_obs_modalities=("proprio",), robot_proprio_keys=EVAL_PROPRIO_KEYS)
    radio = [o for o in wrapper.scene.objects if "radio" in o.name.lower()][0]
    rob = wrapper.scene.robots[0]
    cam = og.sim.viewer_camera
    try:
        from omnigibson.object_states import ToggledOn
        toggled = lambda: bool(radio.states[ToggledOn].get_value())  # noqa: E731
    except Exception:  # noqa: BLE001
        toggled = lambda: None  # noqa: E731

    with h5py.File(f"/root/rawdemos/task-0000/episode_{a.demo:08d}.hdf5", "r") as f:
        key = sorted(k for k in f["data"].keys() if k.startswith("demo_"))[0]
        acts = f[f"data/{key}/action"][:]
    horizon = min(len(acts) - 1, press0 + 40)

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

    def links():
        return [_np(l.get_position_orientation()[0])
                for l in rob.finger_links["right"]]

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
            return False

    def native_ag():
        return rob._ag_obj_constraint_params.get("right") is not None

    def step_cmd(cmd):
        q = q61()
        for _ in range(a.settle):
            wrapper.env.step(cmd)
            q = q61()
            if (np.abs(q[P["left"]["arm_qpos"]] - cmd[7:14]).max() < a.tol
                    and np.abs(q[P["right"]["arm_qpos"]] - cmd[15:22]).max() < a.tol
                    and np.abs(q[P["trunk_qpos"]] - cmd[A_TORSO]).max() < a.tol):
                break

    def grab(target):
        pos = np.asarray(target) + np.array([0.45, -0.42, 0.30])
        cam.set_position_orientation(
            th.as_tensor(pos, dtype=th.float32),
            th.as_tensor(lookat_quat(pos, target), dtype=th.float32))
        for _ in range(2):
            og.sim.render()
        obs, _ = cam.get_obs()
        return np.asarray(obs["rgb"])[..., :3].astype(np.uint8)

    # phase 0: GT capture for the press window (left arm + trunk state targets)
    from scipy.spatial.transform import Rotation as _R
    qLp, trp = {}, {}
    R_gt_wrist = R_gt_radio = None
    for t in range(press0 - 120, min(press0 + 41, len(acts))):
        restore_to_frame(wrapper, 0, t)
        q = q61()
        qLp[t] = q[P["left"]["arm_qpos"]].copy()
        trp[t] = q[P["trunk_qpos"]].copy()
        if t == press0 + 5:
            _, wq = rob.eef_links["left"].get_position_orientation()
            _, rq2 = radio.get_position_orientation()
            R_gt_wrist = _R.from_quat(_np(wq))
            R_gt_radio = _R.from_quat(_np(rq2))
    print("GT press-window captured", len(qLp), "frames", flush=True)

    # phase 1: verified weld (R17/18 recipe)
    t0 = closure + 6
    restore_to_frame(wrapper, 0, t0)
    rob._refresh_rigid_contact_view()
    q = q61()
    qL0, qR0 = q[P["left"]["arm_qpos"]].copy(), q[P["right"]["arm_qpos"]].copy()
    tr0 = q[P["trunk_qpos"]].copy()
    frames = []
    green, gap = 0, 0
    for k in range(400):
        cmd = np.asarray(acts[t0], np.float32).copy()
        cmd[0:3] = 0.0
        cmd[7:14] = qL0
        cmd[15:22] = qR0
        cmd[A_TORSO] = tr0
        cmd[22] = -1.0
        wrapper.env.step(cmd)
        if contact() and inhand():
            green += 1
            gap = 0
        else:
            gap += 1
            if gap > 4:
                green = 0
        if k % 40 == 0:
            print(f"GRN k={k} streak={green}", flush=True)
        if k % 20 == 0:
            frames.append(grab(0.5 * (np.mean(links(), axis=0)
                                      + _np(radio.get_position_orientation()[0]))))
        if native_ag():
            print(f"NATIVE_AG at k={k}", flush=True)
            break
        if k >= 320 and green >= 296:
            blname = radio.root_link_name
            cp = th.as_tensor(np.mean(links(), axis=0), dtype=th.float32)
            rob._establish_grasp(radio, blname, "right", cp, "FixedJoint")
            print(f"VERIFIED_WELD at k={k} streak={green}", flush=True)
            break
    if not native_ag():
        print("WELD_FAILED", flush=True)
        os._exit(0)

    from omnigibson.object_states import ToggledOn as _TO

    def button_pos():
        return _np(radio.states[_TO].link.get_position_orientation()[0])

    def lfinger():
        return np.mean([_np(l.get_position_orientation()[0])
                        for l in rob.finger_links["left"]], axis=0)

    # phase 2: chain replay through the press (action replay until press0-60)
    tgl_frame, ag_lost, anchor = None, None, None
    servo_start = press0 - 60
    for t in range(t0 + 1, servo_start):
        cmd = np.asarray(acts[t], np.float32).copy()
        cmd[22] = -1.0
        if t in qLp:
            cmd[7:14] = qLp[t]
            cmd[A_TORSO] = trp[t]
        step_cmd(cmd)
        tp = _np(radio.get_position_orientation()[0])
        if t == press0 - 100:
            anchor = tp.copy()
        if not native_ag() and ag_lost is None:
            ag_lost = t
            print(f"AG_LOST at frame {t} (rel_press {t - press0})", flush=True)
        tg = toggled()
        if tg and tgl_frame is None:
            tgl_frame = t
            drift = (float(np.linalg.norm(tp - anchor))
                     if anchor is not None else None)
            print(f"TOGGLED at frame {t} (press0 {press0}) "
                  f"ag={native_ag()} drift={drift}", flush=True)
        if t % 10 == 0:
            print(f"CH t={t} z={tp[2]:.3f} ag={native_ag()} tg={tg}", flush=True)
        if t % 6 == 0:
            img = grab(0.5 * (np.mean(links(), axis=0) + tp))
            frames.append(img)
            imageio.imwrite(f"{OUT}/f{t:04d}.png", img)
        if tgl_frame is not None and t > tgl_frame + 15:
            break

    # phase 3a: walk the recorded press approach (collision-free posture path)
    ap_q, ap_e = [], []
    if tgl_frame is None:
        for t in range(servo_start, min(press0 - 15, len(acts))):
            cmd = np.asarray(acts[t], np.float32).copy()
            cmd[0:3] = 0.0
            cmd[22] = -1.0
            if t in qLp:
                cmd[7:14] = qLp[t]
                cmd[A_TORSO] = trp[t]
            step_cmd(cmd)
            q_ = q61()
            ap_q.append(q_[P["left"]["arm_qpos"]].copy())
            ap_e.append(lfinger())
            if toggled():
                tgl_frame = t
                print(f"TOGGLED during approach replay t={t}", flush=True)
                break
            if t % 10 == 0:
                d0 = float(np.linalg.norm(button_pos() - lfinger()))
                print(f"AP t={t} btn_dist={d0:.4f}", flush=True)
            if t % 6 == 0:
                frames.append(grab(0.5 * (lfinger() + button_pos())))

    # phase 3b: resolved-rate servo with the native analytic jacobian
    if tgl_frame is None:
        eefL = rob.eef_links["left"]
        link_list = list(rob.links.values())
        eef_idx = link_list.index(eefL)
        armL_idx = np.asarray(_np(rob.arm_control_idx["left"]), int)

        def fetch_J(sel):
            row, blk, coff = sel
            Jf = _np(rob.get_jacobian())
            return Jf[row, blk:blk + 3, :][:, armL_idx + coff]

        def fetch_J6(sel):
            row, blk, coff = sel
            Jf = _np(rob.get_jacobian())
            lin = Jf[row, blk:blk + 3, :][:, armL_idx + coff]
            ang = Jf[row, 3 - blk:6 - blk, :][:, armL_idx + coff]
            return np.vstack([lin, ang])

        dq_last = np.diff(np.stack(ap_q), axis=0)[-8:]
        de_last = np.diff(np.stack(ap_e), axis=0)[-8:]
        best, best_cos = None, -2.0
        for row in (eef_idx - 1, eef_idx):
            if row < 0 or row >= _np(rob.get_jacobian()).shape[0]:
                continue
            for blk in (0, 3):
                for coff in (0, 6):
                    try:
                        Jc = fetch_J((row, blk, coff))
                        preds = dq_last @ Jc.T
                        cs = float(np.mean([
                            np.dot(p, e) / (np.linalg.norm(p) * np.linalg.norm(e) + 1e-9)
                            for p, e in zip(preds, de_last)]))
                        if cs > best_cos:
                            best, best_cos = (row, blk, coff), cs
                    except Exception:  # noqa: BLE001
                        continue
        print(f"JCAL sel={best} cos={best_cos:.2f}", flush=True)
        hold = np.asarray(acts[min(press0 + 9, len(acts) - 1)],
                          np.float32).copy()
        hold[0:3] = 0.0
        hold[22] = -1.0
        qf = q61()
        hold[15:22] = qf[P["right"]["arm_qpos"]]
        hold[A_TORSO] = qf[P["trunk_qpos"]]
        base = hold.copy()
        J = fetch_J(best)
        hist = []
        wp = 0
        rc0 = _np(radio.get_position_orientation()[0])
        lf0 = lfinger()
        away = lf0 - rc0
        away[2] = 0.0
        away = away / (np.linalg.norm(away) + 1e-9)
        retreat_pt = lf0 + 0.05 * away + np.array([0.0, 0.0, 0.06])
        bad_cos = 0
        for it in range(400):
            bp = button_pos()
            rc = _np(radio.get_position_orientation()[0])
            outv = bp - rc
            outv[2] = 0.0
            outv = outv / (np.linalg.norm(outv) + 1e-9)
            lf = lfinger()
            d_btn = float(np.linalg.norm(bp - lf))
            targets = [bp + 0.045 * outv + np.array([0.0, 0.0, 0.09]),
                       bp + 0.045 * outv, bp]
            target = targets[wp]
            if float(np.linalg.norm(target - lf)) < 0.02 and wp < 2:
                wp += 1
                print(f"SV waypoint -> {wp} at it={it}", flush=True)
                target = targets[wp]
            gapv = target - lf
            d = float(np.linalg.norm(gapv))
            tg = toggled()
            if it % 5 == 0 or tg:
                print(f"SV it={it} wp={wp} dist={d:.4f} btn={d_btn:.4f} "
                      f"tg={tg} ag={native_ag()}", flush=True)
            if tg:
                tgl_frame = -1
                print(f"TOGGLED via servo at it={it} ag={native_ag()}",
                      flush=True)
                break
            hist.append(d)
            step_size = 0.020 if d > 0.03 else 0.006
            _, wq_now = rob.eef_links["left"].get_position_orientation()
            R_now = __import__("scipy.spatial.transform", fromlist=["Rotation"]).Rotation.from_quat(_np(wq_now))
            _, rq_now = radio.get_position_orientation()
            R_rad_now = __import__("scipy.spatial.transform", fromlist=["Rotation"]).Rotation.from_quat(_np(rq_now))
            R_tgt = R_rad_now * R_gt_radio.inv() * R_gt_wrist
            w_err = (R_tgt * R_now.inv()).as_rotvec()
            w_step = np.clip(w_err, -0.12, 0.12)
            J6 = fetch_J6(best)
            twist = np.concatenate([gapv / (d + 1e-9) * step_size, 0.6 * w_step])
            dq = np.clip(np.linalg.pinv(J6) @ twist, -0.09, 0.09)
            if it % 10 == 0:
                print(f"ORN it={it} werr={np.linalg.norm(w_err):.3f}rad",
                      flush=True)
            q = q61()
            e_before = lfinger()
            base[7:14] = q[P["left"]["arm_qpos"]] + dq
            step_cmd(base)
            moved = lfinger() - e_before
            pred = J @ dq
            cosang = float(np.dot(moved, pred) /
                           (np.linalg.norm(moved) * np.linalg.norm(pred) + 1e-9))
            bad_cos = bad_cos + 1 if cosang < 0.3 else 0
            if bad_cos >= 3 and wp > 0:
                wp = max(0, wp - 1)
                bad_cos = 0
                print(f"SV back to wp={wp} at it={it}", flush=True)
            if it % 5 == 0:
                print(f"SVCHK it={it} cos={cosang:.2f} "
                      f"|moved|={np.linalg.norm(moved):.4f}", flush=True)
            if it % 4 == 0:
                frames.append(grab(0.5 * (lfinger() + button_pos())))
                imageio.imwrite(f"{OUT}/sv{it:03d}.png", frames[-1])
        for k2 in range(20):
            wrapper.env.step(base)
            if toggled():
                tgl_frame = -1
                print("TOGGLED in dwell", flush=True)
                break
    tp = _np(radio.get_position_orientation()[0])
    s = dict(toggled=bool(tgl_frame is not None), toggle_frame=tgl_frame,
             press0=press0, ag_intact=bool(native_ag()), ag_lost=ag_lost,
             lifted=bool(_np(radio.get_position_orientation()[0])[2]
                         > lift_z + 0.02))
    print("SUMMARY", json.dumps(s), flush=True)
    json.dump(s, open(f"{OUT}/rt34_d{a.demo}.json", "w"), indent=1)
    imageio.mimsave(f"{OUT}/rt34_chain.mp4", frames, fps=5)
    print("DONE", flush=True)
    os._exit(0)


main()

"""Approach factory: manufacture the MISSING segment — the pre-contact approach.

The grasp+transport clips start at closure+4..8, hand already on the radio: they
repair closure/hold/lift but contain no approach. In the demos the rig slid the
radio ~25cm INTO the closing hand, so the human tape never shows a hand crossing
those centimeters either. This factory produces it honestly:

  restore closure-K (radio at rest, pre-pull; gripper open)
  -> RCAL3 (3-probe native-J row select, + angular-block probe)
  -> servo the open right hand to the grasp pose = demo's certified post-pull
     hand-relative-to-radio geometry applied to the radio's REST pose
     (6-DOF DLS: position primary, orientation HELD at its start value)
  -> close gripper, streak-gated verified weld (or native AG)
  -> servo the HELD radio to the demo's closure+6 radio pose (the pull vector,
     now caused by the hand)  -> proven demo transport replay from closure+6.
Exports cmds + meta (weld_k = absolute cmd index) to /root/factory_clips_approach.
Run:  OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNIGIBSON_HEADLESS=1 \
      python -u /root/factory_approach.py --demo 20
"""
import argparse
import json
import os

import numpy as np

os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")
OUT = "/root/factory_clips_approach"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", type=int, required=True)
    ap.add_argument("--K", type=int, default=30, help="restore at closure-K")
    ap.add_argument("--tol", type=float, default=0.006)
    ap.add_argument("--settle", type=int, default=40)
    a = ap.parse_args()

    import h5py
    import torch as th
    import omnigibson as og
    from omnigibson.macros import gm
    gm.HEADLESS = True; gm.ENABLE_TRANSITION_RULES = False
    from omnigibson.envs.hdf5_data_wrapper import HDF5PlaybackWrapper
    from reverse_curriculum_collect import EVAL_PROPRIO_KEYS, restore_to_frame
    from skill_env_wrapper import _np, P, A_TORSO
    from scipy.spatial.transform import Rotation as R

    os.makedirs(OUT, exist_ok=True)
    man = json.load(open(f"/root/snapshot_bank_v21/manifest_d{a.demo}.json"))
    G = next(r for r in man["entries"] if str(r["stage"]) == "G")
    closure = G["grasp_closure_frame"]
    lift_z = G["lift_z"]
    press = [int(r["frame"]) for r in man["entries"] if r.get("family") == "press"]
    horizon_end = (max(press) - 60) if press else (closure + 280)
    t0off = 6
    try:
        fm = json.load(open(f"/root/factory_clips/d{a.demo:03d}_meta.json"))
        if fm.get("ok"):
            t0off = int(fm["t0"] - fm["closure"])
    except FileNotFoundError:
        pass
    t_post = closure + t0off

    wrapper = HDF5PlaybackWrapper.create_from_hdf5(
        input_path=f"/root/rawdemos/task-0000/episode_{a.demo:08d}.hdf5",
        output_path=f"/root/fap_tmp_{a.demo}.hdf5",
        robot_obs_modalities=("proprio", "rgb", "depth_linear"), robot_proprio_keys=EVAL_PROPRIO_KEYS)
    radio = [o for o in wrapper.scene.objects if "radio" in o.name.lower()][0]
    rob = wrapper.scene.robots[0]
    eefR = rob.eef_links["right"]
    armR_idx = np.asarray(_np(rob.arm_control_idx["right"]), int)

    with h5py.File(f"/root/rawdemos/task-0000/episode_{a.demo:08d}.hdf5", "r") as f:
        key = sorted(k for k in f["data"].keys() if k.startswith("demo_"))[0]
        acts = f[f"data/{key}/action"][:]
    horizon_end = min(horizon_end, len(acts) - 1)

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
        return [_np(l.get_position_orientation()[0]) for l in rob.finger_links["right"]]

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

    def poseR():
        p, q_ = eefR.get_position_orientation()
        return _np(p).copy(), R.from_quat(_np(q_))

    def radio_pose():
        p, q_ = radio.get_position_orientation()
        return _np(p).copy(), R.from_quat(_np(q_))

    import imageio
    cam = og.sim.viewer_camera
    FILM = f"/root/rt_approach_film/d{a.demo:03d}"
    os.makedirs(FILM, exist_ok=True)

    def grab(tag):
        rp = _np(radio.get_position_orientation()[0]); ft = poseR()[0]
        target = 0.5 * (rp + ft); pos = target + np.array([0.55, -0.5, 0.35])
        f = target - pos; f /= np.linalg.norm(f); up = np.array([0.0, 0.0, 1.0])
        r = np.cross(f, up); r /= np.linalg.norm(r); u = np.cross(r, f)
        m = np.stack([r, u, -f], axis=1)
        w = np.sqrt(max(1e-9, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
        q = np.array([(m[2, 1] - m[1, 2]) / (4 * w), (m[0, 2] - m[2, 0]) / (4 * w),
                      (m[1, 0] - m[0, 1]) / (4 * w), w])
        cam.set_position_orientation(th.as_tensor(pos, dtype=th.float32),
                                     th.as_tensor(q, dtype=th.float32))
        for _ in range(2):
            og.sim.render()
        o, _ = cam.get_obs()
        imageio.imwrite(f"{FILM}/{tag}.png", np.asarray(o["rgb"])[..., :3].astype(np.uint8))

    cmds_log = []
    REC = {k: [] for k in ("p", "act", "hz", "hd", "lz", "ld", "rz", "rd", "rp", "bp")}

    def get_obs():
        obs = wrapper.env.get_obs()[0]
        o = {"pro": None, "zed_rgb": None, "zed_dep": None, "l_rgb": None,
             "l_dep": None, "r_rgb": None, "r_dep": None}
        def walk(node, path=""):
            if isinstance(node, dict):
                for k, v in node.items():
                    kl = str(k).lower(); pl = path + "/" + kl
                    if "proprio" in kl:
                        o["pro"] = _np(v).reshape(-1)
                    elif isinstance(v, dict):
                        walk(v, path + "/" + str(k))
                    else:
                        c = ("zed" if "zed" in pl else "l" if "left" in pl else
                             "r" if "right" in pl else None)
                        if c:
                            if kl.endswith("rgb") or kl == "rgb":
                                o[f"{c}_rgb"] = _np(v)
                            elif "depth" in kl:
                                o[f"{c}_dep"] = _np(v)
        walk(obs)
        return o

    def capture(cmd):
        o = get_obs()
        if o["pro"] is None or o["zed_rgb"] is None:
            return
        REC["p"].append(o["pro"].astype(np.float32)); REC["act"].append(np.asarray(cmd, np.float32).copy())
        REC["hz"].append(o["zed_rgb"][..., :3].astype(np.uint8))
        REC["hd"].append(np.zeros((2, 2), np.float16) if o["zed_dep"] is None else o["zed_dep"].astype(np.float16))
        for c, kz, kd in (("l", "lz", "ld"), ("r", "rz", "rd")):
            REC[kz].append(np.zeros((2, 2, 3), np.uint8) if o[f"{c}_rgb"] is None else o[f"{c}_rgb"][..., :3].astype(np.uint8))
            REC[kd].append(np.zeros((2, 2), np.float16) if o[f"{c}_dep"] is None else o[f"{c}_dep"].astype(np.float16))
        rp_, rq_ = radio.get_position_orientation(); bp_, bq_ = rob.get_position_orientation()
        REC["rp"].append(np.concatenate([_np(rp_), _np(rq_)]).astype(np.float32))
        REC["bp"].append(np.concatenate([_np(bp_), _np(bq_)]).astype(np.float32))

    CAP = [False]  # capture switched on at the pre-pull restore (probes excluded)

    def step_cmd(cmd):
        if CAP[0]:
            capture(cmd)
        q = q61()
        for _ in range(a.settle):
            wrapper.env.step(cmd)
            cmds_log.append(np.asarray(cmd, np.float32).copy())
            q = q61()
            if (np.abs(q[P["left"]["arm_qpos"]] - cmd[7:14]).max() < a.tol
                    and np.abs(q[P["right"]["arm_qpos"]] - cmd[15:22]).max() < a.tol
                    and np.abs(q[P["trunk_qpos"]] - cmd[A_TORSO]).max() < a.tol):
                break

    # ---- GT: certified post-pull grasp geometry (hand relative to radio) and the
    # demo's closure+t0off radio pose (transport handoff) -----------------------
    restore_to_frame(wrapper, 0, t_post)
    pE, RE = poseR()
    pRad, RRad = radio_pose()
    rel_p = RRad.inv().apply(pE - pRad)
    rel_R = RRad.inv() * RE
    rel_f = [RRad.inv().apply(f - pRad) for f in links()]
    rel_fm = np.mean(rel_f, axis=0)
    gt_gap = float(np.linalg.norm(rel_f[0] - rel_f[1]))
    print(f"GT fingers in radio frame: {[np.round(f, 3).tolist() for f in rel_f]} "
          f"midpoint {np.round(rel_fm, 3).tolist()} gap {gt_gap:.3f}", flush=True)
    p_rad_post = pRad.copy()
    q_post = q61()
    print(f"GT post-pull: |hand-radio| {np.linalg.norm(pE - pRad):.3f} m", flush=True)

    # ---- restore PRE-pull: radio at rest, hand short of it ---------------------
    t_pre = closure - a.K
    restore_to_frame(wrapper, 0, t_pre)
    rob._refresh_rigid_contact_view()
    pRest, RRest = radio_pose()
    pull = p_rad_post - pRest
    tgt_p = pRest + RRest.apply(rel_p)
    tgt_R = RRest * rel_R
    pE0, RE0 = poseR()
    gap0 = float(np.linalg.norm(tgt_p - pE0))
    print(f"PRE t={t_pre}: pull |{np.linalg.norm(pull):.3f}| m; hand->grasp-pose gap "
          f"{gap0:.3f} m; radio z {pRest[2]:.3f}", flush=True)
    if gap0 > 0.55:
        # arm+trunk reach envelope (~0.5 m of approach): beyond it the demo needs base
        # motion first (d10: pull 1.04 m). Log and skip — a base-drive variant is the fix.
        json.dump(dict(demo=a.demo, t0=t_pre, closure=closure, K=a.K, gap0=round(gap0, 4),
                       pull=np.round(pull, 4).tolist(), ok=False, skip="REACH"),
                  open(f"{OUT}/d{a.demo:03d}_meta.json", "w"), indent=1)
        print(f"RESULT d{a.demo} SKIP_REACH gap0={gap0:.3f} pull={np.linalg.norm(pull):.3f}", flush=True)
        os._exit(0)

    q = q61()
    hold = np.asarray(acts[t_pre], np.float32).copy()
    hold[0:3] = 0.0
    hold[7:14] = q[P["left"]["arm_qpos"]]
    hold[15:22] = q[P["right"]["arm_qpos"]]
    hold[A_TORSO] = q[P["trunk_qpos"]]
    hold[22] = 1.0  # open

    # ---- RCAL3: 3-probe native-J row/block select (position), + angular block --
    pairs = []
    qb0 = q61()[P["right"]["arm_qpos"]].copy()
    for jidx in (16, 18, 20):
        qs = q61()[P["right"]["arm_qpos"]].copy()
        es, Rs = poseR()
        pr = hold.copy(); pr[15:22] = qs; pr[jidx] += 0.05
        step_cmd(pr)
        e1, R1 = poseR()
        pairs.append((q61()[P["right"]["arm_qpos"]] - qs, e1 - es,
                      (R1 * Rs.inv()).as_rotvec()))
        un = pr.copy(); un[15:22] = qs
        step_cmd(un)
    Jf0 = _np(rob.get_jacobian())
    best, bsc = None, -2.0
    for row in range(Jf0.shape[0]):
        for blk in (0, 3):
            sc, ok = [], True
            for dq, de, _ in pairs:
                pred = Jf0[row, blk:blk + 3, :][:, armR_idx] @ dq
                n = np.linalg.norm(pred) * np.linalg.norm(de)
                if n < 1e-12:
                    ok = False; break
                c = float(np.dot(pred, de) / n)
                m = min(np.linalg.norm(pred), np.linalg.norm(de)) / (
                    max(np.linalg.norm(pred), np.linalg.norm(de)) + 1e-12)
                sc.append(c * m)
            if ok and min(sc) > bsc:
                bsc, best = min(sc), (row, blk)
    row, pblk = best
    ablk = 3 if pblk == 0 else 0
    acos = []
    for dq, _, dr in pairs:
        pred = Jf0[row, ablk:ablk + 3, :][:, armR_idx] @ dq
        n = np.linalg.norm(pred) * np.linalg.norm(dr)
        acos.append(float(np.dot(pred, dr) / n) if n > 1e-12 else 0.0)
    asign = 1.0 if np.mean(acos) >= 0 else -1.0
    use_orn = min(abs(c) for c in acos) > 0.5
    rest = hold.copy(); rest[15:22] = qb0; step_cmd(rest)
    print(f"RCAL3 sel={best} worst={bsc:.2f} | angular blk {ablk} cos {np.round(acos, 2).tolist()} "
          f"sign {asign:+.0f} use_orn={use_orn}", flush=True)

    # trunk block: probe the 4 trunk action channels (position + rotation of
    # the right eef) — the reach lever (R51): the radio's REST spot is ~35cm
    # farther than where the human's hand stopped
    Tp, Ta = [], []
    for jch in range(3, 7):
        qs = q61(); e0, R0 = poseR()
        pr = hold.copy(); pr[15:22] = qs[P["right"]["arm_qpos"]]
        pr[A_TORSO] = qs[P["trunk_qpos"]]; pr[jch] += 0.05
        step_cmd(pr); e1, R1 = poseR()
        Tp.append((e1 - e0) / 0.05); Ta.append((R1 * R0.inv()).as_rotvec() / 0.05)
        un = pr.copy(); un[jch] -= 0.05; step_cmd(un)
    Jtp, Jta = np.stack(Tp, 1), np.stack(Ta, 1)
    print(f"TCAL trunk pos-col norms {[round(float(np.linalg.norm(c)), 3) for c in Tp]}", flush=True)

    def J11():
        Jf = _np(rob.get_jacobian())
        Jp = np.concatenate([Jf[row, pblk:pblk + 3, :][:, armR_idx], Jtp], axis=1)
        Ja = np.concatenate([asign * Jf[row, ablk:ablk + 3, :][:, armR_idx], Jta], axis=1)
        return Jp, Ja

    def servo(target_p, target_R, budget, tag, stride=0.012, w_orn=0.3, done=0.015):
        """6-DOF DLS servo of the right eef; orientation held to target_R."""
        best_d = 9.9
        for it in range(budget):
            pE, RE = poseR()
            d = float(np.linalg.norm(target_p - pE)); best_d = min(best_d, d)
            if d < done:
                oerr = float(np.linalg.norm((target_R * RE.inv()).as_rotvec()))
                print(f"{tag} reached it={it} d={d:.4f} orn_err={oerr:.3f} rad", flush=True)
                return True, best_d
            if not native_ag() and tag == "CARRY":
                print(f"{tag} AG_LOST it={it}", flush=True)
                return False, best_d
            Jp, Ja = J11()
            v = (target_p - pE) / (d + 1e-9) * min(stride, d)
            w = (target_R * RE.inv()).as_rotvec() if use_orn else np.zeros(3)
            w = np.clip(w, -0.05, 0.05) * w_orn
            Jst = np.concatenate([Jp, np.sqrt(w_orn) * Ja if use_orn else 0 * Ja], axis=0)
            rhs = np.concatenate([v, np.sqrt(w_orn) * w if use_orn else np.zeros(3)])
            lam = 0.01
            dq = Jst.T @ np.linalg.solve(Jst @ Jst.T + lam * np.eye(6), rhs)
            dq = np.concatenate([np.clip(dq[:7], -0.08, 0.08), np.clip(dq[7:], -0.03, 0.03)])
            q_ = q61()
            cmd = hold.copy()
            cmd[15:22] = q_[P["right"]["arm_qpos"]] + dq[:7]
            cmd[A_TORSO] = q_[P["trunk_qpos"]] + dq[7:]
            step_cmd(cmd)
            if tag in ("STAGE", "APPROACH", "PUSH", "ALIGN"):
                HON["n"] += 1
                _d = float(np.linalg.norm(radio_pose()[0] - HON["radio_rest"]))
                HON["max_disp"] = max(HON["max_disp"], _d)
                if HON["first_contact"] is None:
                    if contact():
                        HON["first_contact"] = f"{tag}:{it}"
                    else:
                        HON["pre_disp"] = max(HON["pre_disp"], _d)
            moved = poseR()[0] - pE
            cos = float(np.dot(moved, target_p - pE) /
                        (np.linalg.norm(moved) * np.linalg.norm(target_p - pE) + 1e-9))
            if it % 10 == 0:
                print(f"{tag} it={it} d={d:.4f} cos={cos:.2f} trunk_dq={np.linalg.norm(dq[7:]):.3f} ag={native_ag()}", flush=True)
                grab(f"{tag.lower()}{it:03d}")
        print(f"{tag} budget exhausted best={best_d:.4f}", flush=True)
        return False, best_d

    # ---- phase A: honest approach, gripper open --------------------------------
    grab("pre")
    CAP[0] = True
    HON = {"radio_rest": radio_pose()[0].copy(), "max_disp": 0.0, "pre_disp": 0.0, "first_contact": None, "n": 0}
    stage_p = tgt_p + np.array([0.0, 0.0, 0.10])
    ok_s, best_s = servo(stage_p, tgt_R, 150, "STAGE")
    print(f"STAGE ok={ok_s} best={best_s:.3f}", flush=True)
    # ORIENT v7 (gentle): converge the wrist to the certified grasp attitude at the
    # staging point with small rotation steps and a firm position hold. v6's
    # 0.12 rad/step + 0.10 dq clip swung the arm 42cm off the point.
    for it in range(150):
        pE, RE = poseR()
        oerr_v = (tgt_R * RE.inv()).as_rotvec(); oerr = float(np.linalg.norm(oerr_v))
        pdrift = float(np.linalg.norm(pE - stage_p))
        if it % 10 == 0:
            print(f"ORIENT it={it} orn_err={oerr:.3f} pos_drift={pdrift:.4f}", flush=True)
        if oerr < 0.05 and pdrift < 0.02:
            print(f"ORIENT converged it={it} orn_err={oerr:.3f} pos_drift={pdrift:.4f}", flush=True)
            break
        if pdrift > 0.08:
            print(f"ORIENT abort: drift {pdrift:.3f} m at it={it} (orn_err {oerr:.3f})", flush=True)
            break
        Jp, Ja = J11()
        v = np.clip(stage_p - pE, -0.02, 0.02) * 2.0          # position hold (gain 2)
        w = oerr_v / (oerr + 1e-9) * min(0.03, oerr)           # <= 0.03 rad per step
        Jst = np.concatenate([2.0 * Jp, Ja], axis=0); rhs = np.concatenate([2.0 * v, w])
        dq = Jst.T @ np.linalg.solve(Jst @ Jst.T + 0.02 * np.eye(6), rhs)
        dq = np.concatenate([np.clip(dq[:7], -0.04, 0.04), np.clip(dq[7:], -0.015, 0.015)])
        q_ = q61(); cmd = hold.copy()
        cmd[15:22] = q_[P["right"]["arm_qpos"]] + dq[:7]; cmd[A_TORSO] = q_[P["trunk_qpos"]] + dq[7:]
        step_cmd(cmd)
    grab("orient_end")
    ok_s2, best_s2 = servo(stage_p, tgt_R, 60, "RESTAGE")   # re-center after orienting
    ok_a, best_a = servo(tgt_p, tgt_R, 120, "APPROACH", stride=0.008, w_orn=0.6)
    grab("approach_end")
    if not ok_a:
        print(f"RESULT d{a.demo} APPROACH_FAILED best={best_a:.3f}", flush=True)
        os._exit(0)
    approach_end = len(cmds_log)

    # ---- phase B0: compliance push — settle the open tines down onto/around the
    # rail (v3 "reached" at 1.39cm looked like tine tips resting ON the rail)
    okp, _ = servo(tgt_p + np.array([0.0, 0.0, -0.015]), tgt_R, 25, "PUSH", stride=0.006, done=0.008)
    grab("push_end")
    pR_, RR_ = radio_pose()
    fl = [RR_.inv().apply(f - pR_) for f in links()]
    print(f"PUSH ok={okp} finger tips in radio frame: {[np.round(f, 3).tolist() for f in fl]} "
          f"| certified rel_p {np.round(rel_p, 3).tolist()}", flush=True)

    # ---- phase B1: ALIGN the finger midpoint onto the certified one (radio frame)
    for k3 in range(4):
        pR_, RR_ = radio_pose()
        fm_now = RR_.inv().apply(np.mean(links(), axis=0) - pR_)
        delta_w = RR_.apply(rel_fm - fm_now)
        print(f"ALIGN pass {k3}: finger-midpoint err {np.round(rel_fm - fm_now, 3).tolist()} "
              f"|{np.linalg.norm(delta_w):.4f}| m", flush=True)
        if np.linalg.norm(delta_w) < 0.004:
            break
        pE_, RE_ = poseR()
        servo(pE_ + delta_w, tgt_R, 30, "ALIGN", stride=0.006, done=0.004)
    grab("align_end")

    print(f"APPROACH_HONESTY radio_disp_precontact={HON['pre_disp']:.4f} m radio_disp_max={HON['max_disp']:.4f} m "
          f"first_contact={HON['first_contact']} servo_steps={HON['n']}", flush=True)

    # ---- phase B: close + streak-gated verified weld ---------------------------
    q = q61()
    hold[15:22] = q[P["right"]["arm_qpos"]]
    hold[A_TORSO] = q[P["trunk_qpos"]]
    hold[22] = -1.0
    green, gap, weld_k = 0, 0, None
    capture(hold)
    for k in range(800):  # late-starting streaks need room (v4: 213 by k=400)
        wrapper.env.step(hold)
        cmds_log.append(hold.copy())
        c_, i_ = contact(), inhand()
        if k < 60 and k % 5 == 0:
            fl_ = links()
            print(f"CLOSE k={k} contact={c_} inhand={i_} finger_gap={np.linalg.norm(fl_[0] - fl_[1]):.3f} (GT {gt_gap:.3f})", flush=True)
        if k in (0, 20, 40, 80, 160):
            grab(f"weld{k:03d}")
        if c_ and i_:
            green += 1; gap = 0
        else:
            gap += 1
            if gap > 4:
                green = 0
        if native_ag():
            weld_k = len(cmds_log) - 1
            print(f"NATIVE_AG k={k} (abs {weld_k})", flush=True)
            break
        if k >= 320 and green >= 296:
            cp = th.as_tensor(np.mean(links(), axis=0), dtype=th.float32)
            rob._establish_grasp(radio, radio.root_link_name, "right", cp, "FixedJoint")
            weld_k = len(cmds_log) - 1
            print(f"VERIFIED_WELD k={k} streak={green} (abs {weld_k})", flush=True)
            break
    if not native_ag():
        print(f"RESULT d{a.demo} WELD_FAILED streak={green}", flush=True)
        os._exit(0)

    # ---- phase C: carry the held radio to the demo's post-pull radio pose ------
    pE, RE = poseR()
    carry_tgt = pE + pull  # hand displacement == the pull vector
    ok_c, best_c = servo(carry_tgt, RE, 200, "CARRY", stride=0.010)
    print(f"CARRY done ok={ok_c} best={best_c:.3f} radio->post |"
          f"{np.linalg.norm(radio_pose()[0] - p_rad_post):.3f}| m", flush=True)
    # phase C2: return the trunk to the demo's closure+t0off posture while the
    # arm compensates to hold the eef (so the absolute-target transport replay
    # starts from the demo's configuration, radio still in hand)
    tr_demo = q_post[P["trunk_qpos"]]
    pE_hold = poseR()[0].copy()
    for k2 in range(60):
        q_ = q61()
        dqt = np.clip(tr_demo - q_[P["trunk_qpos"]], -0.02, 0.02)
        if np.abs(dqt).max() < 1e-3:
            break
        Jp, _ = J11()
        pred = Jp[:, 7:] @ dqt
        Ja7 = Jp[:, :7]
        dqa = -Ja7.T @ np.linalg.solve(Ja7 @ Ja7.T + 0.01 * np.eye(3), pred)
        cmd = hold.copy()
        cmd[15:22] = q_[P["right"]["arm_qpos"]] + np.clip(dqa, -0.08, 0.08)
        cmd[A_TORSO] = q_[P["trunk_qpos"]] + dqt
        step_cmd(cmd)
        if k2 % 10 == 0:
            print(f"POSTURE k={k2} trunk_err={np.abs(tr_demo - q61()[P['trunk_qpos']]).max():.3f} "
                  f"eef_drift={np.linalg.norm(poseR()[0] - pE_hold):.3f} ag={native_ag()}", flush=True)
    grab("posture_end")
    carry_end = len(cmds_log)

    # ---- phase D: proven transport replay from closure+t0off -------------------
    lifted, ag_lost = False, None
    for t in range(t_post + 1, horizon_end):
        cmd = np.asarray(acts[t], np.float32).copy()
        cmd[22] = -1.0
        step_cmd(cmd)
        tp = _np(radio.get_position_orientation()[0])
        lifted = lifted or tp[2] > lift_z + 0.05
        if not native_ag() and ag_lost is None:
            ag_lost = t
            print(f"AG_LOST t={t}", flush=True)
            break
        if t % 40 == 0:
            print(f"CH t={t} z={tp[2]:.3f}", flush=True)
    honest = HON['pre_disp'] <= 0.02 and HON['max_disp'] <= 0.10  # undisturbed until touched; no bulldozing
    if not honest:
        print(f"HONESTY_REJECT radio moved {HON['max_disp']:.3f} m before closure", flush=True)
    ok = bool(native_ag() and lifted and ag_lost is None and honest)
    meta = dict(demo=a.demo, t0=t_pre, closure=closure, K=a.K, t_post=t_post,
                weld_k=weld_k, streak=green, approach_end=approach_end,
                carry_end=carry_end, gap0=round(gap0, 4), pull=np.round(pull, 4).tolist(),
                carry_ok=bool(ok_c), horizon_end=horizon_end, lifted=bool(lifted),
                radio_disp_approach=round(HON['max_disp'], 4), radio_disp_precontact=round(HON['pre_disp'], 4), first_contact=HON['first_contact'], honest=bool(honest),
                ag_intact=bool(native_ag()), ag_lost=ag_lost, ok=ok,
                n_cmds=len(cmds_log), kind="approach",
                radio_z_end=round(float(_np(radio.get_position_orientation()[0])[2]), 4))
    if ok:
        np.savez_compressed(f"{OUT}/d{a.demo:03d}_approach.npz",
                            cmds=np.stack(cmds_log), meta=json.dumps(meta))
        np.savez_compressed(
            f"/root/factory_obs2/rac_{a.demo}_200.npz",
            proprio=np.stack(REC["p"]), actions=np.stack(REC["act"]),
            head_rgb=np.stack(REC["hz"]), head_depth=np.stack(REC["hd"]),
            left_rgb=np.stack(REC["lz"]), left_depth=np.stack(REC["ld"]),
            right_rgb=np.stack(REC["rz"]), right_depth=np.stack(REC["rd"]),
            objpose_radio_89=np.stack(REC["rp"]), base_pose=np.stack(REC["bp"]),
            radio_rest_z=np.float64(pRest[2]), success=np.bool_(True),
            meta=json.dumps({**meta, "episode": "approach_single_pass", "n_obs": len(REC["p"])}))
        print(f"OBS_SAVED rac_{a.demo}_200.npz ({len(REC['p'])} steps)", flush=True)
    json.dump(meta, open(f"{OUT}/d{a.demo:03d}_meta.json", "w"), indent=1)
    print("RESULT", json.dumps(meta), flush=True)
    os._exit(0)


main()

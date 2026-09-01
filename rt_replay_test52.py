"""Round 52: IN-HAND PRESS — the human playbook. Hold the radio in the right
hand (verified weld intact), replay the demo's left-arm + trunk press approach
(state-targeted, collision-free by construction), then finish the last
centimeters with an 11-DOF LEFT-arm + trunk servo onto the LIVE button of the
HELD radio. No set-down, no release: the entire set-down bug family (edge
release, tilt-at-release tip-over — five layers excavated on d190/d260) is out
of the loop.

vs the failed in-hand rounds 21–38: (a) the trunk was frozen then — here the
demo's own trunk trajectory is replayed through the approach and the servo gets
4 trunk channels; (b) press targets were demo-frame retargets (ghost button) —
here the button position is read live from the held radio each iteration;
(c) the trunk block is probed RELATIVELY (d(fingertip − button)/dq): trunk
motion moves the held radio too, so only the relative column is honest.

Success: toggled_on flips while the weld holds (ag_intact) = demo-style
completion. Run:
  OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNIGIBSON_HEADLESS=1 \
      python -u /root/rt_replay_test52.py --demo 30
"""
import argparse
import json
import os

import numpy as np

os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")

OUT = "/root/rt52_film"


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
    globals()["OUT"] = f"/root/rt52_film/d{a.demo:03d}"

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
        output_path=f"/root/rtt52_tmp_{a.demo}.hdf5",
        robot_obs_modalities=("proprio",), robot_proprio_keys=EVAL_PROPRIO_KEYS)
    radio = [o for o in wrapper.scene.objects if "radio" in o.name.lower()][0]
    rob = wrapper.scene.robots[0]
    cam = og.sim.viewer_camera
    from omnigibson.object_states import ToggledOn
    toggled = lambda: bool(radio.states[ToggledOn].get_value())  # noqa: E731

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

    # phase 0: GT capture — the demo's left-arm + trunk press-approach states
    qLp, trp = {}, {}
    for t in range(press0 - 120, min(press0 + 41, len(acts))):
        restore_to_frame(wrapper, 0, t)
        q = q61()
        qLp[t] = q[P["left"]["arm_qpos"]].copy()
        trp[t] = q[P["trunk_qpos"]].copy()
    print("GT press-window captured", len(qLp), "frames", flush=True)

    # phase 1: verified weld (factory-meta anchor)
    t0 = closure + 6
    try:
        fm = json.load(open(f"/root/factory_clips/d{a.demo:03d}_meta.json"))
        if fm.get("ok") and fm.get("closure") is not None:
            t0 = closure + int(fm["t0"] - fm["closure"])
            print(f"ANCHOR from factory meta: t0off={t0 - closure}", flush=True)
    except FileNotFoundError:
        pass
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
        if native_ag():
            print(f"NATIVE_AG at k={k}", flush=True)
            break
        if k >= 320 and green >= 296:
            cp = th.as_tensor(np.mean(links(), axis=0), dtype=th.float32)
            rob._establish_grasp(radio, radio.root_link_name, "right", cp,
                                 "FixedJoint")
            print(f"VERIFIED_WELD at k={k} streak={green}", flush=True)
            break
    if not native_ag():
        print("WELD_FAILED", flush=True)
        os._exit(0)

    def button_pos():
        return _np(radio.states[ToggledOn].link.get_position_orientation()[0])

    def lfinger():
        return np.mean([_np(l.get_position_orientation()[0])
                        for l in rob.finger_links["left"]], axis=0)

    # phase 2: chain replay to press0-60, left arm + trunk on demo state targets
    tgl_frame, ag_lost = None, None
    servo_start = press0 - 60
    for t in range(t0 + 1, servo_start):
        cmd = np.asarray(acts[t], np.float32).copy()
        cmd[22] = -1.0
        if t in qLp:
            cmd[7:14] = qLp[t]
            cmd[A_TORSO] = trp[t]
        step_cmd(cmd)
        tp = _np(radio.get_position_orientation()[0])
        if not native_ag() and ag_lost is None:
            ag_lost = t
            print(f"AG_LOST at frame {t} (rel_press {t - press0})", flush=True)
        tg = toggled()
        if tg and tgl_frame is None:
            tgl_frame = t
            print(f"TOGGLED during replay at frame {t}", flush=True)
        if t % 10 == 0:
            print(f"CH t={t} z={tp[2]:.3f} ag={native_ag()} tg={tg}", flush=True)
        if t % 6 == 0:
            img = grab(0.5 * (np.mean(links(), axis=0) + tp))
            frames.append(img)
            imageio.imwrite(f"{OUT}/f{t:04d}.png", img)
        if tgl_frame is not None and t > tgl_frame + 15:
            break

    # phase 3IH: 11-DOF LEFT-arm + trunk servo onto the live button, in hand
    best_btn = 9.9
    pre_toggled = toggled()
    if tgl_frame is None:
        eefL = rob.eef_links["left"]
        armL_idx = np.asarray(_np(rob.arm_control_idx["left"]), int)
        qf = q61()
        holdI = np.asarray(acts[servo_start], np.float32).copy()
        holdI[0:3] = 0.0
        holdI[7:14] = qf[P["left"]["arm_qpos"]]
        holdI[15:22] = qf[P["right"]["arm_qpos"]]
        holdI[A_TORSO] = qf[P["trunk_qpos"]]
        holdI[22] = -1.0
        print(f"IH press start: pre_toggled={pre_toggled} ag={native_ag()}",
              flush=True)

        def poseL():
            return _np(eefL.get_position_orientation()[0])

        # LCAL3: 3-probe row select for the LEFT eef in the native jacobian
        pairs = []
        qb0 = q61()[P["left"]["arm_qpos"]].copy()
        for jidx in (8, 10, 12):
            qs = q61()[P["left"]["arm_qpos"]].copy()
            es = poseL().copy()
            probeC = holdI.copy()
            probeC[7:14] = qs
            probeC[jidx] += 0.05
            step_cmd(probeC)
            pairs.append((q61()[P["left"]["arm_qpos"]] - qs, poseL() - es))
            undoC = probeC.copy()
            undoC[7:14] = qs
            step_cmd(undoC)
        Jf0 = _np(rob.get_jacobian())
        bestC, bcC = None, -2.0
        for rowC in range(Jf0.shape[0]):
            for blkC in (0, 3):
                scores, ok = [], True
                for dqC, deC in pairs:
                    predC = Jf0[rowC, blkC:blkC + 3, :][:, armL_idx] @ dqC
                    nC = np.linalg.norm(predC) * np.linalg.norm(deC)
                    if nC < 1e-12:
                        ok = False
                        break
                    cC = float(np.dot(predC, deC) / nC)
                    mC = min(np.linalg.norm(predC), np.linalg.norm(deC)) / (
                        max(np.linalg.norm(predC), np.linalg.norm(deC)) + 1e-12)
                    scores.append(cC * mC)
                if ok and min(scores) > bcC:
                    bcC, bestC = min(scores), (rowC, blkC)
        rest = holdI.copy()
        rest[7:14] = qb0
        step_cmd(rest)
        print(f"LCAL3 sel={bestC} worst_score={bcC:.2f}", flush=True)
        LSEL = bestC

        def fetch_JL():
            Jf = _np(rob.get_jacobian())
            return Jf[LSEL[0], LSEL[1]:LSEL[1] + 3, :][:, armL_idx]

        # trunk block: RELATIVE probe — trunk moves the held radio (and its
        # button) too, so only d(fingertip - button) is an honest column
        Tcols = []
        for jch in range(3, 7):
            qs = q61()
            e0 = lfinger() - button_pos()
            pr = holdI.copy()
            pr[7:14] = qs[P["left"]["arm_qpos"]]
            pr[A_TORSO] = qs[P["trunk_qpos"]]
            pr[jch] = pr[jch] + 0.05
            step_cmd(pr)
            Tcols.append(((lfinger() - button_pos()) - e0) / 0.05)
            un = pr.copy()
            un[jch] = un[jch] - 0.05
            step_cmd(un)
        Jt = np.stack(Tcols, axis=1)
        print(f"TCALrel norms: {[round(float(np.linalg.norm(c)), 3) for c in Tcols]}",
              flush=True)

        def fetch_J11():
            return np.concatenate([fetch_JL(), Jt], axis=1)

        wp2 = 0
        for it in range(300):
            bp = button_pos()
            rc = _np(radio.get_position_orientation()[0])
            nrm = bp - rc
            nrm = nrm / (np.linalg.norm(nrm) + 1e-9)
            lf = lfinger()
            d_btn = float(np.linalg.norm(bp - lf))
            best_btn = min(best_btn, d_btn)
            targets = [bp + 0.05 * nrm, bp - 0.012 * nrm]
            target = targets[wp2]
            dtn = float(np.linalg.norm(target - lf))
            if dtn < 0.02 and wp2 < 1:
                wp2 += 1
                print(f"IH waypoint -> {wp2} at it={it}", flush=True)
                continue
            if toggled() and not pre_toggled:
                tgl_frame = -52
                print(f"TOGGLED by IN-HAND press at it={it} ag={native_ag()}",
                      flush=True)
                break
            if not native_ag() and ag_lost is None:
                ag_lost = -it
                print(f"AG_LOST during press at it={it}", flush=True)
            e_b = lf
            dq = np.clip(np.linalg.pinv(fetch_J11())
                         @ ((target - lf) / (dtn + 1e-9) * min(0.015, dtn)),
                         -0.09, 0.09)
            q_ = q61()
            cmd = holdI.copy()
            cmd[7:14] = q_[P["left"]["arm_qpos"]] + dq[:7]
            cmd[A_TORSO] = q_[P["trunk_qpos"]] + np.clip(dq[7:11], -0.04, 0.04)
            step_cmd(cmd)
            moved = lfinger() - e_b
            cos2 = float(np.dot(moved, target - e_b) /
                         (np.linalg.norm(moved)
                          * np.linalg.norm(target - e_b) + 1e-9))
            if it % 10 == 0:
                print(f"IH it={it} wp={wp2} dist={dtn:.4f} btn={d_btn:.4f} "
                      f"cos={cos2:.2f} trunk_dq={np.linalg.norm(dq[7:11]):.3f} "
                      f"ag={native_ag()} tg={toggled()}", flush=True)
                img = grab(0.5 * (lfinger() + button_pos()))
                frames.append(img)
                imageio.imwrite(f"{OUT}/ih{it:03d}.png", img)
        print(f"IH done: best_btn={best_btn:.4f} toggled={toggled()} "
              f"ag={native_ag()}", flush=True)
        if toggled() and tgl_frame is None:
            tgl_frame = -52

    s = dict(toggled=bool(tgl_frame is not None), toggle_frame=tgl_frame,
             press0=press0, ag_intact=bool(native_ag()), ag_lost=ag_lost,
             best_btn=float(best_btn), style="in_hand",
             lifted=bool(_np(radio.get_position_orientation()[0])[2]
                         > lift_z + 0.02))
    print("SUMMARY", json.dumps(s), flush=True)
    json.dump(s, open(f"{OUT}/rt52_d{a.demo}.json", "w"), indent=1)
    imageio.mimsave(f"{OUT}/rt52_chain.mp4", frames, fps=5)
    print("DONE", flush=True)
    os._exit(0)


main()

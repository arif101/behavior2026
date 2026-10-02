"""FinishFactoryWrapper (2026-10-01) — FINISH TWINS from the policy's OWN stall states (DATA GENERATION ONLY; never an eval arm).

The 4dall policy parks the base, hovers the right hand 10-25 cm from the radio and freezes; with an oracle target it still
stops 14-20 cm short (10-01 oracle 0/10). No training clip starts from such a state, so the finish motion is absent. This
wrapper manufactures it, honestly, from the harvested stall states (stall_harvest_4dall.sh -> /root/stall_states/*.npz):

  reset():  env.reset() (the harness builds the TRAIN layout the state was harvested on)
            for each perturbation spec in FF_PERTURBS (';'-separated "dlat,ddepth,yaw_deg,dy,dz,hyaw_deg,code"):
              og.sim.load_state(stall npz) -> hold -> settle
              radio teleport by (dlat, ddepth, yaw) in the hand->radio frame, settle 30, reject if it moved/tipped (ODART rule)
              optional hand perturbation (dy lateral, dz up, hyaw wrist yaw) via the arm servo (NOT recorded)
              grasp target = canonical d20 rail grasp applied at the radio's rest pose
              RECORDED scripted finish (factory v13 phases, 10-DOF DLS servo = right arm 7 + trunk joints 1-3 via the TCAL probe, native
              Jacobian row 42 / blocks 0,3; FF_NO_TRUNK=1 for the arm-only variant):
                BASE_STAGE (BCAL-calibrated planar drive to 20 cm back on the corridor, only when the hand is beyond arm reach) ->
                RETREAT (if the hand is beside the body) -> ORIENT (attitude at the landing point) -> STAGE (10 cm out on the
                corridor) -> APPROACH -> PUSH 1.5 cm along the tines -> CLOSE with streak-gated verified weld / native AG ->
                LIFT FF_LIFT m holding the attitude -> hold
              honesty: radio displaced <= 2 cm / rotated <= 15 deg before contact, <= 10 cm overall (honest_strict), as the factory
              clip -> FF_OUT/rac_9<id:03d><code:02d>_200.npz in the converter's LIVE RaC format (+ FF_META_OUT/<tag>_meta.json)
            end the episode (step() returns truncated=True at once).
Env: FF_STATE (stall npz), FF_ID (train instance id), FF_PERTURBS, FF_OUT (/root/factory_obs_finish), FF_META_OUT
     (/root/factory_clips_finish), FF_CANON (/root/canonical_grasp_d20.json), FF_LIFT 0.10, FF_SETTLE 20, FF_MAX_GAP 1.20 (skip spin-outs), FF_ARM_REACH 0.40 (base drive-up beyond).
Prints per clip: FINISH_RESULT code=.. ok=.. strict=.. phase=.. n_obs=.. and FINISH_DONE n_ok=..
"""
import json
import os
import time

import numpy as np

from behavior2026_eval.affordance_map_fullres import AffordanceMapFullRes
from behavior2026_eval.freeze_harvest import proprio_of, _np, Q_ARM_L, Q_ARM_R, Q_TRUNK

STATE = os.environ.get("FF_STATE", "")
FID = int(os.environ.get("FF_ID", "0"))
PERTURBS = [s for s in os.environ.get("FF_PERTURBS", "0,0,0,0,0,0,00").split(";") if s.strip()]
OUT = os.environ.get("FF_OUT", "/root/factory_obs_finish")
META_OUT = os.environ.get("FF_META_OUT", "/root/factory_clips_finish")
CANON = os.environ.get("FF_CANON", "/root/canonical_grasp_d20.json")
LIFT = float(os.environ.get("FF_LIFT", 0.10))
SETTLE = int(os.environ.get("FF_SETTLE", 20))
MAX_GAP = float(os.environ.get("FF_MAX_GAP", 1.20))     # beyond this the stall is a spin-out, not a hover: skip
ARM_REACH = float(os.environ.get("FF_ARM_REACH", 0.40))  # hand->grasp gap the 7-DOF arm servo covers; farther -> base drive-up first
RETREAT_D = float(os.environ.get("FF_RETREAT", 0.30))    # landing point back along the (lifted) corridor where the attitude is set
ORIENT_BUDGET = int(os.environ.get("FF_ORIENT_BUDGET", 300))
NO_TRUNK = os.environ.get("FF_NO_TRUNK", "0") == "1"       # 1 = arm-only servo (the 10-01 smokes); default = factory 10-DOF
A_TORSO, A_ARM_L, A_GRIP_L, A_ARM_R, A_GRIP_R = slice(3, 7), slice(7, 14), 14, slice(15, 22), 22   # 23-d action
J_ROW, J_PBLK, J_ABLK = 42, 0, 3   # RCAL3 result on every factory demo (freeze_continue.py)
OPEN_AIR = ("RETREAT", "POSTURE", "ORIENT", "STAGE", "APPROACH")
POSTURE_FILE = os.environ.get("FF_POSTURE", "/root/canonical_pregrasp.json")   # canonical pre-grasp arm posture (canon_posture.py)


class FinishFactoryWrapper(AffordanceMapFullRes):
    def __init__(self, env):
        super().__init__(env)
        self._n_reset = 0
        self._done = False
        self._radio = None

    # ---- sim helpers ---------------------------------------------------------------------------
    def _q61(self):
        obs = self.env.get_obs()
        obs = obs[0] if isinstance(obs, tuple) else obs
        p = proprio_of(self._robot.name, obs)
        if p is None:
            for k, v in obs.items():
                if isinstance(v, dict) and "proprio" in v:
                    p = np.asarray(v["proprio"], np.float64).reshape(-1)
                    break
        return p

    def _hold(self, grip_r=1.0):
        q = self._q61()
        h = np.zeros(23, np.float32)
        h[A_TORSO] = q[Q_TRUNK]
        h[6] = 0.0
        h[A_ARM_L] = q[Q_ARM_L]
        h[A_ARM_R] = q[Q_ARM_R]
        h[A_GRIP_L] = 1.0
        h[A_GRIP_R] = grip_r
        return h

    def _radio_obj(self):
        import omnigibson as og
        if self._radio is None:
            self._radio = [o for o in og.sim.scenes[0].objects if "radio" in o.name.lower()][0]
        return self._radio

    def _poseR(self):
        from scipy.spatial.transform import Rotation as R
        p, q = self._robot.eef_links["right"].get_position_orientation()
        return _np(p).astype(np.float64).copy(), R.from_quat(_np(q))

    def _radio_pose(self):
        from scipy.spatial.transform import Rotation as R
        p, q = self._radio_obj().get_position_orientation()
        return _np(p).astype(np.float64).copy(), R.from_quat(_np(q))

    def _links(self):
        return [_np(l.get_position_orientation()[0]) for l in self._robot.finger_links["right"]]

    def _contact(self):
        try:
            cs, _ = self._robot._find_gripper_contacts(arm="right")
            return any(self._radio_obj().name in c for c in cs)
        except Exception:  # noqa: BLE001
            return False

    def _inhand(self):
        try:
            return self._robot._calculate_in_hand_object(arm="right") is not None
        except Exception:  # noqa: BLE001
            return False

    def _saturated(self, margin=0.05):
        """Right-arm joints within `margin` rad of a limit: [(idx, 'lo'|'hi')]."""
        try:
            rob = self._robot; idx = np.asarray(_np(rob.arm_control_idx["right"]), int)
            lo = _np(rob.joint_lower_limits)[idx]; hi = _np(rob.joint_upper_limits)[idx]; q = self._q61()[Q_ARM_R]
            return [(int(i), "lo") for i in range(7) if q[i] - lo[i] < margin] + [(int(i), "hi") for i in range(7) if hi[i] - q[i] < margin]
        except Exception as e:  # noqa: BLE001
            return f"n/a {e!r}"

    def _posture(self, hold, q_target, tag="POSTURE", steps=60, tol=0.03):
        """RECORDED joint-space move of the right arm (and trunk) to q_target = (arm7, trunk4): linear interpolation in joint
        space, honesty-ticked (open-air phase). The hand sweeps through free space at the retreat landing point."""
        q0 = self._q61(); qa0 = q0[Q_ARM_R].copy(); qt0 = q0[Q_TRUNK].copy()
        qa1 = np.asarray(q_target[0], float); qt1 = np.asarray(q_target[1], float) if q_target[1] is not None else qt0
        for k in range(1, steps + 1):
            a = k / steps
            cmd = hold.copy(); cmd[A_ARM_R] = (1 - a) * qa0 + a * qa1; cmd[A_TORSO] = (1 - a) * qt0 + a * qt1; cmd[6] = 0.0
            self._step_cmd(cmd, settle=8)
            hold = cmd
            if not self._honesty_tick(tag, k):
                return False, hold
        err = float(np.abs(self._q61()[Q_ARM_R] - qa1).max())
        print(f"{tag} done arm_err={err:.3f} sat={self._saturated()}", flush=True)
        return err < tol, hold

    def _tcal(self, hold):
        """Factory TCAL: probe trunk action channels 3..5 (+0.05 rad) -> right-eef position/rotation columns per rad. NOT recorded."""
        Tp, Ta = [], []
        cap = self._cap; self._cap = False
        for jch in range(3, 6):
            q = self._q61(); e0, R0 = self._poseR()
            pr = hold.copy(); pr[A_ARM_R] = q[Q_ARM_R]; pr[A_TORSO] = q[Q_TRUNK]; pr[jch] += 0.05; pr[6] = 0.0
            self._step_cmd(pr, settle=SETTLE)
            e1, R1 = self._poseR()
            Tp.append((e1 - e0) / 0.05); Ta.append((R1 * R0.inv()).as_rotvec() / 0.05)
            un = pr.copy(); un[jch] -= 0.05
            self._step_cmd(un, settle=SETTLE)
        self._cap = cap
        self._Jtp, self._Jta = np.stack(Tp, 1), np.stack(Ta, 1)
        print(f"TCAL trunk pos-col norms {[round(float(np.linalg.norm(c)), 3) for c in Tp]}", flush=True)

    def _native_ag(self):
        return self._robot._ag_obj_constraint_params.get("right") is not None

    # ---- recording (factory capture(): obs_t -> action_t, one frame per commanded step) -------------
    def _get_obs(self):
        obs = self.env.get_obs()
        obs = obs[0] if isinstance(obs, tuple) else obs
        o = {"pro": None, "zed_rgb": None, "zed_dep": None, "l_rgb": None, "l_dep": None, "r_rgb": None, "r_dep": None}

        def walk(node, path=""):
            if isinstance(node, dict):
                for k, v in node.items():
                    kl = str(k).lower(); pl = path + "/" + kl
                    if "proprio" in kl:
                        o["pro"] = _np(v).reshape(-1)
                    elif isinstance(v, dict):
                        walk(v, path + "/" + str(k))
                    else:
                        c = ("zed" if "zed" in pl else "l" if "left" in pl else "r" if "right" in pl else None)
                        if c:
                            if kl.endswith("rgb") or kl == "rgb":
                                o[f"{c}_rgb"] = _np(v)
                            elif "depth" in kl:
                                o[f"{c}_dep"] = _np(v)
        walk(obs)
        return o

    def _capture(self, cmd):
        if not self._cap:
            return
        o = self._get_obs()
        if o["pro"] is None or o["zed_rgb"] is None:
            return
        REC = self._rec
        REC["p"].append(o["pro"].astype(np.float32)); REC["act"].append(np.asarray(cmd, np.float32).copy())
        REC["hz"].append(o["zed_rgb"][..., :3].astype(np.uint8))
        REC["hd"].append(np.zeros((2, 2), np.float16) if o["zed_dep"] is None else o["zed_dep"].astype(np.float16))
        for c, kz, kd in (("l", "lz", "ld"), ("r", "rz", "rd")):
            REC[kz].append(np.zeros((2, 2, 3), np.uint8) if o[f"{c}_rgb"] is None else o[f"{c}_rgb"][..., :3].astype(np.uint8))
            REC[kd].append(np.zeros((2, 2), np.float16) if o[f"{c}_dep"] is None else o[f"{c}_dep"].astype(np.float16))
        rp_, rq_ = self._radio_obj().get_position_orientation(); bp_, bq_ = self._robot.get_position_orientation()
        REC["rp"].append(np.concatenate([_np(rp_), _np(rq_)]).astype(np.float32))
        REC["bp"].append(np.concatenate([_np(bp_), _np(bq_)]).astype(np.float32))

    def _step_cmd(self, cmd, settle=None, tol=0.01):
        self._capture(cmd)
        settle = SETTLE if settle is None else settle
        out = None
        for _ in range(settle):
            out = self.env.step(np.asarray(cmd, np.float32), n_render_iterations=1)
            self._n_env += 1
            q = self._q61()
            if (np.abs(q[Q_ARM_R] - cmd[A_ARM_R]).max() < tol and np.abs(q[Q_ARM_L] - cmd[A_ARM_L]).max() < tol
                    and np.abs(q[Q_TRUNK] - cmd[A_TORSO]).max() < tol):
                break
        return out

    def _honesty_tick(self, tag, it):
        if tag not in OPEN_AIR + ("PUSH", "ALIGN", "BASE") or getattr(self, "_hon", None) is None:
            return True
        H = self._hon
        H["n"] += 1
        p, Rr = self._radio_pose()
        d = float(np.linalg.norm(p - H["radio_rest"]))
        H["max_disp"] = max(H["max_disp"], d)
        if H["first_contact"] is None and self._contact():
            H["first_contact"] = f"{tag}:{it}"
        if tag in OPEN_AIR:
            H["pre_disp"] = max(H["pre_disp"], d)
            H["pre_rot"] = max(H["pre_rot"], float((Rr * H["R_rest"].inv()).magnitude() * 180 / np.pi))
            if H["pre_disp"] > 0.04 or H["pre_rot"] > 15.0:
                print(f"{tag} RADIO_TOUCHED it={it} disp={H['pre_disp']:.3f} rot={H['pre_rot']:.1f} -- stopping", flush=True)
                return False
        return True

    def _servo(self, target_p, target_R, hold, budget, tag, stride=0.006, w_orn=1.0, done=0.02, orn_done=0.15, grip=None):
        """7-DOF DLS servo of the right eef (freeze_continue.py's calibrated servo) with the factory's honesty tick and
        'reached' = position AND attitude. Returns (ok, best_d, oerr); self._last_pos_ok set."""
        rob = self._robot
        arm_idx = np.asarray(_np(rob.arm_control_idx["right"]), int)
        best_d = 9.9; self._last_pos_ok = False; oerr = 9.9
        for it in range(budget):
            pE, RE = self._poseR()
            d = float(np.linalg.norm(target_p - pE)); best_d = min(best_d, d)
            oerr = float(np.linalg.norm((target_R * RE.inv()).as_rotvec()))
            if d < done:
                self._last_pos_ok = True
                if oerr < orn_done:
                    print(f"{tag} reached it={it} d={d:.4f} orn_err={oerr:.3f}", flush=True)
                    return True, best_d, oerr
            if tag == "LIFT" and not self._native_ag():
                print(f"{tag} AG_LOST it={it}", flush=True)
                return False, best_d, oerr
            Jf = _np(rob.get_jacobian())
            Jp = Jf[J_ROW, J_PBLK:J_PBLK + 3, :][:, arm_idx]
            Ja = Jf[J_ROW, J_ABLK:J_ABLK + 3, :][:, arm_idx]
            trunk = getattr(self, "_Jtp", None) is not None and not NO_TRUNK
            if trunk:   # factory J11: arm columns + TCAL-probed trunk columns (joints 1-3) -> 10-DOF reach
                Jp = np.concatenate([Jp, self._Jtp], axis=1); Ja = np.concatenate([Ja, self._Jta], axis=1)
            v = (target_p - pE) / (d + 1e-9) * min(stride, d)
            w = np.clip((target_R * RE.inv()).as_rotvec(), -0.05, 0.05) * w_orn
            Jst = np.concatenate([Jp, np.sqrt(w_orn) * Ja], axis=0)
            rhs = np.concatenate([v, np.sqrt(w_orn) * w])
            dq = Jst.T @ np.linalg.solve(Jst @ Jst.T + 0.01 * np.eye(6), rhs)
            q_ = self._q61()
            cmd = hold.copy()
            cmd[A_ARM_R] = q_[Q_ARM_R] + np.clip(dq[:7], -0.08, 0.08)
            if trunk:
                cmd[3:6] = q_[Q_TRUNK][:3] + np.clip(dq[7:10], -0.03, 0.03)
            cmd[6] = 0.0
            if grip is not None:
                cmd[A_GRIP_R] = grip
            self._step_cmd(cmd)
            hold = cmd
            if not self._honesty_tick(tag, it):
                return False, best_d, oerr
            if it % 20 == 0:
                print(f"{tag} it={it} d={d:.4f} orn_err={oerr:.3f} ag={self._native_ag()} sat={self._saturated()}", flush=True)
        pE, RE = self._poseR()
        oerr = float(np.linalg.norm((target_R * RE.inv()).as_rotvec()))
        print(f"{tag} budget exhausted best={best_d:.4f} orn_err={oerr:.3f}", flush=True)
        return False, best_d, oerr

    # ---- base drive (factory v11 phase A0: BCAL probe -> closed-loop planar drive; recorded when capture is on) ------
    def _base_xy_yaw(self):
        from scipy.spatial.transform import Rotation as R
        pB, qB = self._robot.get_position_orientation()
        pB = _np(pB).astype(np.float64).copy(); yaw = float(R.from_quat(_np(qB)).as_euler("xyz")[2])
        return pB, yaw

    def _step_base(self, cmd):
        self._capture(cmd)
        self.env.step(np.asarray(cmd, np.float32), n_render_iterations=1); self._n_env += 1

    def _bcal(self, hold):
        """Probe the two planar base command channels (8 steps each at 0.15) -> base-frame m/step per unit command; drive back."""
        Kc = []
        for ch in (0, 1):
            p0, y0 = self._base_xy_yaw(); pr = hold.copy(); pr[ch] = 0.15
            for _ in range(8):
                self.env.step(pr, n_render_iterations=1); self._n_env += 1
            p1, _ = self._base_xy_yaw(); dw = (p1[:2] - p0[:2]) / 8.0 / 0.15
            c, s = np.cos(y0), np.sin(y0)
            Kc.append(np.array([c * dw[0] + s * dw[1], -s * dw[0] + c * dw[1]]))
            for _ in range(8):
                self.env.step(hold, n_render_iterations=1); self._n_env += 1
        Kmat = np.stack(Kc, 1)
        try:
            self._bcal_k = np.linalg.inv(Kmat) * np.linalg.norm(Kmat, ord=2)
        except np.linalg.LinAlgError:
            self._bcal_k = np.eye(2)
        print(f"BCAL ch0 {np.round(Kc[0], 4).tolist()} ch1 {np.round(Kc[1], 4).tolist()}", flush=True)
        return Kc

    def _drive_base(self, goal_xy, hold, tag, v_max=0.30, tol=0.03, max_steps=600, watch=True):
        blocked = 0; dist = 9.9
        for it in range(max_steps):
            pB, yaw = self._base_xy_yaw()
            err = goal_xy - pB[:2]; dist = float(np.linalg.norm(err))
            if dist < tol:
                print(f"{tag} reached it={it} dist={dist:.3f}", flush=True)
                return True, dist, it
            v_w = err / dist * min(1.0, dist / 0.15)
            c, s = np.cos(yaw), np.sin(yaw)
            v_b = np.array([c * v_w[0] + s * v_w[1], -s * v_w[0] + c * v_w[1]])
            cmd = hold.copy(); cmd[0:2] = np.clip(self._bcal_k @ v_b * v_max, -0.30, 0.30); cmd[2] = 0.0
            self._step_base(cmd)
            if watch and getattr(self, "_hon", None) is not None:
                H = self._hon; d = float(np.linalg.norm(self._radio_pose()[0] - H["radio_rest"]))
                H["pre_disp"] = max(H["pre_disp"], d); H["max_disp"] = max(H["max_disp"], d)
                if d > 0.012:
                    print(f"{tag} RADIO_TOUCHED it={it} disp={d:.3f} dist={dist:.3f} -- stopping the drive", flush=True)
                    return False, dist, it
            moved = float(np.linalg.norm(self._base_xy_yaw()[0][:2] - pB[:2]))
            blocked = blocked + 1 if moved < 2e-4 else 0
            if blocked >= 40:
                print(f"{tag} BLOCKED it={it} dist={dist:.3f}", flush=True)
                return False, dist, it
            if it % 30 == 0:
                print(f"{tag} it={it} dist={dist:.3f} moved/step={moved:.4f}", flush=True)
        print(f"{tag} budget exhausted dist={dist:.3f}", flush=True)
        return False, dist, max_steps

    def _hand_drive(self, target_hand_p, hold, tag, max_steps=600, tol=0.03):
        """Drive the base so the (held) right hand lands at target_hand_p (planar)."""
        pB, _ = self._base_xy_yaw(); goal = pB[:2] + (np.asarray(target_hand_p)[:2] - self._poseR()[0][:2])
        return self._drive_base(goal, hold, tag, tol=tol, max_steps=max_steps)

    # ---- one clip ------------------------------------------------------------------------------------
    def _one_clip(self, spec):
        import omnigibson as og
        import torch as th
        from scipy.spatial.transform import Rotation as R
        dlat, ddep, oyaw, hdy, hdz, hyaw = [float(v) for v in spec.split(",")[:6]]
        code = spec.split(",")[6].strip() if len(spec.split(",")) > 6 else "00"
        tag = f"tr{FID:03d}_{code}"
        t0 = time.time()
        self._rec = {k: [] for k in ("p", "act", "hz", "hd", "lz", "ld", "rz", "rd", "rp", "bp")}
        self._cap = False; self._n_env = 0; self._Jtp = None; self._Jta = None
        meta = dict(tag=tag, state=STATE, train_id=FID, code=code, perturb=dict(dlat=dlat, ddepth=ddep, yaw=oyaw, hdy=hdy, hdz=hdz, hyaw=hyaw),
                    ok=False, honest=False, honest_strict=False, phase="init", kind="finish_v1_stall")
        z = np.load(STATE)
        og.sim.load_state(th.as_tensor(np.asarray(z["state"], np.float32)), serialized=True)
        hold = self._hold()
        self._step_cmd(hold, settle=10)
        for _ in range(20):
            self.env.step(hold, n_render_iterations=1)
        try:
            self._robot._refresh_rigid_contact_view()
        except Exception:  # noqa: BLE001
            pass
        pE0, RE0 = self._poseR(); pRad0, RRad0 = self._radio_pose()
        canon = json.load(open(CANON))
        rel_p = np.array(canon["rel_p"]); rel_R = R.from_quat(canon["rel_R_quat"]); tine_axis = np.array(canon["tine_axis_radio"])
        # corridor from the CURRENT hand toward the grasp pose (lifted to >= 35 deg elevation, factory v13d rule)
        tgt0 = pRad0 + RRad0.apply(rel_p)
        corr = pE0 - tgt0; cn = float(np.linalg.norm(corr))
        if cn < 0.05:
            corr = np.array([0.0, 0.0, 1.0]); cn = 1.0
        sdir = corr / cn
        el = np.degrees(np.arcsin(np.clip(sdir[2], -1, 1)))
        if el < 35.0:
            xy = sdir[:2] / (np.linalg.norm(sdir[:2]) + 1e-9)
            sdir = np.array([xy[0] * np.cos(np.radians(35.0)), xy[1] * np.cos(np.radians(35.0)), np.sin(np.radians(35.0))])
        lat = np.array([-sdir[1], sdir[0], 0.0]); lat /= (np.linalg.norm(lat) + 1e-9)
        dep = np.array([sdir[0], sdir[1], 0.0]); dep /= (np.linalg.norm(dep) + 1e-9)
        meta.update(gap_stall=round(float(np.linalg.norm(tgt0 - pE0)), 4), base_to_radio=round(float(np.linalg.norm((pRad0 - _np(self._robot.get_position_orientation()[0]))[:2])), 3))
        # ---- radio perturbation (ODART rule: teleport, settle 30, reject if it moved > 2 cm or tipped > 5 deg) ----
        if abs(dlat) > 1e-6 or abs(ddep) > 1e-6 or abs(oyaw) > 1e-6:
            pn = pRad0 + dlat * lat + ddep * dep
            Rn = R.from_euler("z", oyaw, degrees=True) * RRad0
            self._radio_obj().set_position_orientation(th.as_tensor(pn, dtype=th.float32), th.as_tensor(Rn.as_quat(), dtype=th.float32))
            for _ in range(30):
                self.env.step(hold, n_render_iterations=1)
            ps, Rs = self._radio_pose()
            sd = float(np.linalg.norm(ps - pn)); sr = float((Rs * Rn.inv()).magnitude() * 180 / np.pi)
            print(f"OBJPERTURB {tag} dlat={dlat:+.3f} ddep={ddep:+.3f} yaw={oyaw:+.1f} settled_d={sd:.3f} rot={sr:.1f}", flush=True)
            if sd > 0.02 or sr > 5.0:
                meta["phase"] = "SKIP_OBJPERTURB_UNSETTLED"; return self._skip_meta(meta)
        # ---- hand perturbation (NOT recorded, not honesty-tracked) ----
        if abs(hdy) > 1e-6 or abs(hdz) > 1e-6 or abs(hyaw) > 1e-6:
            gp = pE0 + hdy * lat + np.array([0.0, 0.0, hdz]); gR = R.from_euler("z", hyaw, degrees=True) * RE0
            okp, dp, oe = self._servo(gp, gR, hold, 120, "HANDPERTURB", stride=0.006, w_orn=1.0, done=0.012, orn_done=0.08)
            hold = self._hold()
            print(f"HANDPERTURB {tag} dy={hdy:+.3f} dz={hdz:+.3f} yaw={hyaw:+.1f} ok={okp} d={dp:.3f}", flush=True)
            if float(np.linalg.norm(self._radio_pose()[0] - (pRad0 if abs(dlat) + abs(ddep) + abs(oyaw) < 1e-6 else ps))) > 0.02:
                meta["phase"] = "SKIP_HANDPERTURB_TOUCHED"; return self._skip_meta(meta)
        pRest, RRest = self._radio_pose()
        tgt_p = pRest + RRest.apply(rel_p); tgt_R = RRest * rel_R
        pE, RE = self._poseR()
        gap0 = float(np.linalg.norm(tgt_p - pE))
        meta.update(gap0=round(gap0, 4), radio_rest=np.round(pRest, 4).tolist())
        print(f"PRE {tag} hand->grasp {gap0:.3f} m, corridor {np.round(sdir, 3).tolist()}", flush=True)
        try:   # diagnostic: what the robot body is touching at the stall (a base parked against the table makes any base drive dishonest)
            bodies = set(); rp = str(self._robot.prim_path)
            for ln in list(self._robot.links.values())[:40]:
                if not hasattr(ln, "contact_list"):
                    continue
                for c in (ln.contact_list() or []):
                    for b in (getattr(c, "body0", ""), getattr(c, "body1", "")):
                        b = str(b)
                        if b and rp not in b:
                            bodies.add(b.split("/")[-2] if b.count("/") >= 2 else b)
            bodies = sorted(bodies)
            print(f"CONTACTS {tag}: {bodies[:8]}", flush=True)
            meta["contacts"] = bodies[:8]
        except Exception as e:  # noqa: BLE001
            print(f"CONTACTS {tag}: n/a {e!r}", flush=True)
        if gap0 > MAX_GAP:
            meta["phase"] = "SKIP_GAP_TOO_LARGE"; return self._skip_meta(meta)
        self._hon = {"radio_rest": pRest.copy(), "R_rest": RRest, "max_disp": 0.0, "pre_disp": 0.0, "pre_rot": 0.0, "first_contact": None, "n": 0}
        # ---- phase A0: BASE DRIVE-UP when the hover is beyond arm reach (planar landing point 20 cm back along the corridor;
        #      BCAL probe jiggles are not recorded, the drive is) ----
        base = {"driven": False}
        land_p = tgt_p + 0.20 * sdir
        planar = float(np.linalg.norm((land_p - pE)[:2]))
        if gap0 > ARM_REACH or planar > 0.12:
            self._cap = False
            self._bcal(hold)
            p_start, _ = self._base_xy_yaw()
            self._drive_base(p_start[:2], hold, "BCAL_RESET", tol=0.01, max_steps=200, watch=False)
            self._cap = True
            okb, leftb, nb = self._hand_drive(land_p, hold, "BASE_STAGE", max_steps=600)
            hold = self._hold()
            pE, RE = self._poseR(); gap_b = float(np.linalg.norm(tgt_p - pE))
            base.update(driven=True, ok=bool(okb), left=round(leftb, 3), steps=nb, gap_after=round(gap_b, 4))
            print(f"BASE_STAGE ok={okb} left={leftb:.3f} steps={nb} hand->grasp {gap_b:.3f} m", flush=True)
            if self._hon["pre_disp"] > 0.012:
                meta.update(phase="RADIO_TOUCHED_BASE" if nb > 10 else "BASE_BLOCKED_TABLE", base=base); return self._finish_meta(meta)
            if gap_b > ARM_REACH + 0.10:
                meta.update(phase="BASE_STAGE_FAILED", base=base); return self._finish_meta(meta)
            gap0 = gap_b
        meta["base"] = base
        self._cap = True
        # ---- phase A: RETREAT to a clear landing point -> ORIENT there -> STAGE -> APPROACH ----
        # Policy hovers sit 10-25 cm from the radio with the wrist up to ~3 rad off the grasp attitude (smoke 10-01: orienting
        # 15 cm from the table stalled at 0.37 rad / drifted 12 cm); the 09-17 freeze-continue servo converged from 2.8 rad when
        # the hand was >= 0.5 m clear. So: land RETREAT_D back along the corridor (which is lifted >= 35 deg -> up and away).
        stage_p = tgt_p + 0.10 * sdir
        land = tgt_p + RETREAT_D * sdir
        if float(np.linalg.norm(land - pE)) > 0.03:
            ok_r, _, _ = self._servo(land, RE, hold, 120, "RETREAT", stride=0.012, w_orn=0.3, done=0.02, orn_done=9.9)
            hold = self._hold()
            if not ok_r and self._hon["pre_disp"] > 0.04:
                meta["phase"] = "RADIO_TOUCHED_RETREAT"; return self._finish_meta(meta)
        print(f"RETREAT_END sat={self._saturated()} orn_err={float(np.linalg.norm((tgt_R * self._poseR()[1].inv()).as_rotvec())):.3f}", flush=True)
        if os.path.exists(POSTURE_FILE):   # joint-space move to the canonical pre-grasp posture: clears wrist-limit lockups (smoke 10-01)
            cp = json.load(open(POSTURE_FILE))
            ok_p, hold = self._posture(hold, (cp["arm_right_qpos"], cp.get("trunk_qpos")))
            if self._hon["pre_disp"] > 0.04 or self._hon["pre_rot"] > 15.0:
                meta["phase"] = "RADIO_TOUCHED_POSTURE"; return self._finish_meta(meta)
            # the posture moved the hand: re-land at the retreat point before orienting
            land = tgt_p + RETREAT_D * sdir
            self._servo(land, self._poseR()[1], hold, 120, "RETREAT", stride=0.012, w_orn=0.3, done=0.02, orn_done=9.9)
            hold = self._hold()
            print(f"POSTURE_END orn_err={float(np.linalg.norm((tgt_R * self._poseR()[1].inv()).as_rotvec())):.3f} hand->grasp {float(np.linalg.norm(tgt_p - self._poseR()[0])):.3f}", flush=True)
        if not NO_TRUNK:
            self._tcal(hold); hold = self._hold()
        anchor = self._poseR()[0].copy()
        ok_o, _, oe_o = self._servo(anchor, tgt_R, hold, ORIENT_BUDGET, "ORIENT", stride=0.006, w_orn=1.0, done=0.03, orn_done=0.10)
        hold = self._hold()
        print(f"ORIENT ok={ok_o} orn_err={oe_o:.3f}", flush=True)
        ok_s, best_s, _ = self._servo(stage_p, tgt_R, hold, 260, "STAGE", stride=0.012, w_orn=0.6, done=0.015, orn_done=0.10)
        hold = self._hold()
        print(f"STAGE ok={ok_s} best={best_s:.3f}", flush=True)
        ok_a, best_a, _ = self._servo(tgt_p, tgt_R, hold, 160, "APPROACH", stride=0.008, w_orn=0.6, done=0.025, orn_done=0.10)
        hold = self._hold()
        print(f"APPROACH ok={ok_a} best={best_a:.3f} pos_ok={self._last_pos_ok}", flush=True)
        if self._hon["pre_disp"] > 0.04 or self._hon["pre_rot"] > 15.0:
            meta["phase"] = "RADIO_TOUCHED"; return self._finish_meta(meta)
        if not ok_a and not self._last_pos_ok:
            meta.update(phase="APPROACH_FAILED", best_a=round(best_a, 4)); return self._finish_meta(meta)
        # ---- PUSH 1.5 cm along the tines ----
        tine_w = RRest.apply(tine_axis)
        self._servo(tgt_p + 0.015 * tine_w, tgt_R, hold, 25, "PUSH", stride=0.006, w_orn=0.6, done=0.008, orn_done=9.9)
        hold = self._hold()
        # ---- phase B: CLOSE + streak-gated verified weld ----
        hold[A_GRIP_R] = -1.0
        green, gap, weld_k = 0, 0, None
        self._capture(hold)
        for k in range(800):
            self.env.step(hold, n_render_iterations=1); self._n_env += 1
            if k % 8 == 0:
                self._capture(hold)
            c_, i_ = self._contact(), self._inhand()
            if k < 40 and k % 10 == 0:
                fl = self._links(); print(f"CLOSE k={k} contact={c_} inhand={i_} finger_gap={np.linalg.norm(fl[0] - fl[1]):.3f}", flush=True)
            if c_ and i_:
                green += 1; gap = 0
            else:
                gap += 1
                if gap > 4:
                    green = 0
            if self._native_ag():
                weld_k = k; print(f"NATIVE_AG k={k}", flush=True); break
            if k >= 320 and green >= 296:
                cp = th.as_tensor(np.mean(self._links(), axis=0), dtype=th.float32)
                self._robot._establish_grasp(self._radio_obj(), self._radio_obj().root_link_name, "right", cp, "FixedJoint")
                weld_k = k; print(f"VERIFIED_WELD k={k} streak={green}", flush=True); break
        meta.update(weld_k=weld_k, streak=green)
        if not self._native_ag():
            meta["phase"] = "WELD_FAILED"; return self._finish_meta(meta)
        # ---- phase C: LIFT holding the attitude, then hold ----
        hold = self._hold(grip_r=-1.0)
        pE, RE = self._poseR()
        ok_l, best_l, _ = self._servo(pE + np.array([0.0, 0.0, LIFT]), RE, hold, 150, "LIFT", stride=0.010, w_orn=0.6, done=0.02, orn_done=9.9, grip=-1.0)
        hold = self._hold(grip_r=-1.0)
        for _ in range(30):
            self._step_cmd(hold, settle=1)
        z_end = float(self._radio_pose()[0][2])
        lifted = (z_end - pRest[2]) > 0.05
        meta.update(lift_ok=bool(ok_l), lifted=bool(lifted), radio_z_end=round(z_end, 4), ag_intact=bool(self._native_ag()), phase="DONE")
        return self._finish_meta(meta)

    def _skip_meta(self, meta):
        os.makedirs(META_OUT, exist_ok=True)
        meta.update(n_obs=0, n_env_steps=self._n_env)
        json.dump(meta, open(f"{META_OUT}/{meta['tag']}_meta.json", "w"), indent=1)
        print(f"FINISH_RESULT tag={meta['tag']} code={meta['code']} ok=False strict=False phase={meta['phase']} n_obs=0 gap0={meta.get('gap0', meta.get('gap_stall'))}", flush=True)
        return meta

    def _finish_meta(self, meta):
        H = self._hon
        strict = H["pre_disp"] <= 0.02 and H["pre_rot"] <= 15.0 and H["max_disp"] <= 0.10
        honest = H["pre_disp"] <= 0.04 and H["pre_rot"] <= 15.0 and H["max_disp"] <= 0.12
        ok = bool(meta.get("phase") == "DONE" and meta.get("ag_intact") and meta.get("lifted") and honest)
        meta.update(honest=bool(honest), honest_strict=bool(strict), ok=ok, radio_disp_precontact=round(H["pre_disp"], 4),
                    radio_rot_precontact=round(H["pre_rot"], 2), radio_disp_max=round(H["max_disp"], 4), first_contact=H["first_contact"],
                    n_obs=len(self._rec["p"]), n_env_steps=self._n_env)
        os.makedirs(OUT, exist_ok=True); os.makedirs(META_OUT, exist_ok=True)
        if ok and meta["n_obs"] > 10:
            REC = self._rec
            name = f"rac_9{FID:03d}{meta['code']}_200.npz"
            np.savez_compressed(f"{OUT}/{name}",
                                proprio=np.stack(REC["p"]), actions=np.stack(REC["act"]),
                                head_rgb=np.stack(REC["hz"]), head_depth=np.stack(REC["hd"]),
                                left_rgb=np.stack(REC["lz"]), left_depth=np.stack(REC["ld"]),
                                right_rgb=np.stack(REC["rz"]), right_depth=np.stack(REC["rd"]),
                                objpose_radio_89=np.stack(REC["rp"]), base_pose=np.stack(REC["bp"]),
                                radio_rest_z=np.float64(self._hon["radio_rest"][2]), success=np.bool_(True),
                                meta=json.dumps({**meta, "episode": "finish_v1_stall"}))
            with open(f"{OUT}/name_map.txt", "a") as f:
                f.write(f"{name} {meta['tag']} {STATE}\n")
            meta["clip"] = name
            print(f"OBS_SAVED {name} ({meta['n_obs']} steps)", flush=True)
        json.dump(meta, open(f"{META_OUT}/{meta['tag']}_meta.json", "w"), indent=1)
        print(f"FINISH_RESULT tag={meta['tag']} code={meta['code']} ok={ok} strict={strict} phase={meta['phase']} n_obs={meta['n_obs']} "
              f"pre_disp={H['pre_disp']:.3f} pre_rot={H['pre_rot']:.1f} max_disp={H['max_disp']:.3f}", flush=True)
        return meta

    # ---- wrapper protocol -------------------------------------------------------------------------------
    def reset(self):
        out = super().reset()
        info = out[1] if isinstance(out, tuple) and len(out) > 1 else {}
        self._n_reset += 1
        if not STATE:
            print("FINISH_NO_STATE (FF_STATE unset) -- plain reset", flush=True)
            return out
        if self._n_reset == 1:   # the evaluator's pre-instance reset (eval.py: reset -> load_task_instance -> reset per rollout)
            print("FINISH_SKIP_PREINSTANCE_RESET", flush=True)
            return out
        n_ok = 0
        for spec in PERTURBS:
            try:
                m = self._one_clip(spec)
                n_ok += int(bool(m.get("ok")))
            except Exception as e:  # noqa: BLE001
                import traceback; traceback.print_exc()
                print(f"FINISH_CLIP_ERROR spec={spec} {e!r}", flush=True)
        print(f"FINISH_DONE id={FID} n_specs={len(PERTURBS)} n_ok={n_ok}", flush=True)
        self._done = True
        obs = self.env.get_obs(); obs = obs[0] if isinstance(obs, tuple) else obs
        return (self._inject(obs), info)

    def step(self, action, n_render_iterations=1):
        out = super().step(action, n_render_iterations=n_render_iterations)
        if self._done and isinstance(out, tuple) and len(out) >= 5:
            out = (out[0], out[1], out[2], True, out[4])
        return out

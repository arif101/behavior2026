"""FreezeContinueWrapper (2026-09-17 freeze diagnostic, DIAGNOSTIC ONLY - never an eval arm).

reset(): env.reset() -> og.sim.load_state(<FREEZE_STATE>.npz) -> hold the loaded joint targets -> optional ORIENT
(FREEZE_ORIENT=1): a 7-DOF DLS servo (the v13d factory servo, right arm only, native Jacobian row 42 / blocks 0,3 as
calibrated on every factory demo) rotates the RIGHT hand to the canonical rail-grasp attitude at the CURRENT radio pose
and advances FREEZE_ADVANCE m along the hand->grasp line. This mirrors S1 run 7's own escape (wrist 90 -> 23 deg to the
radio, hand 25 cm closer) and then hands control to the policy, which runs to the end under the parity serving stack.
FREEZE_ORIENT=0 = control: restore the freeze state and hand over untouched.
Reads the radio pose and the robot Jacobian from the sim during reset only: a diagnostic of the policy's exit from the
freeze, not an eval. Prints FREEZE_CONTINUE_READY orient=<0|1> orn_err=<rad> adv=<m>.
Env: FREEZE_STATE (npz from freeze_harvest.py), FREEZE_ORIENT 0|1, FREEZE_ADVANCE 0.25, FREEZE_CANON
     (/root/canonical_grasp_d20.json), FREEZE_BUDGET 200.
"""
import json
import os

import numpy as np

from behavior2026_eval.affordance_map_fullres import AffordanceMapFullRes
from behavior2026_eval.freeze_harvest import proprio_of, _np, Q_ARM_L, Q_ARM_R, Q_TRUNK

STATE = os.environ.get("FREEZE_STATE", "")
ORIENT = os.environ.get("FREEZE_ORIENT", "1") == "1"
ADVANCE = float(os.environ.get("FREEZE_ADVANCE", 0.25))
CANON = os.environ.get("FREEZE_CANON", "/root/canonical_grasp_d20.json")
BUDGET = int(os.environ.get("FREEZE_BUDGET", 200))
A_TORSO, A_ARM_L, A_GRIP_L, A_ARM_R, A_GRIP_R = slice(3, 7), slice(7, 14), 14, slice(15, 22), 22   # 23-d action
J_ROW, J_PBLK, J_ABLK = 42, 0, 3   # RCAL3 result on every factory demo: sel=(42, 0), angular blk 3, sign +1


class FreezeContinueWrapper(AffordanceMapFullRes):
    def __init__(self, env):
        super().__init__(env)
        self.last_orient = None
        self._n_reset = 0

    # ---- sim helpers (reset-time only) --------------------------------------------------------
    def _q61(self):
        obs = self.env.get_obs()
        obs = obs[0] if isinstance(obs, tuple) else obs
        p = proprio_of(self._robot.name, obs)
        if p is None:
            for k, v in obs.items():   # nested robot node under another key
                if isinstance(v, dict) and "proprio" in v:
                    p = np.asarray(v["proprio"], np.float64).reshape(-1)
                    break
        return p

    def _hold(self):
        q = self._q61()
        h = np.zeros(23, np.float32)
        h[A_TORSO] = q[Q_TRUNK]
        h[6] = 0.0            # torso joint 4 locked (10-DOF demo convention)
        h[A_ARM_L] = q[Q_ARM_L]
        h[A_ARM_R] = q[Q_ARM_R]
        h[A_GRIP_L] = 1.0
        h[A_GRIP_R] = 1.0     # open
        return h

    def _step_cmd(self, cmd, settle=20, tol=0.01):
        out = None
        for _ in range(settle):
            out = self.env.step(np.asarray(cmd, np.float32), n_render_iterations=1)
            q = self._q61()
            if (np.abs(q[Q_ARM_R] - cmd[A_ARM_R]).max() < tol and np.abs(q[Q_ARM_L] - cmd[A_ARM_L]).max() < tol
                    and np.abs(q[Q_TRUNK] - cmd[A_TORSO]).max() < tol):
                break
        return out

    def _poseR(self):
        from scipy.spatial.transform import Rotation as R
        p, q = self._robot.eef_links["right"].get_position_orientation()
        return _np(p).astype(np.float64).copy(), R.from_quat(_np(q))

    def _radio(self):
        import omnigibson as og
        from scipy.spatial.transform import Rotation as R
        radio = [o for o in og.sim.scenes[0].objects if "radio" in o.name.lower()][0]
        p, q = radio.get_position_orientation()
        return _np(p).astype(np.float64).copy(), R.from_quat(_np(q))

    def _servo(self, target_p, target_R, hold, budget, stride=0.006, w_orn=1.0, done=0.02, orn_done=0.15):
        rob = self._robot
        arm_idx = np.asarray(_np(rob.arm_control_idx["right"]), int)
        best = (9.9, 9.9)
        for it in range(budget):
            pE, RE = self._poseR()
            d = float(np.linalg.norm(target_p - pE))
            oerr = float(np.linalg.norm((target_R * RE.inv()).as_rotvec()))
            best = (min(best[0], d), oerr)
            if d < done and oerr < orn_done:
                print(f"ORIENT reached it={it} d={d:.4f} orn_err={oerr:.3f}", flush=True)
                return True, d, oerr
            Jf = _np(rob.get_jacobian())
            Jp = Jf[J_ROW, J_PBLK:J_PBLK + 3, :][:, arm_idx]
            Ja = Jf[J_ROW, J_ABLK:J_ABLK + 3, :][:, arm_idx]
            v = (target_p - pE) / (d + 1e-9) * min(stride, d)
            w = np.clip((target_R * RE.inv()).as_rotvec(), -0.05, 0.05) * w_orn
            Jst = np.concatenate([Jp, np.sqrt(w_orn) * Ja], axis=0)
            rhs = np.concatenate([v, np.sqrt(w_orn) * w])
            dq = Jst.T @ np.linalg.solve(Jst @ Jst.T + 0.01 * np.eye(6), rhs)
            dq = np.clip(dq, -0.08, 0.08)
            cmd = hold.copy()
            cmd[A_ARM_R] = self._q61()[Q_ARM_R] + dq
            self._step_cmd(cmd)
            hold = cmd
            if it % 20 == 0:
                print(f"ORIENT it={it} d={d:.4f} orn_err={oerr:.3f}", flush=True)
        pE, RE = self._poseR()
        oerr = float(np.linalg.norm((target_R * RE.inv()).as_rotvec()))
        print(f"ORIENT budget exhausted d={np.linalg.norm(target_p - pE):.4f} orn_err={oerr:.3f}", flush=True)
        return False, float(np.linalg.norm(target_p - pE)), oerr

    # ---- wrapper protocol ---------------------------------------------------------------------
    def reset(self):
        import omnigibson as og
        import torch as th
        out = super().reset()
        info = out[1] if isinstance(out, tuple) and len(out) > 1 else {}
        self._n_reset += 1
        if not STATE:
            print("FREEZE_CONTINUE_NO_STATE (FREEZE_STATE unset) -- plain reset", flush=True)
            return out
        if self._n_reset == 1:   # the evaluator's pre-instance reset (eval.py: reset -> load_task_instance -> reset per rollout)
            print("FREEZE_CONTINUE_SKIP_PREINSTANCE_RESET", flush=True)
            return out
        z = np.load(STATE)
        og.sim.load_state(th.as_tensor(np.asarray(z["state"], np.float32)), serialized=True)
        hold = self._hold()
        last = self._step_cmd(hold, settle=10)      # first physics steps carry the loaded joint targets
        for _ in range(20):
            last = self.env.step(hold, n_render_iterations=1)
        pE0, RE0 = self._poseR()
        pRad, RRad = self._radio()
        from scipy.spatial.transform import Rotation as R
        canon = json.load(open(CANON))
        tgt_p = pRad + RRad.apply(np.array(canon["rel_p"]))
        tgt_R = RRad * R.from_quat(canon["rel_R_quat"])
        oerr0 = float(np.linalg.norm((tgt_R * RE0.inv()).as_rotvec()))
        gap0 = float(np.linalg.norm(tgt_p - pE0))
        rec = {"state": STATE, "orient": int(ORIENT), "gap0": gap0, "orn_err0": oerr0, "advance": ADVANCE if ORIENT else 0.0}
        if ORIENT:
            u = (tgt_p - pE0) / (gap0 + 1e-9)
            goal_p = pE0 + u * min(ADVANCE, max(gap0 - 0.15, 0.0))
            ok, d, oerr = self._servo(goal_p, tgt_R, hold, BUDGET)
            rec.update({"ok": bool(ok), "d_end": d, "orn_err_end": oerr, "gap_end": float(np.linalg.norm(tgt_p - self._poseR()[0]))})
            hold = self._hold()
            for _ in range(10):
                last = self.env.step(hold, n_render_iterations=1)
        self.last_orient = rec
        print(f"FREEZE_CONTINUE_READY orient={rec['orient']} gap0={gap0:.3f} orn_err0={oerr0:.3f} "
              f"gap_end={rec.get('gap_end', gap0):.3f} orn_err_end={rec.get('orn_err_end', oerr0):.3f} adv={rec['advance']}", flush=True)
        try:
            json.dump(rec, open("/root/freeze_continue_last.json", "w"), indent=1)
        except Exception:  # noqa: BLE001
            pass
        obs = last[0] if isinstance(last, tuple) else last
        return (self._inject(obs), info)

"""JacobianProbeWrapper (2026-09-21 diagnostic; sim-state reads and radio moves; never an eval arm).

Measures the policy's LOCAL CORRECTIVE FIELD at the pre-grasp. Each rollout: restore a harvested near-grasp state
(JAC_STATE), servo the right hand (7-DOF DLS, ground-truth radio pose) to the canonical pre-grasp pose P0 = grasp target +
0.10 m up the approach corridor at the canonical attitude, then apply ONE condition:
  base           nothing
  y+3 y-3 y+6 y-6   hand offset laterally (perpendicular to the corridor, horizontal) by cm
  z+3 z-3 z+6 z-6   hand offset vertically by cm
  yaw+10 yaw-10     hand attitude rotated about world z by deg
  vis_y+3 vis_y-3 vis_x+3 vis_x-3   hand stays at P0, the RADIO is moved by the negated offset (image changes, proprio does not)
then hands over to the policy for JAC_STEPS (16 = one executed chunk) and records the EE motion. Conditions cycle over
rollouts (JAC_SAMPLES rollouts per condition). Record per rollout -> JAC_OUT/<tag>.jsonl:
  progress  = EE displacement projected on (target - P0) direction [m]      (does it still approach?)
  restore   = -(EE displacement - base displacement) . offset_dir [m]      (does it push back against the offset? >0 = corrects)
  yaw_restore = change of attitude error toward tgt_R [rad] (for yaw conditions)
"""
import json, os
import numpy as np
from behavior2026_eval.stage_v2_wrapper import StageV2AffordanceWrapper
from behavior2026_eval.freeze_continue import FreezeContinueWrapper, A_ARM_R, A_TORSO, A_ARM_L, A_GRIP_L, A_GRIP_R
from behavior2026_eval.freeze_harvest import _np, Q_ARM_L, Q_ARM_R, Q_TRUNK

STATE = os.environ.get("JAC_STATE", ""); OUT = os.environ.get("JAC_OUT", "/root/jacobian_probe"); TAG = os.environ.get("JAC_TAG", "untagged")
STEPS = int(os.environ.get("JAC_STEPS", 16)); SAMPLES = int(os.environ.get("JAC_SAMPLES", 3)); CANON = os.environ.get("FREEZE_CANON", "/root/canonical_grasp_d20.json")
CONDS = [c for c in os.environ.get("JAC_CONDS", "base,y+3,y-3,y+6,y-6,z+3,z-3,z+6,z-6,yaw+10,yaw-10,vis_y+3,vis_y-3,vis_x+3,vis_x-3").split(",") if c]


class JacobianProbeWrapper(FreezeContinueWrapper, StageV2AffordanceWrapper):
    """MRO: FreezeContinue (sim helpers _q61/_hold/_step_cmd/_poseR/_radio/_servo) -> StageV2Affordance (v2 passthrough the
    full checkpoint is served with) -> AffordanceMapFullRes."""
    def __init__(self, env):
        super().__init__(env); self._k = 0; self._rec = None; self._trace = []; self._done = False
        os.makedirs(OUT, exist_ok=True)

    def _cond(self):
        i = self._k // SAMPLES
        return CONDS[i % len(CONDS)], self._k % SAMPLES

    def reset(self):
        import omnigibson as og, torch as th
        from scipy.spatial.transform import Rotation as R
        if self._rec is not None and self._trace:
            self._finish()
        out = StageV2AffordanceWrapper.reset(self)
        info = out[1] if isinstance(out, tuple) and len(out) > 1 else {}
        self._n_reset += 1
        if self._n_reset == 1 or not STATE:
            print("JAC_SKIP_PREINSTANCE_RESET", flush=True); return out
        cond, sample = self._cond(); self._k += 1
        z = np.load(STATE); og.sim.load_state(th.as_tensor(np.asarray(z["state"], np.float32)), serialized=True)
        hold = self._hold(); self._step_cmd(hold, settle=10)
        for _ in range(10): last = self.env.step(hold, n_render_iterations=1)
        pRad, RRad = self._radio(); canon = json.load(open(CANON))
        tgt_p = pRad + RRad.apply(np.array(canon["rel_p"])); tgt_R = RRad * R.from_quat(canon["rel_R_quat"])
        pE0, _ = self._poseR()
        d = pE0 - tgt_p; dxy = d[:2] / (np.linalg.norm(d[:2]) + 1e-9)
        corr = np.array([dxy[0] * np.cos(np.radians(40)), dxy[1] * np.cos(np.radians(40)), np.sin(np.radians(40))])   # corridor, 40 deg elevation
        P0 = tgt_p + 0.10 * corr
        lat = np.array([-dxy[1], dxy[0], 0.0])                                                                          # horizontal, perpendicular to corridor
        goal_p, goal_R, off_dir, off_m, radio_shift = P0.copy(), tgt_R, None, 0.0, None
        if cond.startswith("yaw"): a = float(cond[3:]); goal_R = R.from_euler("z", a, degrees=True) * tgt_R; off_m = np.radians(a)   # before the "y" branch (09-22 fix)
        elif cond.startswith("y"): m = float(cond[1:]) / 100; goal_p = P0 + m * lat; off_dir, off_m = lat * np.sign(m), abs(m)
        elif cond.startswith("z"): m = float(cond[1:]) / 100; goal_p = P0 + m * np.array([0, 0, 1.0]); off_dir, off_m = np.array([0, 0, np.sign(m)]), abs(m)
        elif cond.startswith("vis_"):
            ax, m = cond[4], float(cond[5:]) / 100
            vec = lat if ax == "y" else -corr * np.array([1, 1, 0]) / (np.linalg.norm(corr[:2]) + 1e-9)   # x = along the corridor (horizontal)
            radio_shift = -m * vec; off_dir, off_m = vec * np.sign(m), abs(m)
        ok, dd, oerr = self._servo(goal_p, goal_R, hold, 200, stride=0.006, w_orn=1.0, done=0.012, orn_done=0.08)
        if radio_shift is not None:
            radio = [o for o in og.sim.scenes[0].objects if "radio" in o.name.lower()][0]
            p, q = radio.get_position_orientation(); radio.set_position_orientation(position=th.as_tensor(_np(p) + radio_shift, dtype=th.float32), orientation=q)
            for _ in range(5): last = self.env.step(self._hold(), n_render_iterations=1)
        hold = self._hold()
        for _ in range(5): last = self.env.step(hold, n_render_iterations=1)
        pS, RS = self._poseR(); pRad2, _ = self._radio()
        self._rec = {"tag": TAG, "state": STATE, "cond": cond, "sample": sample, "servo_ok": bool(ok), "servo_d": dd, "servo_oerr": oerr,
                     "tgt_p": tgt_p.tolist(), "P0": P0.tolist(), "goal_p": goal_p.tolist(), "p_start": pS.tolist(), "radio_p": pRad2.tolist(),
                     "off_dir": None if off_dir is None else off_dir.tolist(), "off_m": off_m, "corr": corr.tolist(), "lat": lat.tolist(),
                     "oerr_start": float(np.linalg.norm((tgt_R * RS.inv()).as_rotvec())), "tgt_R_quat": tgt_R.as_quat().tolist()}
        self._trace = [pS.tolist()]; self._done = False
        print(f"JAC_READY cond={cond} sample={sample} servo_ok={ok} d={dd:.3f} oerr={oerr:.3f} |P0-tgt|={np.linalg.norm(P0 - tgt_p):.3f}", flush=True)
        obs = last[0] if isinstance(last, tuple) else last
        return (self._inject(obs), info)

    def _finish(self):
        from scipy.spatial.transform import Rotation as R
        r = self._rec; tr = np.array(self._trace); pS, pEnd = tr[0], tr[-1]; disp = pEnd - pS
        tgt = np.array(r["tgt_p"]); to_tgt = (tgt - pS) / (np.linalg.norm(tgt - pS) + 1e-9)
        r["p_end"] = pEnd.tolist(); r["disp"] = disp.tolist(); r["disp_norm"] = float(np.linalg.norm(disp)); r["progress"] = float(disp @ to_tgt)
        r["dist_start"] = float(np.linalg.norm(tgt - pS)); r["dist_end"] = float(np.linalg.norm(tgt - pEnd)); r["n_steps"] = len(tr) - 1
        r["restore_raw"] = None if r["off_dir"] is None else float(-(disp @ np.array(r["off_dir"])))
        try:
            _, RE = self._poseR(); r["oerr_end"] = float(np.linalg.norm((R.from_quat(r["tgt_R_quat"]) * RE.inv()).as_rotvec()))
        except Exception: r["oerr_end"] = None
        r["grasp"] = bool(self._stats.get("ag_weld_right_step") is not None)
        with open(f"{OUT}/{TAG}.jsonl", "a") as f: f.write(json.dumps(r) + "\n")
        print(f"JAC_RESULT cond={r['cond']} s={r['sample']} progress={r['progress']:+.3f} restore_raw={r['restore_raw']} dist {r['dist_start']:.3f}->{r['dist_end']:.3f} oerr {r['oerr_start']:.2f}->{r['oerr_end']}", flush=True)
        self._rec = None; self._trace = []

    def step(self, action, n_render_iterations=1):
        out = StageV2AffordanceWrapper.step(self, action, n_render_iterations=n_render_iterations)
        if self._rec is None or self._done: return out
        p, _ = self._poseR(); self._trace.append(p.tolist())
        if len(self._trace) - 1 >= STEPS:
            self._done = True; self._finish()
            if isinstance(out, tuple) and len(out) >= 5: out = (out[0], out[1], out[2], True, out[4])
        return out

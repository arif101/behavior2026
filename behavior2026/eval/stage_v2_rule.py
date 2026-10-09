"""Causal, serve-side mirror of the v2 labels (RELABEL_V2.md). Pure numpy; no sim, no object state.
Inputs per step: the (predicted) toggle-button position in the BASE frame and the two EE positions from proprio.
Outputs: stage_v2 in {0 approach, 1 grasp, 2 transport, 3 press}, progress in [0, 1], and the stage-indexed target
(rail = button + 0.141 z before lift, button after). Differences from the offline labels, by necessity:
  * lifted = button base-z rises > LIFT_DZ over the running pre-lift minimum (offline: metalink world z over frame 0).
  * press = lifted and the FREE hand within REACH_R of the button (offline: the final reach before the TOGGLE, which is
    not observable causally). Fires at the first inward crossing; humans hover the free hand early, so this can lead the
    label — the benign direction (the policy reaches earlier).
  * progress = (stage + elapsed-in-stage / typical stage length) / 4 with typical lengths from the human demos.
"""
import numpy as np

RAIL_UP = np.array([0.0, 0.0, 0.141], np.float32)
GRASP_R, REACH_R, LIFT_DZ = 0.12, 0.20, 0.08   # reach 0.20: joint label+mirror sweep (map 99.6%, episodes 98.8%)
# 2026-09-22: LIFT_DZ 0.03 -> 0.08 and the lift now also requires a CLOSED gripper for LIFT_HOLD consecutive steps. With
# the served pointer 0.2-0.3 m noisy, a 3 cm rise of the predicted button fired on noise: every probe episode jumped
# 0 -> 2 within a dozen steps with the hand 10 cm away and the gripper open (stage_v2_wrapper_stats 09-22). A
# stage-conditioned policy then ran under "transport" conditioning while reaching (closed gripper, lift), and the
# gripper-reopen rule (which requires NOT lifted) was blocked.
LIFT_HOLD = 10
TYPICAL_LEN = np.array([1065.0, 189.0, 154.0, 741.0])   # human demos: frames per stage (map source, 200 eps)


class StageV2Tracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.z_min = None; self.lifted = False; self.hold_R = None; self.press = False
        self.stage = 0; self.stage_t0 = 0; self.t = 0; self._lift_run = 0

    def step(self, p_button_base, ee_L, ee_R, grip_closed=True):
        """All three in the base frame; grip_closed = either gripper reads closed (proprio). Returns
        (stage, progress, target_points_v2[2,3])."""
        p = np.asarray(p_button_base, np.float64); eL = np.asarray(ee_L, np.float64); eR = np.asarray(ee_R, np.float64)
        bL, bR = np.linalg.norm(p - eL), np.linalg.norm(p - eR)
        if not self.lifted:
            self.z_min = p[2] if self.z_min is None else min(self.z_min, p[2])
            if p[2] - self.z_min > LIFT_DZ and grip_closed:
                self._lift_run += 1
            else:
                self._lift_run = 0
            if self._lift_run >= LIFT_HOLD:
                self.lifted = True; self.hold_R = bR <= bL
        raw = 0
        if not self.lifted:
            rail = p + RAIL_UP; d_near = min(np.linalg.norm(rail - eL), np.linalg.norm(rail - eR))
            raw = 1 if d_near <= GRASP_R else 0
        else:
            b_free = bL if self.hold_R else bR
            if b_free <= REACH_R: self.press = True
            raw = 3 if self.press else 2
        stage = max(self.stage, raw)                      # monotone
        if stage != self.stage: self.stage, self.stage_t0 = stage, self.t
        frac = min((self.t - self.stage_t0) / TYPICAL_LEN[stage], 0.99)
        progress = (stage + frac) / 4.0
        tgt = p + (RAIL_UP if stage <= 1 else 0.0)
        pts = np.stack([tgt - eL, tgt - eR]).astype(np.float32)
        self.t += 1
        return int(stage), float(progress), pts

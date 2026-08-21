"""STAGED (not applied): anti-drift shaping for held-object press phases.
Apply with:  python3 /root/staged_antidrift_patch.py  && restart trainer as v21q.

Counter to the filmed failure mode: during press (bridge-press 'H' and chain phase 2),
the right arm carries the held radio off-station. Anchors the radio's position when the
press context begins; penalizes displacement beyond a 5 cm deadzone, potential-based.
"""

W = "/root/skill_env_wrapper_v2.py"
T = "/root/train_skill_v2.py"

src = open(W).read()

# 1) anchor state in reset (press-family entries holding the target: H bridges + rungs)
src = src.replace(
    """        self._lift_dwell = 0
        return self._obs()""",
    """        self._lift_dwell = 0
        # anti-drift anchor (v21q): where the held object should STAY during pressing
        self.hold_anchor = (_np(self.target.get_position_orientation()[0]).copy()
                            if e.get("holding_arm") else None)
        return self._obs()""", 1)

# 2) shaping term in step(), potential-based on displacement beyond deadzone
src = src.replace(
    """            if self.entry.get("family") == "pick_up_from":""",
    """            if (self.entry.get("family") == "press"
                    and getattr(self, "hold_anchor", None) is not None):
                _disp = float(np.linalg.norm(
                    _np(self.target.get_position_orientation()[0]) - self.hold_anchor))
                _ex = max(0.0, _disp - 0.05)
                if not hasattr(self, "_drift_pot"):
                    self._drift_pot = 0.0
                r -= 1.5 * (_ex - self._drift_pot)   # potential-based: pays going BACK
                self._drift_pot = _ex
            if self.entry.get("family") == "pick_up_from":""", 1)

open(W, "w").write(src)

# 3) trainer: set the anchor at the chain's goal switch
src2 = open(T).read()
src2 = src2.replace(
    """                    env.target_base = env._live_target_base()
                    done = False
                    obs = env._obs()
                    continue""",
    """                    env.target_base = env._live_target_base()
                    from skill_env_wrapper import _np as __np
                    env.hold_anchor = __np(
                        env.target.get_position_orientation()[0]).copy()
                    env._drift_pot = 0.0
                    done = False
                    obs = env._obs()
                    continue""", 1)
open(T, "w").write(src2)

import ast
ast.parse(open(W).read()); ast.parse(open(T).read())
print("anti-drift patch APPLIED; restart trainer as v21q")

"""AffordanceMapFullRes + the causal v2-label mirror (stage_v2_rule.StageV2Tracker). Exposes `_last_v2` =
{"target_points_v2", "stage_v2", "progress"} for the evaluator passthrough (patch_v2_passthrough.py); the parent's
`_last` (button-relative target_points) stays for v1 checkpoints. Stats -> /root/stage_v2_wrapper_stats.json."""
import json
import numpy as np
from behavior2026_eval.affordance_map_fullres import AffordanceMapFullRes, EEF_LEFT, EEF_RIGHT
from behavior2026_eval.stage_v2_rule import StageV2Tracker


class StageV2AffordanceWrapper(AffordanceMapFullRes):
    def __init__(self, env):
        super().__init__(env)
        self._tracker = StageV2Tracker(); self._last_v2 = None; self._v2_stats = {"episodes": [], "cur": []}

    def _proprio(self, obs):
        name = self._robot.name; node = obs.get(name)
        if isinstance(node, dict) and "proprio" in node: return np.asarray(node["proprio"], np.float64).reshape(-1)
        if f"{name}::proprio" in obs: return np.asarray(obs[f"{name}::proprio"], np.float64).reshape(-1)
        return None

    def _inject(self, obs):
        obs = super()._inject(obs)
        try:
            self._last_v2 = None
            if self._last is None or not isinstance(obs, dict): return obs
            prop = self._proprio(obs)
            if prop is None: return obs
            eL, eR = prop[EEF_LEFT], prop[EEF_RIGHT]
            p_button = np.asarray(self._last[1], np.float64) + eR       # recover the predicted button (base frame)
            grip_closed = bool(prop.shape[0] >= 51 and (float(np.mean(prop[49:51])) < 0.02 or float(np.mean(prop[24:26])) < 0.02))
            stage, progress, pts = self._tracker.step(p_button, eL, eR, grip_closed=grip_closed)
            self._last_v2 = {"target_points_v2": pts, "stage_v2": np.int32(stage), "progress": np.float32(progress)}
            self._v2_stats["cur"].append(stage)
            self._stats.setdefault("stage_series_full", []).append(int(stage))   # per-step served stage (09-22)
        except Exception as e:  # never take down a rollout over conditioning
            self._last_v2 = None
        return obs

    def _dump_stats(self):
        super()._dump_stats()
        try:   # append the served-stage timeline to the wrapper stats file (per-rollout processes never call reset() at the end)
            import json as _j
            d = _j.load(open("/root/affordance_wrapper_stats.json")); st = self._stats.get("stage_series_full", [])
            d["stage_series"] = st[::10]; d["stage_counts"] = np.bincount(np.asarray(st, int), minlength=4).tolist() if st else None
            d["stage_first"] = [int(np.argmax(np.asarray(st) >= c)) if st and (np.asarray(st) >= c).any() else None for c in range(4)]
            _j.dump(d, open("/root/affordance_wrapper_stats.json", "w"), indent=1)
        except Exception:
            pass

    def reset(self):
        if self._v2_stats["cur"]:
            st = np.asarray(self._v2_stats["cur"]); self._v2_stats["episodes"].append({"n": int(len(st)), "counts": np.bincount(st, minlength=4).tolist(),
                                                                                       "first": [int(np.argmax(st >= c)) if (st >= c).any() else None for c in range(4)]})
            try: json.dump(self._v2_stats["episodes"], open("/root/stage_v2_wrapper_stats.json", "w"), indent=1)
            except Exception: pass
        self._v2_stats["cur"] = []; self._tracker.reset(); self._last_v2 = None
        return super().reset()

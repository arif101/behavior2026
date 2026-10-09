"""CloseVetoWrapper (2026-09-20 diagnostic, sim-state read; never an eval arm).
StageV2AffordanceWrapper + a gate on the policy's gripper-close command: while the RIGHT EE is farther than VETO_R from
the toggle button (ground truth; humans close at 0.13-0.185 m EE->button), the right-gripper action is forced OPEN.
Separates "the policy closes at the wrong time" (vetoed -> it keeps reaching and closes in the human band) from "the
reach itself dies short" (vetoed -> it hovers open and never gets closer). Env: VETO_R (0.19), VETO_LEFT (1 = also gate
the left gripper). Prints VETO_SUMMARY at reset; stats in /root/close_veto_stats.json."""
import json, os
import numpy as np
from behavior2026_eval.stage_v2_wrapper import StageV2AffordanceWrapper

VETO_R = float(os.environ.get("VETO_R", 0.19)); VETO_LEFT = os.environ.get("VETO_LEFT", "1") == "1"
A_GRIP_L, A_GRIP_R = 14, 22


def _np(x):
    try: return x.detach().cpu().numpy()
    except Exception: return np.asarray(x)


class CloseVetoWrapper(StageV2AffordanceWrapper):
    def __init__(self, env):
        super().__init__(env); self._v = {"n": 0, "vetoed_R": 0, "vetoed_L": 0, "first_allowed_close_R": None, "dist_at_first_close_R": None, "closes_attempted_R": 0, "min_dR": 9.9}

    def _dists(self):
        tp = _np(self._targets[0].get_position_orientation()[0]).astype(np.float64).reshape(3)
        eR = _np(self._robot.eef_links["right"].get_position_orientation()[0]).astype(np.float64).reshape(3)
        eL = _np(self._robot.eef_links["left"].get_position_orientation()[0]).astype(np.float64).reshape(3)
        return float(np.linalg.norm(tp - eR)), float(np.linalg.norm(tp - eL))

    def step(self, action, n_render_iterations=1):
        try:
            dR, dL = self._dists(); v = self._v; v["n"] += 1; v["min_dR"] = min(v["min_dR"], dR)
            a = action
            is_torch = hasattr(a, "cpu")
            arr = _np(a).copy()
            flat = arr.reshape(-1, arr.shape[-1])
            if flat[0, A_GRIP_R] < 0.0:
                v["closes_attempted_R"] += 1
                if dR > VETO_R:
                    flat[0, A_GRIP_R] = 1.0; v["vetoed_R"] += 1
                elif v["first_allowed_close_R"] is None:
                    v["first_allowed_close_R"] = v["n"]; v["dist_at_first_close_R"] = round(dR, 3)
                    print(f"VETO_ALLOWED_CLOSE_R step={v['n']} dR={dR:.3f}", flush=True)
            if VETO_LEFT and flat[0, A_GRIP_L] < 0.0 and dL > VETO_R:
                flat[0, A_GRIP_L] = 1.0; v["vetoed_L"] += 1
            if is_torch:
                import torch as th
                a = th.as_tensor(arr.reshape(_np(action).shape), dtype=action.dtype, device=action.device)
            else:
                a = arr.reshape(np.asarray(action).shape)
            if v["n"] % 500 == 0:
                print(f"VETO step={v['n']} dR={dR:.3f} vetoed_R={v['vetoed_R']} attempts_R={v['closes_attempted_R']} min_dR={v['min_dR']:.3f}", flush=True)
                json.dump(v, open("/root/close_veto_stats.json", "w"))
            action = a
        except Exception as e:  # never take down a rollout
            print(f"VETO_ERR {e!r}", flush=True)
        return super().step(action, n_render_iterations=n_render_iterations)

    def reset(self):
        if self._v["n"]:
            print(f"VETO_SUMMARY {json.dumps(self._v)}", flush=True)
            json.dump(self._v, open("/root/close_veto_stats.json", "w"))
        self._v = {"n": 0, "vetoed_R": 0, "vetoed_L": 0, "first_allowed_close_R": None, "dist_at_first_close_R": None, "closes_attempted_R": 0, "min_dR": 9.9}
        return super().reset()

"""GripperReopenWrapper (2026-09-22, LEGAL serving rule; no sim-state reads). The 2025 winner's largest measured gain
("if the gripper is closed and it was never closed in training data for this task at the same stage, treat it as a failed
grasp and completely open the gripper", ~2x on grasp-limited tasks), on our stage tracker.

Rule, per hand: if the gripper joints (proprio 49:51 right / 24:26 left) read CLOSED for REOPEN_N consecutive steps while
the StageV2 tracker has NOT observed a lift (predicted button z rising > 3 cm), the grasp has failed -> force the
gripper command OPEN for REOPEN_M steps, then hand the channel back (cooldown REOPEN_COOL before the next reopen). Up
to REOPEN_MAX events per episode. Env: REOPEN_N 150 (5 s), REOPEN_M 45, REOPEN_COOL 90, REOPEN_MAX 8, REOPEN_Q 0.02
(closed threshold on mean gripper qpos; open ~0.05, closed ~0.001-0.01). Stats -> /root/gripper_reopen_stats.json."""
import json, os
import numpy as np
from behavior2026_eval.stage_v2_wrapper import StageV2AffordanceWrapper

N = int(os.environ.get("REOPEN_N", 150)); M = int(os.environ.get("REOPEN_M", 45)); COOL = int(os.environ.get("REOPEN_COOL", 90))
MAXE = int(os.environ.get("REOPEN_MAX", 8)); QCL = float(os.environ.get("REOPEN_Q", 0.02))
A_GRIP = {"left": 14, "right": 22}; Q_GRIP = {"left": slice(24, 26), "right": slice(49, 51)}


def _np(x):
    try: return x.detach().cpu().numpy()
    except Exception: return np.asarray(x)


class GripperReopenWrapper(StageV2AffordanceWrapper):
    def __init__(self, env):
        super().__init__(env); self._last_prop = None; self._reset_state()

    def _reset_state(self):
        self._n = 0; self._closed = {"left": 0, "right": 0}; self._force_until = {"left": -1, "right": -1}; self._cool_until = {"left": -1, "right": -1}
        self._ev = {"events": [], "n_reopen": {"left": 0, "right": 0}, "closed_steps": {"left": 0, "right": 0}}

    def _inject(self, obs):
        obs = super()._inject(obs)
        try:
            name = self._robot.name; node = obs.get(name) if isinstance(obs, dict) else None
            if isinstance(node, dict) and "proprio" in node: self._last_prop = np.asarray(node["proprio"], np.float64).reshape(-1)
            elif isinstance(obs, dict) and f"{name}::proprio" in obs: self._last_prop = np.asarray(obs[f"{name}::proprio"], np.float64).reshape(-1)
        except Exception: pass
        return obs

    def step(self, action, n_render_iterations=1):
        try:
            self._n += 1; prop = self._last_prop
            lifted = bool(getattr(self._tracker, "lifted", False))
            arr = _np(action).copy(); flat = arr.reshape(-1, arr.shape[-1]); changed = False
            for hand in ("left", "right"):
                if prop is not None and prop.shape[0] >= 51 and float(np.mean(prop[Q_GRIP[hand]])) < QCL:
                    self._closed[hand] += 1; self._ev["closed_steps"][hand] += 1
                else:
                    self._closed[hand] = 0
                if (self._closed[hand] >= N and not lifted and self._n > self._cool_until[hand]
                        and self._ev["n_reopen"][hand] < MAXE and self._n > self._force_until[hand]):
                    self._force_until[hand] = self._n + M; self._cool_until[hand] = self._n + M + COOL
                    self._ev["n_reopen"][hand] += 1; self._closed[hand] = 0
                    d = None
                    try:
                        if self._last is not None: d = round(float(np.linalg.norm(self._last[1 if hand == "right" else 0])), 3)
                    except Exception: pass
                    self._ev["events"].append({"hand": hand, "step": self._n, "stage": int(getattr(self._tracker, "stage", -1)), "hand_to_pred_target": d})
                    print(f"REOPEN hand={hand} step={self._n} stage={getattr(self._tracker, 'stage', -1)} hand_to_pred={d}", flush=True)
                if self._n <= self._force_until[hand]:
                    flat[0, A_GRIP[hand]] = 1.0; changed = True
            if changed:
                if hasattr(action, "cpu"):
                    import torch as th
                    action = th.as_tensor(arr.reshape(_np(action).shape), dtype=action.dtype, device=action.device)
                else:
                    action = arr.reshape(np.asarray(action).shape)
            if self._n % 500 == 0:
                json.dump(self._ev, open("/root/gripper_reopen_stats.json", "w"))
        except Exception as e:
            print(f"REOPEN_ERR {e!r}", flush=True)
        return super().step(action, n_render_iterations=n_render_iterations)

    def reset(self):
        if self._n:
            print(f"REOPEN_SUMMARY {json.dumps(self._ev)}", flush=True)
            try: json.dump(self._ev, open("/root/gripper_reopen_stats.json", "w"))
            except Exception: pass
        self._reset_state()
        return super().reset()

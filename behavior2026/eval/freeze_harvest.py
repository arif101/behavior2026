"""FreezeHarvestWrapper (2026-09-17 freeze diagnostic, DIAGNOSTIC ONLY - never an eval arm).

AffordanceMapFullRes + a stationarity detector. Once the robot has been motionless for FREEZE_WIN steps after step
FREEZE_MIN (both EEs moved < FREEZE_EE m, arm+trunk joints < FREEZE_Q rad, mean base |twist| < FREEZE_V), the wrapper
dumps the full sim state to FREEZE_OUT/<FREEZE_TAG>.npz (+ .json) and ends the episode (truncated=True). The dump is a
sim read used only to seed the continue-from-freeze test (freeze_continue.py); nothing here feeds the policy, and the
policy sees exactly the parity serving stack. If the policy never freezes (grasp/weld, or the step cap), no file is
written and the runner records NO_FREEZE.
Env: FREEZE_OUT (default /root/freeze_states), FREEZE_TAG (file stem), FREEZE_WIN 150, FREEZE_MIN 450, FREEZE_EE 0.02,
     FREEZE_Q 0.03, FREEZE_V 0.01.
"""
import json
import os

import numpy as np

from behavior2026_eval.affordance_map_fullres import AffordanceMapFullRes, EEF_LEFT, EEF_RIGHT

Q_ARM_L, Q_ARM_R, Q_TRUNK = slice(3, 10), slice(28, 35), slice(53, 57)   # 61-d eval proprio (skill_env_wrapper.P)
WIN = int(os.environ.get("FREEZE_WIN", 150))
MIN_STEP = int(os.environ.get("FREEZE_MIN", 450))
EE_TOL = float(os.environ.get("FREEZE_EE", 0.04))    # 09-17: S1 idles with the arm dithering (cmd-norm std 0.15); 0.02/0.03 never fired on 301
Q_TOL = float(os.environ.get("FREEZE_Q", 0.10))
V_TOL = float(os.environ.get("FREEZE_V", 0.02))
CAP = int(os.environ.get("FREEZE_CAP", 1500))       # fallback: dump the state at this step if parked-but-dithering never met the bars
NEAR_R = float(os.environ.get("FREEZE_NEAR_R", 0))    # >0: ALSO dump the first time the right EE is within NEAR_R of the target (near-grasp harvest, 09-21)
OUT = os.environ.get("FREEZE_OUT", "/root/freeze_states")
TAG = os.environ.get("FREEZE_TAG", "untagged")


def _np(x):
    try:
        return x.detach().cpu().numpy()
    except Exception:  # noqa: BLE001
        return np.asarray(x)


def proprio_of(robot_name, obs):
    node = obs.get(robot_name) if isinstance(obs, dict) else None
    if isinstance(node, dict) and "proprio" in node:
        return np.asarray(node["proprio"], np.float64).reshape(-1)
    if isinstance(obs, dict) and f"{robot_name}::proprio" in obs:
        return np.asarray(obs[f"{robot_name}::proprio"], np.float64).reshape(-1)
    return None


class FreezeHarvestWrapper(AffordanceMapFullRes):
    def __init__(self, env):
        super().__init__(env)
        self._hist = []
        self._frozen = False
        self._n = 0

    def _record(self, obs):
        prop = proprio_of(self._robot.name, obs)
        if prop is None or prop.shape[0] < 61:
            return
        self._n += 1
        self._hist.append(np.concatenate([prop[EEF_LEFT], prop[EEF_RIGHT], prop[0:3], prop[Q_ARM_L], prop[Q_ARM_R], prop[Q_TRUNK]]))
        if len(self._hist) > WIN:
            self._hist.pop(0)

    def _metrics(self):
        H = np.stack(self._hist)
        now = H[-1]
        dL = np.linalg.norm(H[:, 0:3] - now[0:3], axis=1).max()
        dR = np.linalg.norm(H[:, 3:6] - now[3:6], axis=1).max()
        v = np.abs(H[:, 6:9]).mean()
        dq = np.abs(H[:, 9:] - now[9:]).max()
        return float(dL), float(dR), float(v), float(dq)

    def _stationary(self):
        if self._n < MIN_STEP or len(self._hist) < WIN:
            return False
        dL, dR, v, dq = self._metrics()
        if self._n % 100 == 0:
            print(f"FREEZE_METRICS step={self._n} dL={dL:.3f} dR={dR:.3f} v={v:.4f} dq={dq:.3f}", flush=True)
        return bool(dL < EE_TOL and dR < EE_TOL and v < V_TOL and dq < Q_TOL)

    def _dump_freeze(self, obs):
        import omnigibson as og
        os.makedirs(OUT, exist_ok=True)
        state = _np(og.sim.dump_state(serialized=True)).astype(np.float32)
        rob = self._robot
        bp, bq = rob.get_position_orientation()
        bp, bq = _np(bp), _np(bq)
        radio_pos = None
        try:
            radio = [o for o in og.sim.scenes[0].objects if "radio" in o.name.lower()][0]
            radio_pos = _np(radio.get_position_orientation()[0])
        except Exception:  # noqa: BLE001
            pass
        prop = proprio_of(rob.name, obs)
        s = self._stats
        meta = {"tag": TAG, "step": int(self._n), "base_pos": bp.tolist(), "base_quat": bq.tolist(),
                "radio_pos": None if radio_pos is None else radio_pos.tolist(),
                "base_to_radio_xy": None if radio_pos is None else float(np.linalg.norm((radio_pos - bp)[:2])),
                "dist_L_last": (s.get("dist_L") or [None])[-1], "dist_R_last": (s.get("dist_R") or [None])[-1],
                "wrist_angL_last": (s.get("wrist_angL") or [None])[-1], "n_inject": s.get("n_inject"), "n_steps": s.get("n_steps"),
                "state_len": int(state.shape[0])}
        np.savez(f"{OUT}/{TAG}.npz", state=state, proprio=prop, base_pos=bp, base_quat=bq,
                 radio_pos=np.zeros(3) if radio_pos is None else radio_pos, step=np.int64(self._n))
        json.dump(meta, open(f"{OUT}/{TAG}.json", "w"), indent=1)
        return meta

    def step(self, action, n_render_iterations=1):
        out = super().step(action, n_render_iterations=n_render_iterations)
        if self._frozen or not isinstance(out, tuple) or len(out) < 5:
            return out
        self._record(out[0])
        stationary = self._stationary()
        at_cap = (self._n >= CAP) and not self._stats.get("grasp") and self._stats.get("ag_weld_right_step") is None
        near = False
        if NEAR_R > 0 and self._n > 100:
            dr = self._stats.get("dist_R_true") or []
            near = bool(dr) and dr[-1] < NEAR_R
        if stationary or at_cap or near:
            self._frozen = True
            try:
                meta = self._dump_freeze(out[0])
                meta["stationary"] = bool(stationary); meta["near"] = bool(near)
                json.dump(meta, open(f"{OUT}/{TAG}.json", "w"), indent=1)
                print(f"FREEZE_HARVESTED tag={TAG} step={self._n} stationary={int(stationary)} base_to_radio={meta['base_to_radio_xy']} "
                      f"distL={meta['dist_L_last']} wristL={meta['wrist_angL_last']}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"FREEZE_DUMP_FAILED {e!r}", flush=True)
            out = (out[0], out[1], out[2], True, out[4])
        return out

    def reset(self):
        self._hist = []
        self._frozen = False
        self._n = 0
        return super().reset()


class FreezeHarvestV2Wrapper(FreezeHarvestWrapper, __import__("behavior2026_eval.stage_v2_wrapper", fromlist=["StageV2AffordanceWrapper"]).StageV2AffordanceWrapper):
    """Same harvest on top of the stage-v2 passthrough (for checkpoints served with StageV2AffordanceWrapper, e.g. `full`)."""
    pass

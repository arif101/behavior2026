"""Counterfactual probe: is the policy reading the gripper command off its own PROPRIO?

THE HYPOTHESIS (causal confusion / copycat shortcut)
---------------------------------------------------
de Haan et al. NeurIPS 2019 (arXiv 1905.11979); Wen et al. NeurIPS 2020 (arXiv 2010.14876):
an imitator can "cheat" by predicting from a nuisance correlate of the expert action rather than
from the causal input. Here the suspect correlate is the gripper's own proprioceptive state.

It uniquely explains our numbers, which COVARIATE SHIFT DOES NOT:
  open-loop, teacher-forced on training frames : gripper closure recall 0.993
  closed-loop, on its own rollout states       : gripper pinned open, 100% of steps, std 0.005
Compounding error predicts ERRATIC, high-variance, out-of-distribution actions. We measured the
opposite -- mean |z| 0.64, 1% beyond 3 sigma, SMOOTHER than training. That is collapse onto the
unconditional mean, i.e. a degenerate conditioning signal, not a drifted state.

Mechanism if true: open-loop the gripper proprio is the ground-truth answer, so recall ~1.0.
Closed-loop the gripper starts open, the model copies "open", which keeps it open -- a
self-reinforcing fixed point. Codevilla ICCV 2019 (arXiv 1904.08980) documents the identical
pathology on ego-velocity as the "inertia problem", and notes it WORSENS with more data.

WHAT THIS DOES
--------------
For every inference call, run the policy TWICE on the same observation:
  (1) unmodified                      -> what it actually commands
  (2) gripper proprio dims overwritten with the CLOSED value -> counterfactual
and log both gripper outputs.

  If (2) flips to closed while (1) stays open  -> SHORTCUT CONFIRMED. The policy is conditioning
      on its own gripper state, not on vision. Fix is a representation change (drop/noise the
      gripper proprio dims, or an auxiliary vision-only head), NOT noise augmentation --
      Wen et al. report "Dropout-BC performs uniformly poorly across all tasks".
  If (2) stays open too -> the gripper proprio is NOT the culprit; look elsewhere.

Needs the R1Pro proprio layout. b1k.py: proprio is base_qvel[0:3], trunk[3:7], left_arm[7:14],
left_grip[14], right_arm[15:22], right_grip[22] -- 23 dims -- but the SERVED observation carries
the full 61-dim state, so the gripper indices are resolved from the robot config at runtime rather
than hardcoded.
"""

import argparse
import pathlib

INSERT_AFTER = "                action = self._policy.act(obs)\n"

PROBE = '''
                # --- copycat counterfactual (patch_copycat_probe.py) -----------------------
                try:
                    import json as _json
                    from copy import deepcopy as _dc

                    import numpy as _np

                    _key = f"{self._policy.robot.name}::proprio"
                    _gidx = list(getattr(self._policy, "gripper_indices", []) or [])
                    if _key in obs and _gidx:
                        _cf = _dc(obs)
                        _p = _np.array(_cf[_key], dtype=_np.float32)
                        # overwrite ONLY the gripper proprio dims with the CLOSED value (-1)
                        _flatp = _p.reshape(-1)
                        for _gi in _gidx:
                            if _gi < _flatp.shape[0]:
                                _flatp[_gi] = -1.0
                        _cf[_key] = _flatp.reshape(_p.shape)
                        _cf_act = self._policy.act(_cf)
                        _cf_a = _cf_act.cpu().numpy() if hasattr(_cf_act, "cpu") else _np.asarray(_cf_act)
                        _a = action.cpu().numpy() if hasattr(action, "cpu") else _np.asarray(action)
                        _f0 = _np.asarray(_a).reshape(-1, _np.asarray(_a).shape[-1])[0]
                        _f1 = _np.asarray(_cf_a).reshape(-1, _np.asarray(_cf_a).shape[-1])[0]
                        with open("/root/copycat_log.jsonl", "a") as _fh:
                            _fh.write(_json.dumps({
                                "gripper_indices": _gidx,
                                "real_grip": [float(_f0[g]) for g in _gidx if g < _f0.shape[0]],
                                "cf_closed_grip": [float(_f1[g]) for g in _gidx if g < _f1.shape[0]],
                                "real_base": [float(x) for x in _f0[0:3]],
                                "cf_base": [float(x) for x in _f1[0:3]],
                            }) + "\\n")
                except Exception:  # never let the probe break the eval
                    pass
                # --------------------------------------------------------------------------
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="/root/openpi_fork/src/openpi/serving/websocket_b1k_server.py")
    a = ap.parse_args()
    p = pathlib.Path(a.server)
    s = p.read_text()
    if "patch_copycat_probe.py" in s:
        raise SystemExit("already patched")
    if INSERT_AFTER not in s:
        raise SystemExit(f"anchor not found in {p}")
    p.write_text(s.replace(INSERT_AFTER, INSERT_AFTER + PROBE, 1))
    print(f"patched {p} -> logs to /root/copycat_log.jsonl")


if __name__ == "__main__":
    main()

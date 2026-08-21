"""Re-attach target_points AFTER flatten, where task_id is added and survives.

THE PROBLEM, MEASURED
---------------------
OraclePointFullRes injects obs["target_points"] on every step -- verified, n_injections=401 with
real values ([1.177, 0.653, -0.209] etc) against the correct object (radio_89). But a FRESH capture
of what the policy server receives contains only:

    robot_r1::cam_rel_poses, robot_r1::proprio, 3x rgb, 3x depth_linear, task_id

No target_points. And the model, captured after Observation.from_dict, gets
target_points [0,0,0,0,0,0] with mask [False, False] -- B1KInputs' "no point" sentinel.

So the points are injected and then lost before the websocket payload is packed. Note that
`task_id` DOES survive, and it is added at evaluator.py:349 -- i.e. AFTER
`obs = flatten_obs_dict(obs)` on line 331. Anything the wrapper attaches earlier does not make it
through, whatever the reason.

THE FIX
-------
Do what task_id does: attach after the flatten, inside _preprocess_obs, reading the value the
wrapper already stores on itself (`self._last`). This sidesteps the question of where exactly the
earlier attachment is lost, and it is verifiable end-to-end -- the serving capture either shows
target_points or it does not.

SCOPE: this is for the ORACLE diagnostic, which reads object poses from the simulator and is NOT a
submission path. It exists to separate targeting from motor competence. The honest source is the
grounding head, and that will need the same plumbing.
"""

import argparse
import pathlib

ANCHOR = '        obs["task_id"] = th.tensor([TASK_NAMES_TO_INDICES[self.cfg.task.name]], dtype=th.int64)\n'

PATCH = '''        # --- point conditioning pass-through (patch_point_passthrough.py) -------------
        # Attach AFTER flatten_obs_dict, exactly like task_id above: anything the env wrapper
        # attaches BEFORE the flatten does not survive into the websocket payload (measured).
        try:
            import numpy as _np

            _w = self.env
            for _ in range(6):  # walk the wrapper chain
                _pts = getattr(_w, "_last", None)
                if _pts is not None:
                    obs["target_points"] = _np.asarray(_pts, dtype=_np.float32)
                    obs["target_points_mask"] = _np.array([True, True], dtype=bool)
                    break
                _w = getattr(_w, "env", None)
                if _w is None:
                    break
        except Exception:
            pass
        # ------------------------------------------------------------------------------
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluator", default="/root/bw/BEHAVIOR-1K/OmniGibson/omnigibson/eval/evaluator.py")
    a = ap.parse_args()
    p = pathlib.Path(a.evaluator)
    s = p.read_text()
    if "patch_point_passthrough.py" in s:
        raise SystemExit("already patched")
    if ANCHOR not in s:
        raise SystemExit(f"anchor not found in {p}")
    p.write_text(s.replace(ANCHOR, ANCHOR + PATCH, 1))
    print(f"patched {p}: target_points now attached post-flatten")


if __name__ == "__main__":
    main()

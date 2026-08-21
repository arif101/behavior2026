"""Capture the exact policy-input dict the SERVING path builds, so it can be diffed against training.

WHY THIS IS THE TEST THAT DOESN'T NEED A THEORY
-----------------------------------------------
Three measurements, taken together, over-constrain the problem:
  1. open-loop on TRAINING frames  -> gripper closure recall 0.993 (the model works)
  2. closed-loop on its own states -> gripper pinned open, variance collapsed 200x (frozen, NOT
     erratic -- so this is collapse onto the unconditional mean, not compounding error)
  3. forcing the gripper PROPRIO to closed changes the output by 0.0002 (no proprio shortcut)

A model that is correct on one observation distribution, degenerate on another, and unresponsive to
the input we perturbed, is a model whose CONDITIONING IS NOT ARRIVING at eval. Rather than guess at
a mechanism (seven guesses died today), capture what the serving path actually hands the policy and
compare it, field by field, against what the training dataloader hands the model.

PRIME SUSPECT: our RGBDFullResWrapperFixed renders head 720 / wrist 480 while the model trained at
240. Both are resized to 224 for the model, but if the resize path, channel order, dtype or value
range differ between the training transform chain and the serving one, the images arrive wrong and
the policy would fall back to exactly the degenerate behaviour we measure.

Captures the dict B1KPolicyWrapper builds just before handing off to the openpi transform chain.
Writes ONE npz (the first call) and then gets out of the way.
"""

import argparse
import pathlib

INSERT_AFTER = "                obs = deepcopy(result)\n"

CAPTURE = '''
                # --- serving-input capture (patch_obs_capture.py) --------------------------
                try:
                    import os as _os

                    import numpy as _np

                    if not _os.path.exists("/root/serving_obs.npz"):
                        _flat = {}
                        for _k, _v in obs.items():
                            _a = _np.asarray(_v)
                            if _a.dtype == object:
                                continue
                            _flat[str(_k)] = _a
                        _np.savez_compressed("/root/serving_obs.npz", **_flat)
                        print("CAPTURED serving obs ->", sorted(_flat.keys()), flush=True)
                except Exception as _e:  # never break the eval
                    print("capture failed:", _e, flush=True)
                # --------------------------------------------------------------------------
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="/root/openpi_fork/src/openpi/serving/websocket_b1k_server.py")
    a = ap.parse_args()
    p = pathlib.Path(a.server)
    s = p.read_text()
    if "patch_obs_capture.py" in s:
        raise SystemExit("already patched")
    if INSERT_AFTER not in s:
        raise SystemExit(f"anchor not found in {p}")
    p.write_text(s.replace(INSERT_AFTER, INSERT_AFTER + CAPTURE, 1))
    print(f"patched {p} -> writes /root/serving_obs.npz on first inference")


if __name__ == "__main__":
    main()

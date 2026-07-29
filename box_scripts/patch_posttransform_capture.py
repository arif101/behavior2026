"""Capture the POST-transform Observation the model actually receives at serving.

policy.py:71 `inputs = self._input_transform(inputs)` is where the raw eval dict becomes what the
model sees. Capturing there gives us the apples-to-apples comparison against the training
dataloader's Observation, which is the one diff that needs no hypothesis.

Motivation: the pre-transform diff showed the serving path receives RGBA uint8 at 480/720 in
[0,255] and a 61-dim proprio, while training feeds RGB float32 224x224 in [-1,1] and a 32-dim
state. Those differences are EXPECTED to be reconciled by the transform chain. The question is
whether they actually are. If the post-transform images or state differ materially from training,
the model is being asked a question it never saw -- which would produce exactly the degenerate,
low-variance, unconditional-mean behaviour we measured (0.993 open-loop vs 0.000 closed-loop, with
gripper variance collapsed 200x rather than erratic).
"""

import argparse
import pathlib

ANCHOR = "        inputs = self._input_transform(inputs)\n"

CAPTURE = '''
        # --- post-transform capture (patch_posttransform_capture.py) ------------------
        try:
            import os as _os

            import numpy as _np

            if not _os.path.exists("/root/posttransform_obs.npz"):
                _flat = {}

                def _walk(_o, _p=""):
                    if isinstance(_o, dict):
                        for _k, _v in _o.items():
                            _walk(_v, f"{_p}/{_k}" if _p else str(_k))
                    else:
                        try:
                            _a = _np.asarray(_o)
                            if _a.dtype != object and _a.size:
                                _flat[_p] = _a
                        except Exception:
                            pass

                _walk(inputs)
                if _flat:
                    _np.savez_compressed("/root/posttransform_obs.npz", **_flat)
                    print("CAPTURED post-transform:", sorted(_flat.keys()), flush=True)
        except Exception as _e:
            print("post-transform capture failed:", _e, flush=True)
        # -----------------------------------------------------------------------------
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="/root/openpi_fork/src/openpi/policies/policy.py")
    a = ap.parse_args()
    p = pathlib.Path(a.policy)
    s = p.read_text()
    if "patch_posttransform_capture.py" in s:
        raise SystemExit("already patched")
    if ANCHOR not in s:
        raise SystemExit(f"anchor not found in {p}")
    p.write_text(s.replace(ANCHOR, ANCHOR + CAPTURE, 1))
    print(f"patched {p} -> writes /root/posttransform_obs.npz on first inference")


if __name__ == "__main__":
    main()

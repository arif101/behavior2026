"""Extend the evaluator passthrough: attach map_tokens post-flatten (like target_points).

Independent of the target walk — the map has geometry tokens even when the affordance head
is not confident (target fields zeroed), so map_tokens attach whenever the wrapper exposes
_last_map_tokens. B1KInputs at K=0 ignores the key (measured parity); K=8 ckpts consume it.
"""

import py_compile

P = "/root/bw/BEHAVIOR-1K/OmniGibson/omnigibson/eval/evaluator.py"
s = open(P).read()
if "_last_map_tokens" in s:
    raise SystemExit("already patched")

anchor = "        # --- point conditioning pass-through (patch_point_passthrough.py) -------------\n"
assert anchor in s, "passthrough anchor not found"
block = (
    "        # --- map-token pass-through (patch_map_passthrough2.py) ------------------------\n"
    "        try:\n"
    "            import numpy as _np2\n"
    "            _w2 = self.env\n"
    "            for _ in range(6):\n"
    "                _mt = getattr(_w2, \"_last_map_tokens\", None)\n"
    "                if _mt is not None:\n"
    "                    obs[\"map_tokens\"] = _np2.asarray(_mt, dtype=_np2.float32)\n"
    "                    break\n"
    "                _w2 = getattr(_w2, \"env\", None)\n"
    "                if _w2 is None:\n"
    "                    break\n"
    "        except Exception:\n"
    "            pass\n"
)
s = s.replace(anchor, block + anchor, 1)
compile(s, P, "exec")
open(P, "w").write(s)
py_compile.compile(P, doraise=True)
print("evaluator map-token passthrough patched")

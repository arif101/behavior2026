"""Evaluator passthrough for the v2 keys: attach wrapper._last_v2 (target_points_v2 / stage_v2 / progress) post-flatten,
like patch_point_passthrough.py does for target_points. Idempotent."""
import py_compile
P = "/root/bw/BEHAVIOR-1K/OmniGibson/omnigibson/eval/evaluator.py"
s = open(P).read()
if "_last_v2" in s:
    raise SystemExit("already patched")
anchor = "        # --- point conditioning pass-through (patch_point_passthrough.py) -------------\n"
assert anchor in s, "passthrough anchor not found (apply patch_point_passthrough.py first)"
block = (
    "        # --- v2 label pass-through (patch_v2_passthrough.py) ----------------------------\n"
    "        try:\n"
    "            import numpy as _np3\n"
    "            _w3 = self.env\n"
    "            for _ in range(6):\n"
    "                _v2 = getattr(_w3, \"_last_v2\", None)\n"
    "                if _v2 is not None:\n"
    "                    obs[\"target_points_v2\"] = _np3.asarray(_v2[\"target_points_v2\"], dtype=_np3.float32)\n"
    "                    obs[\"stage_v2\"] = _np3.int32(_v2[\"stage_v2\"])\n"
    "                    obs[\"progress\"] = _np3.float32(_v2[\"progress\"])\n"
    "                    break\n"
    "                _w3 = getattr(_w3, \"env\", None)\n"
    "                if _w3 is None:\n"
    "                    break\n"
    "        except Exception:\n"
    "            pass\n"
)
s = s.replace(anchor, block + anchor, 1)
compile(s, P, "exec"); open(P, "w").write(s); py_compile.compile(P, doraise=True)
print("evaluator v2 passthrough patched")

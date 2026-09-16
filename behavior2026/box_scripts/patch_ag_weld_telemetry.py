"""Add GRASP-COMPLETION telemetry (RUN3_EVAL_PREREG primary metric) to the installed AffordanceMapFullRes: the first step the
eval's assisted-grasp weld fires per arm, read from the sim for logging only; exported as ag_weld_{left,right}_step + grasp
in /root/affordance_wrapper_stats.json. Idempotent; applies on top of patch_wrapper_arms.py."""
import py_compile
P = "/root/behavior2026_eval/affordance_map_fullres.py"
s = open(P).read()
if "ag_weld_" in s:
    raise SystemExit("already patched")
a = '            self._stats["n_steps"] += 1\n'
assert s.count(a) == 1
b = ('            # GRASP-COMPLETION telemetry (patch_ag_weld_telemetry.py): the eval\'s own assisted-grasp weld, LOGGED ONLY\n'
     '            try:\n'
     '                _agp = getattr(self._robot, "_ag_obj_constraint_params", {}) or {}\n'
     '                for _arm in ("left", "right"):\n'
     '                    _c = _agp.get(_arm)\n'
     '                    if _c is not None and self._stats.get(f"ag_weld_{_arm}_step") is None:\n'
     '                        _nm = str(_c.get("ag_obj_prim_path", _c.get("ag_obj", "")))\n'
     '                        self._stats[f"ag_weld_{_arm}_step"] = int(self._stats["n_steps"]); self._stats[f"ag_weld_{_arm}_obj"] = _nm\n'
     '            except Exception:\n'
     '                pass\n' + a)
s = s.replace(a, b, 1)
a2 = '        out = {"n_steps": s["n_steps"], "n_inject": s["n_inject"],\n'
assert s.count(a2) == 1
b2 = ('        out = {"ag_weld_left_step": s.get("ag_weld_left_step"), "ag_weld_right_step": s.get("ag_weld_right_step"),\n'
      '               "ag_weld_obj": s.get("ag_weld_right_obj") or s.get("ag_weld_left_obj"),\n'
      '               "grasp": bool(s.get("ag_weld_left_step") is not None or s.get("ag_weld_right_step") is not None),\n'
      '               "n_steps": s["n_steps"], "n_inject": s["n_inject"],\n')
s = s.replace(a2, b2, 1)
compile(s, P, "exec"); open(P, "w").write(s); py_compile.compile(P, doraise=True)
print("wrapper patched: ag weld telemetry")

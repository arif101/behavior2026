"""v13d (2026-09-15 ~02:15 UTC), applied in place to factory_approach_cap_v13.py. Sweep #6 traces (d100/d110/d120/d130/d230):
the servo declared "reached" on position alone, so STAGE/APPROACH stopped with the wrist 11-22 deg off whenever the
custom ORIENT loop had not finished (it diverges for errors > ~0.15 rad: d50 0.25->0.70, d110 0.16->0.40), and the
last centimetres then twisted the radio (15-101 deg). d140/d40(v12): human lines below ~20 deg elevation graze the body.
Changes: (1) servo(orn_done=...) -> reached needs position AND attitude; position is held while the rotation finishes;
(2) ORIENT = servo position-hold at the landing point with w_orn=1 (the working implementation) instead of the custom loop;
(3) RESTAGE removed (redundant after ORIENT-FIRST; diverged 12 cm on d100); (4) corridor elevation clamped >= 35 deg;
(5) open-air phases (ORIENT/STAGE/APPROACH) return immediately once the radio has moved > 4 cm (relaxed bar blown);
(6) STAGE w_orn 0.3 -> 0.6; APPROACH fallback to BASE_APPROACH only when the POSITION was not reached.
  python make_v13d.py factory_approach_cap_v13.py
"""
import pathlib, sys
p = pathlib.Path(sys.argv[1]); s = p.read_text()
assert "v13d" not in s
R = []
# (1) servo signature + reached rule
R.append(('    def servo(target_p, target_R, budget, tag, stride=0.012, w_orn=0.3, done=0.015):\n        """6-DOF DLS servo of the right eef; orientation held to target_R."""\n        best_d = 9.9\n        for it in range(budget):\n            pE, RE = poseR()\n            d = float(np.linalg.norm(target_p - pE)); best_d = min(best_d, d)\n            if d < done:\n                oerr = float(np.linalg.norm((target_R * RE.inv()).as_rotvec()))\n                print(f"{tag} reached it={it} d={d:.4f} orn_err={oerr:.3f} rad", flush=True)\n                return True, best_d\n',
          '    def servo(target_p, target_R, budget, tag, stride=0.012, w_orn=0.3, done=0.015, orn_done=None):\n        """6-DOF DLS servo of the right eef; orientation held to target_R. v13d: with orn_done set, "reached" needs\n        position AND attitude (position is held while the rotation finishes); servo.last_pos_ok / last_oerr exposed."""\n        best_d = 9.9; servo.last_pos_ok = False; servo.last_oerr = 9.9\n        for it in range(budget):\n            pE, RE = poseR()\n            d = float(np.linalg.norm(target_p - pE)); best_d = min(best_d, d)\n            oerr = float(np.linalg.norm((target_R * RE.inv()).as_rotvec())); servo.last_oerr = oerr\n            if d < done:\n                servo.last_pos_ok = True\n                if orn_done is None or oerr < orn_done:\n                    print(f"{tag} reached it={it} d={d:.4f} orn_err={oerr:.3f} rad", flush=True)\n                    return True, best_d\n'))
# (5) open-air abort once the relaxed bar is blown
R.append(('                if tag in ("STAGE", "APPROACH", "RESTAGE"):\n                    # open-air phases only: the PUSH/ALIGN touch is deliberate\n                    HON["pre_disp"] = max(HON["pre_disp"], _d)\n                    HON["pre_rot"] = max(HON["pre_rot"], float((radio_pose()[1] * HON["R_rest"].inv()).magnitude() * 180 / np.pi))\n',
          '                if tag in ("STAGE", "APPROACH", "RESTAGE", "ORIENT"):\n                    # open-air phases only: the PUSH/ALIGN touch is deliberate\n                    HON["pre_disp"] = max(HON["pre_disp"], _d)\n                    HON["pre_rot"] = max(HON["pre_rot"], float((radio_pose()[1] * HON["R_rest"].inv()).magnitude() * 180 / np.pi))\n                    if HON["pre_disp"] > 0.04 or HON["pre_rot"] > 15.0:   # v13d: relaxed bar blown -> stop flailing\n                        print(f"{tag} RADIO_TOUCHED it={it} disp={HON[\'pre_disp\']:.3f} rot={HON[\'pre_rot\']:.1f} -- stopping", flush=True)\n                        return False, best_d\n'))
R.append(('            if tag in ("STAGE", "RESTAGE", "APPROACH", "PUSH", "ALIGN"):   # RESTAGE tracked too (was missed pre-09-14)\n',
          '            if tag in ("STAGE", "RESTAGE", "APPROACH", "PUSH", "ALIGN", "ORIENT"):   # v13d: ORIENT is a servo phase now\n'))
# (2)+(3) ORIENT via servo, no RESTAGE
i1 = s.index('    orient_anchor = poseR()[0].copy()  # hold HERE while orienting')
i2 = s.index('    grab("orient_end")\n')
orient_new = ('    orient_anchor = poseR()[0].copy()\n'
              '    # v13d: ORIENT = the working DLS servo holding the landing point with full orientation weight; reached needs both.\n'
              '    ok_o, _ = servo(orient_anchor, tgt_R, 150, "ORIENT", stride=0.004, w_orn=1.0, done=0.02, orn_done=0.10)\n'
              '    print(f"ORIENT ok={ok_o} orn_err={servo.last_oerr:.3f} pos_ok={servo.last_pos_ok}", flush=True)\n')
s = s[:i1] + orient_new + s[i2:]
R.append(('    ok_s, best_s = servo(stage_p, tgt_R, 260, "STAGE")\n    print(f"STAGE ok={ok_s} best={best_s:.3f}", flush=True)\n    ok_s2, best_s2 = servo(stage_p, tgt_R, 30, "RESTAGE")   # short re-center (attitude already set at the landing point)\n    ok_a, best_a = servo(tgt_p, tgt_R, 120, "APPROACH", stride=0.008, w_orn=0.6, done=0.025)\n    if not ok_a and best_a < 0.10:\n',
          '    ok_s, best_s = servo(stage_p, tgt_R, 260, "STAGE", w_orn=0.6, orn_done=0.10)\n    print(f"STAGE ok={ok_s} best={best_s:.3f} orn_err={servo.last_oerr:.3f}", flush=True)\n    ok_a, best_a = servo(tgt_p, tgt_R, 160, "APPROACH", stride=0.008, w_orn=0.6, done=0.025, orn_done=0.10)\n    print(f"APPROACH ok={ok_a} best={best_a:.3f} orn_err={servo.last_oerr:.3f} pos_ok={servo.last_pos_ok}", flush=True)\n    if not ok_a and not servo.last_pos_ok and best_a < 0.10 and HON["pre_disp"] <= 0.04:\n'))
R.append(('        ok_a, best_a = servo(tgt_p, tgt_R, 80, "APPROACH", stride=0.006, w_orn=0.6, done=0.025)\n',
          '        ok_a, best_a = servo(tgt_p, tgt_R, 80, "APPROACH", stride=0.006, w_orn=0.6, done=0.025, orn_done=0.10)\n'))
# (4) corridor elevation clamp
R.append(("    stage_dir_w = _corr / _cn\n",
          "    stage_dir_w = _corr / _cn\n"
          "    _el = np.degrees(np.arcsin(np.clip(stage_dir_w[2], -1, 1)))\n"
          "    if _el < 35.0:   # v13d: shallow human lines (d140 17 deg, d40 12 deg) graze the body -> lift the corridor to 35 deg\n"
          "        _xy = stage_dir_w[:2] / (np.linalg.norm(stage_dir_w[:2]) + 1e-9)\n"
          "        stage_dir_w = np.array([_xy[0] * np.cos(np.radians(35.0)), _xy[1] * np.cos(np.radians(35.0)), np.sin(np.radians(35.0))])\n"
          "        print(f\"CORRIDOR lifted from {_el:.1f} to 35.0 deg elevation: {np.round(stage_dir_w, 3).tolist()}\", flush=True)\n"))
R.append(('kind="approach_v13c_owngrasp"', 'kind="approach_v13d_owngrasp"'))
R.append(('"episode": "approach_v13c_owngrasp"', '"episode": "approach_v13d_owngrasp"'))
for a, b in R:
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
compile(s, str(p), "exec"); p.write_text(s); print("v13d applied")

"""Post-process factory_approach_cap_v11.py: ORIENT-FIRST (2026-09-14).
Rotate the wrist at the 20 cm base-landing point where the hand is clear, THEN stage in to 10 cm and
approach. Films (d110): orienting at the 10 cm staging point swept the handle with the gripper and flipped
the radio before the approach started, and that phase was not honesty-tracked. Now: orient first, radio
watched during the rotation (abort at 1.2 cm), short re-stage afterwards.
  python make_v11_orient.py factory_approach_cap_v11.py
"""
import pathlib, sys

p = pathlib.Path(sys.argv[1]); src = p.read_text()
assert "ORIENT-FIRST" not in src, "already applied"
stage_line = '    ok_s, best_s = servo(stage_p, tgt_R, 260, "STAGE")\n    print(f"STAGE ok={ok_s} best={best_s:.3f}", flush=True)\n'
orient_start = '    orient_anchor = poseR()[0].copy()  # hold HERE while orienting (d50: stage never converged, drift check vs stage_p aborted at it=0)\n'
orient_end = '    grab("orient_end")\n'
i0 = src.index(stage_line); i1 = src.index(orient_start); i2 = src.index(orient_end) + len(orient_end)
assert i0 < i1 < i2
orient_block = src[i1:i2]
track = (
    "        step_cmd(cmd)\n"
    "        HON[\"n\"] += 1\n"
    "        _d = float(np.linalg.norm(radio_pose()[0] - HON[\"radio_rest\"]))\n"
    "        HON[\"max_disp\"] = max(HON[\"max_disp\"], _d); HON[\"pre_disp\"] = max(HON[\"pre_disp\"], _d)\n"
    "        HON[\"pre_rot\"] = max(HON[\"pre_rot\"], float((radio_pose()[1] * HON[\"R_rest\"].inv()).magnitude() * 180 / np.pi))\n"
    "        if HON[\"first_contact\"] is None and contact():\n"
    "            HON[\"first_contact\"] = f\"ORIENT:{it}\"\n"
    "        if _d > 0.012:\n"
    "            print(f\"ORIENT RADIO_TOUCHED it={it} disp={_d:.3f} -- stopping\", flush=True)\n"
    "            break\n"
)
assert orient_block.count("        step_cmd(cmd)\n") == 1
orient_block = orient_block.replace("        step_cmd(cmd)\n", track, 1)
header = "    # ORIENT-FIRST (2026-09-14): attitude is set here at the 20 cm landing point (hand clear), then STAGE in.\n"
src = src[:i0] + header + orient_block + stage_line + src[i2:]
old_restage = '    ok_s2, best_s2 = servo(stage_p, tgt_R, 60, "RESTAGE")   # re-center after orienting\n'
assert src.count(old_restage) == 1
src = src.replace(old_restage, '    ok_s2, best_s2 = servo(stage_p, tgt_R, 30, "RESTAGE")   # short re-center (attitude already set at the landing point)\n')
compile(src, str(p), "exec")
p.write_text(src)
print("ORIENT-FIRST applied:", p)

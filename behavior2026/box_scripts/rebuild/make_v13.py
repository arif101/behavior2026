"""factory_approach_cap_v13.py = v12 + the demo's OWN grasp geometry (2026-09-14 22:15 UTC).
v9 introduced the d20 canonical rail grasp because, under the template-pose restore bug, each demo's own
post-closure geometry looked like a non-rest grasp (the sim radio had been kicked onto its side). With the
restore fixed, the demo's own grasp at closure+t0off IS a rest-state grasp (radio moved <5 mm / <3 deg), it
is reachable from that demo's pre-grasp posture by construction, and its approach attitude is what the
human actually used. Sweep #5 (v12, canonical grasp): d50 ORIENT abort with 62 deg left (canonical attitude
unreachable on a radio yawed 135 deg from d20's); d40 palm shoved the body 9 cm along a 12-deg approach.
--grasp canon keeps the old behaviour.
  python make_v13.py factory_approach_cap_v12.py factory_approach_cap_v13.py
"""
import pathlib, sys
src = pathlib.Path(sys.argv[1]).read_text(); out = pathlib.Path(sys.argv[2])
old = ('    canon = json.load(open("/root/canonical_grasp_d20.json"))\n'
       '    rel_p = np.array(canon["rel_p"]); rel_R = R.from_quat(canon["rel_R_quat"])\n'
       '    rel_f = [np.array(f) for f in canon["rel_f"]]; rel_fm = np.array(canon["rel_fm"]); gt_gap = float(canon["gap"])\n'
       '    tine_axis = np.array(canon["tine_axis_radio"])\n')
new = ('    if a.grasp == "canon":\n'
       '        canon = json.load(open("/root/canonical_grasp_d20.json"))\n'
       '        rel_p = np.array(canon["rel_p"]); rel_R = R.from_quat(canon["rel_R_quat"])\n'
       '        rel_f = [np.array(f) for f in canon["rel_f"]]; rel_fm = np.array(canon["rel_fm"]); gt_gap = float(canon["gap"])\n'
       '        tine_axis = np.array(canon["tine_axis_radio"])\n'
       '    else:\n'
       '        # v13 default: THIS demo\'s own certified grasp (rel_p / rel_R computed at t_post above)\n'
       '        rel_f = [np.array(f) for f in rel_f_demo]; rel_fm = np.mean(np.stack(rel_f), axis=0)\n'
       '        gt_gap = float(np.linalg.norm(rel_f[0] - rel_f[1]))\n'
       '        _ta = rel_fm - rel_p; tine_axis = _ta / (np.linalg.norm(_ta) + 1e-9)\n'
       '        print(f"OWN grasp (this demo): rel_p {np.round(rel_p, 3).tolist()} gap {gt_gap:.3f} tine_axis(radio) {np.round(tine_axis, 3).tolist()}", flush=True)\n')
assert src.count(old) == 1; src = src.replace(old, new)
old_arg = '    ap.add_argument("--K", type=int, default=30, help="restore at closure-K")\n'
assert src.count(old_arg) == 1
src = src.replace(old_arg, old_arg + '    ap.add_argument("--grasp", choices=("own", "canon"), default="own", help="v13: the demo\'s own grasp (default) or the d20 canonical")\n')
src = src.replace('kind="approach_v12_humanline"', 'kind="approach_v13_owngrasp"').replace('"episode": "approach_v12_humanline"', '"episode": "approach_v13_owngrasp"')
compile(src, str(out), "exec"); out.write_text(src); print("v13 written:", out)

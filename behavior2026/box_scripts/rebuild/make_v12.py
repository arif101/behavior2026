"""factory_approach_cap_v12.py = v11 (ORIENT-FIRST) + two changes forced by the fixed restore (2026-09-14):
(a) the approach corridor is the HUMAN'S OWN approach line (grasp pose -> hand at the restored pre-grasp
    frame), not "-tine_axis" from the canonical grasp: that body-frame axis points UP in world on an upright
    radio (0.42, 0.12, 0.90), so the corridor pointed INTO the radio body; d20 STAGE hit it at step 4 and
    flung it 46 cm. It only ever worked because the old restore had the radio lying on its side.
(b) fail fast: after APPROACH, if the pre-contact displacement already exceeds the relaxed 4 cm bar, stop
    (RESULT RADIO_TOUCHED) instead of flailing through PUSH/ALIGN/weld for 15 minutes.
  python make_v12.py factory_approach_cap_v11.py factory_approach_cap_v12.py
"""
import pathlib, sys
src = pathlib.Path(sys.argv[1]).read_text(); out = pathlib.Path(sys.argv[2])
old = '    stage_dir_w = -RRest.apply(tine_axis)  # back off along the tines: the approach corridor\n    print(f"CANON grasp applied at rest attitude; stage corridor dir {np.round(stage_dir_w, 3).tolist()}", flush=True)\n    pE0, RE0 = poseR()\n'
new = ('    pE0, RE0 = poseR()\n'
       '    # v12: corridor = the human\'s own approach line at this pre-grasp frame (collision-free by construction).\n'
       '    # v11 used -tine_axis (body frame): on an UPRIGHT radio that points down into the body (d20: STAGE hit at it=4).\n'
       '    _corr = pE0 - tgt_p; _cn = float(np.linalg.norm(_corr))\n'
       '    if _cn < 0.05:\n'
       '        _corr = np.array([0.0, 0.0, 1.0]); _cn = 1.0   # degenerate: hand already on the grasp pose -> come from above\n'
       '    stage_dir_w = _corr / _cn\n'
       '    print(f"CANON grasp applied at rest attitude; corridor = human approach line dir {np.round(stage_dir_w, 3).tolist()} '
       '(hand->grasp {_cn:.3f} m; -tine_axis would be {np.round(-RRest.apply(tine_axis), 3).tolist()})", flush=True)\n')
assert src.count(old) == 1; src = src.replace(old, new)
old2 = '    grab("approach_end")\n    if not ok_a:\n'
new2 = ('    grab("approach_end")\n'
        '    if HON["pre_disp"] > 0.04 or HON["pre_rot"] > 15.0:\n'
        '        # v12 fail-fast: the relaxed honesty bar is already blown before contact -> no point pushing/aligning/welding\n'
        '        json.dump(dict(demo=a.demo, t0=t_pre, closure=closure, K=a.K, gap0=round(gap0, 4), pull=np.round(pull, 4).tolist(),\n'
        '                       base=BASE, ok=False, honest=False, honest_strict=False, skip="RADIO_TOUCHED",\n'
        '                       radio_disp_precontact=round(HON["pre_disp"], 4), radio_rot_precontact=round(HON["pre_rot"], 2),\n'
        '                       first_contact=HON["first_contact"]), open(f"{OUT}/d{a.demo:03d}_meta.json", "w"), indent=1)\n'
        '        print(f"RESULT d{a.demo} RADIO_TOUCHED pre_disp={HON[\'pre_disp\']:.3f} pre_rot={HON[\'pre_rot\']:.1f} first_contact={HON[\'first_contact\']}", flush=True)\n'
        '        os._exit(0)\n'
        '    if not ok_a:\n')
assert src.count(old2) == 1; src = src.replace(old2, new2)
src = src.replace('kind="approach_v11_basedrive"', 'kind="approach_v12_humanline"')
src = src.replace('"episode": "approach_v11_basedrive"', '"episode": "approach_v12_humanline"')
compile(src, str(out), "exec"); out.write_text(src); print("v12 written:", out)

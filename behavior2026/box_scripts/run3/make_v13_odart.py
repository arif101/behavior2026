"""Generate factory_approach_cap_v13_odart.py from factory_approach_cap_v13_dart.py: OBJECT-perturbation DART
(counterfactual pairs, 2026-09-26). Adds --perturb-object "dlat,ddepth,yaw_deg,tag": at the point where the hand DART
perturbation would run (robot already driven/staged exactly as in the unperturbed clip, capture still OFF), teleport the
RADIO by dlat along the corridor normal + ddepth along the corridor (horizontal, metres) and yaw it about z, settle 30
steps, verify it rested (< 2 cm, < 5 deg), then recompute the grasp goal from the NEW rest pose; the unchanged pipeline
(ORIENT/STAGE/APPROACH/PUSH/closure/carry) records the correction. Same robot start state as the unperturbed clip,
different picture, different actions -> a training pair the flow loss cannot fit from proprioception alone.
Outputs: /root/factory_clips_odart, /root/factory_obs_odart (rac_<demo>_<tag>_200.npz), film /root/rt_odart_film.
  python make_v13_odart.py factory_approach_cap_v13_dart.py factory_approach_cap_v13_odart.py
"""
import pathlib, sys
src = pathlib.Path(sys.argv[1]).read_text(); out = pathlib.Path(sys.argv[2])
assert "--perturb-object" not in src and "--perturb" in src
def rep(old, new, count=1):
    global src
    assert src.count(old) == count, (old[:60], src.count(old))
    src = src.replace(old, new)
rep('    ap.add_argument("--perturb", type=str, default="", help="DART: dy,dz,yaw_deg,tag (m, m, deg) applied before recording")\n',
    '    ap.add_argument("--perturb", type=str, default="", help="DART: dy,dz,yaw_deg,tag (m, m, deg) applied before recording")\n'
    '    ap.add_argument("--perturb-object", type=str, default="", help="ODART: dlat,ddepth,yaw_deg,tag -> teleport the RADIO before recording (counterfactual pair)")\n')
rep('    PTAG = ("_" + a.perturb.split(",")[3]) if a.perturb else ""\n',
    '    PTAG = ("_" + a.perturb.split(",")[3]) if a.perturb else (("_" + a.perturb_object.split(",")[3]) if a.perturb_object else "")\n')
rep('OUT = "/root/factory_clips_dart"', 'OUT = "/root/factory_clips_odart"')
src = src.replace("/root/factory_obs_dart", "/root/factory_obs_odart").replace("/root/rt_dart_film", "/root/rt_odart_film")
assert "/root/factory_obs_odart" in src
block = '''    if a.perturb_object:
        # ODART: move the RADIO, not the hand. The robot is in the unperturbed clip's start state; only the scene differs.
        _dl, _dd, _oyaw = [float(v) for v in a.perturb_object.split(",")[:3]]
        _lat = np.array([-stage_dir_w[1], stage_dir_w[0], 0.0]); _lat = _lat / (np.linalg.norm(_lat) + 1e-9)
        _dep = np.array([stage_dir_w[0], stage_dir_w[1], 0.0]); _dep = _dep / (np.linalg.norm(_dep) + 1e-9)
        _p0, _R0 = radio_pose()
        _pn = _p0 + _dl * _lat + _dd * _dep
        _Rn = R.from_euler("z", _oyaw, degrees=True) * _R0
        radio.set_position_orientation(th.as_tensor(_pn, dtype=th.float32), th.as_tensor(_Rn.as_quat(), dtype=th.float32))
        for _ in range(30):
            wrapper.env.step(hold); cmds_log.append(hold.copy())
        _ps, _Rs = radio_pose()
        _sd = float(np.linalg.norm(_ps - _pn)); _sr = float((_Rs * _Rn.inv()).magnitude() * 180 / np.pi)
        print(f"OBJPERTURB dlat={_dl:+.3f} ddep={_dd:+.3f} yaw={_oyaw:+.1f} settled_d={_sd:.3f} rot={_sr:.1f} "
              f"hand->new-grasp {float(np.linalg.norm(_ps + _Rs.apply(rel_p) - poseR()[0])):.3f} m", flush=True)
        if _sd > 0.02 or _sr > 5.0:
            print(f"RESULT d{a.demo} SKIP_OBJPERTURB_UNSETTLED", flush=True); os._exit(0)
        pRest, RRest = radio_pose()
        tgt_p = pRest + RRest.apply(rel_p); tgt_R = RRest * rel_R
        gap0 = float(np.linalg.norm(tgt_p - poseR()[0]))
'''
rep('    grab("pre")\n    CAP[0] = True\n', block + '    grab("pre")\n    CAP[0] = True\n')
out.write_text(src); print("wrote", out, "with --perturb-object")

"""Generate factory_approach_cap_v13_dart.py from factory_approach_cap_v13.py (DART corrective clips, ARCH_4D_ATTENTION_SPEC B1).
Adds --perturb "dy,dz,yaw_deg,tag": after the pre-pull restore and BEFORE recording starts, servo the right hand to its
restored pose + (dy along the lateral corridor normal, dz vertical) with the wrist yawed by yaw_deg (tag "PERTURB", not
honesty-tracked), then run the unchanged v13d pipeline (ORIENT/STAGE/APPROACH/PUSH/closure/carry), which records the
CORRECTION from the perturbed start. Outputs get a per-perturbation suffix so clips coexist: d<demo>_<tag>_*.
  python make_v13_dart.py factory_approach_cap_v13.py factory_approach_cap_v13_dart.py
"""
import pathlib, sys
src = pathlib.Path(sys.argv[1]).read_text(); out = pathlib.Path(sys.argv[2])
assert "--perturb" not in src
# 1. argparse option
old = '    ap.add_argument("--settle", type=int, default=40)\n'
assert src.count(old) == 1
src = src.replace(old, old + '    ap.add_argument("--perturb", type=str, default="", help="DART: dy,dz,yaw_deg,tag (m, m, deg) applied before recording")\n')
# 2. output naming: per-perturbation suffix
src = src.replace('OUT = "/root/factory_clips_approach"', 'OUT = "/root/factory_clips_dart"')
for pat in ('f"{OUT}/d{a.demo:03d}_meta.json"', 'f"{OUT}/d{a.demo:03d}_approach.npz"'):
    assert pat in src, pat
src = src.replace('f"{OUT}/d{a.demo:03d}_meta.json"', 'f"{OUT}/d{a.demo:03d}{PTAG}_meta.json"')
src = src.replace('f"{OUT}/d{a.demo:03d}_approach.npz"', 'f"{OUT}/d{a.demo:03d}{PTAG}_approach.npz"')
src = src.replace('f"/root/factory_obs2/rac_{a.demo}_200.npz"', 'f"/root/factory_obs_dart/rac_{a.demo}{PTAG}_200.npz"')
src = src.replace('print(f"OBS_SAVED rac_{a.demo}_200.npz', 'print(f"OBS_SAVED rac_{a.demo}{PTAG}_200.npz')
src = src.replace('FILM = f"/root/rt_approach_film/d{a.demo:03d}"', 'FILM = f"/root/rt_dart_film/d{a.demo:03d}{PTAG}"')
# 3. PTAG definition right after args parse
old = '    a = ap.parse_args()\n'
assert src.count(old) == 1
src = src.replace(old, old + '    PTAG = ("_" + a.perturb.split(",")[3]) if a.perturb else ""\n    os.makedirs("/root/factory_obs_dart", exist_ok=True)\n')
# 4. the perturbation stage: before grab("pre") / CAP[0] = True
old = '    grab("pre")\n    CAP[0] = True\n'
assert src.count(old) == 1
new = '''    if a.perturb:
        # DART perturbation (not recorded, not honesty-tracked): displace the restored hand pose and yaw the wrist.
        _dy, _dz, _yaw = [float(v) for v in a.perturb.split(",")[:3]]
        _pE, _RE = poseR()
        _lat = np.array([-stage_dir_w[1], stage_dir_w[0], 0.0]); _lat = _lat / (np.linalg.norm(_lat) + 1e-9)
        _goal_p = _pE + _dy * _lat + np.array([0.0, 0.0, _dz])
        _goal_R = R.from_euler("z", _yaw, degrees=True) * _RE
        _okp, _dp = servo(_goal_p, _goal_R, 120, "PERTURB", stride=0.006, w_orn=1.0, done=0.012, orn_done=0.08)
        print(f"PERTURB dy={_dy:+.3f} dz={_dz:+.3f} yaw={_yaw:+.1f} ok={_okp} d={_dp:.3f} orn_err={servo.last_oerr:.3f}", flush=True)
        if float(np.linalg.norm(radio_pose()[0] - pRest)) > 0.02:
            print(f"RESULT d{a.demo} SKIP_PERTURB_TOUCHED", flush=True); os._exit(0)
    grab("pre")
    CAP[0] = True
'''
src = src.replace(old, new)
compile(src, str(out), "exec"); out.write_text(src); print("written", out, "lines", src.count("\\n"))

"""Derive the ODART (object-perturbation DART) round/convert/loop scripts from the DART ones on the sim box.
  odart_convert.sh : renders /root/factory_obs_odart -> LeRobot root /root/b1k_radio_odart_r<TAG> -> HF b1k_radio_odart_r<TAG>
                     (tag codes 11-18 for the 8 object perturbations); clip cmds/meta -> HF factory_clips_odart/
  odart_round.sh   : convert+push pending renders (unique RTAG per conversion), then attempts over DEMOS x OPERTS with the
                     odart factory, /root/odart_logs, /root/odart_skip_demos.txt, 1800 s timeout, auto-skip after 2 failures
  odart_loop.sh    : chains rounds (START..START+5) until a round has no attempts left
Run on the sim box: python3 make_odart_scripts.py"""
import pathlib, re
conv = pathlib.Path("/root/dart_convert.sh").read_text()
conv = conv.replace("/root/factory_obs_dart", "/root/factory_obs_odart").replace("/root/b1k_radio_dart_r", "/root/b1k_radio_odart_r")
conv = conv.replace('"b1k_radio_dart_r"', '"b1k_radio_odart_r"').replace("/root/factory_clips_dart", "/root/factory_clips_odart").replace('path_in_repo="factory_clips_dart"', 'path_in_repo="factory_clips_odart"')
old = 'CODES = {"y5": 1, "ym5": 2, "z5": 3, "zm4": 4, "yaw20": 5, "yawm20": 6, "mix1": 7, "mix2": 8}'
assert conv.count(old) == 1
conv = conv.replace(old, 'CODES = {"ol5": 11, "olm5": 12, "od5": 13, "odm5": 14, "oy15": 15, "oym15": 16, "omix1": 17, "omix2": 18}')
conv = conv.replace('commit_message="b1k_radio_approach_v2: v11 ORIENT-FIRST base-drive approach clips (joint 4 locked, 10-DOF)"', 'commit_message="ODART: object-perturbation counterfactual clips"')
conv = conv.replace('commit_message="DART r1 clip cmds + honesty meta"', 'commit_message="ODART clip cmds + honesty meta"')
assert conv.count("odart") >= 6, conv.count("odart")
pathlib.Path("/root/odart_convert.sh").write_text(conv)
rnd = pathlib.Path("/root/dart_round.sh").read_text()
assert "RTAG=" in rnd and "dart_skip_demos" in rnd, "dart_round.sh must carry the 09-26 patches"
rnd = rnd.replace("/root/dart_convert.sh", "/root/odart_convert.sh").replace("/root/dart_convert_r", "/root/odart_convert_r")
rnd = rnd.replace("/root/factory_obs_dart", "/root/factory_obs_odart").replace("/root/factory_clips_dart", "/root/factory_clips_odart")
rnd = rnd.replace("/root/dart_logs", "/root/odart_logs").replace("/root/dart_skip_demos.txt", "/root/odart_skip_demos.txt").replace("/root/dart_fail_demos.txt", "/root/odart_fail_demos.txt")
old = 'PERTS="0.05,0,0,y5 -0.05,0,0,ym5 0,0.05,0,z5 0,-0.04,0,zm4 0,0,20,yaw20 0,0,-20,yawm20 0.04,0.03,12,mix1 -0.04,0.03,-12,mix2"'
assert rnd.count(old) == 1
rnd = rnd.replace(old, 'PERTS="0.05,0,0,ol5 -0.05,0,0,olm5 0,0.05,0,od5 0,-0.05,0,odm5 0,0,15,oy15 0,0,-15,oym15 0.04,0.03,10,omix1 -0.04,-0.03,-10,omix2"   # ODART: dlat,ddepth,yaw_deg,tag (RADIO moved, hand untouched)')
old = '/root/factory_approach_cap_v13_dart.py --demo $d --perturb="$pert"'
assert rnd.count(old) == 1
rnd = rnd.replace(old, '/root/factory_approach_cap_v13_odart.py --demo $d --perturb-object="$pert"')
rnd = rnd.replace('grep -qE "RESULT d|OBS_SAVED|PERTURB dy"', 'grep -qE "RESULT d|OBS_SAVED|OBJPERTURB dlat"')
rnd = rnd.replace('[dart_round ', '[odart_round ').replace("ROUND_$", "OROUND_$") if "ROUND_$" in rnd else rnd.replace('[dart_round ', '[odart_round ')
pathlib.Path("/root/odart_round.sh").write_text(rnd)
loop = pathlib.Path("/root/dart_loop.sh").read_text()
loop = loop.replace("/root/dart_round.sh", "/root/odart_round.sh").replace("/root/dart_round$r.out", "/root/odart_round$r.out").replace("/root/factory_clips_dart", "/root/factory_clips_odart").replace("[dart_loop", "[odart_loop").replace("DART_LOOP_DONE", "ODART_LOOP_DONE")
pathlib.Path("/root/odart_loop.sh").write_text(loop)
print("wrote /root/odart_convert.sh /root/odart_round.sh /root/odart_loop.sh")

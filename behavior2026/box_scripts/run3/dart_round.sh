#!/bin/bash
# DART round driver (disk-safe): convert+push what is rendered, delete renders, then run the next batch of perturbations.
# Usage: ROUND=1 setsid nohup bash /root/dart_round.sh > /root/dart_round.out 2>&1 &
set -u; say(){ echo "[dart_round $(date -u +%m-%dT%H:%M:%S)] $*"; }
PY=/root/miniconda3/envs/behavior/bin/python; ROUND=${ROUND:-1}
# ---- 0. convert + push the current renders, then delete them --------------------------------------------------
if ls /root/factory_obs_dart/rac_*_200.npz >/dev/null 2>&1; then
  BAR=strict ROUND=$ROUND bash /root/dart_convert.sh > /root/dart_convert_r$ROUND.out 2>&1
  grep -E "SELECTED|EPISODES|HF_PUSH_OK|DART_CONVERT_DONE|Traceback|Error" /root/dart_convert_r$ROUND.out | tail -5
  if grep -q HF_PUSH_OK /root/dart_convert_r$ROUND.out; then rm -f /root/factory_obs_dart/rac_*_200.npz; say "round $ROUND converted+pushed, renders deleted"; else say "CONVERT/PUSH FAILED round $ROUND (renders kept)"; exit 1; fi
fi
# ---- 1. FK precompute (fixed: camera sensors instantiated) ---------------------------------------------------------
if [ ! -f /root/fk/cam_pose.npy ]; then
  say "FK precompute"; OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 OMNIGIBSON_HEADLESS=1 XDG_RUNTIME_DIR=/tmp/xdg CUDA_VISIBLE_DEVICES=0 \
    timeout 14400 $PY -u /root/fk_cam_poses.py 10 > /root/fk/fk.log 2>&1; say "FK: $(grep -E "FK_DONE|Error" /root/fk/fk.log | tail -1 | cut -c1-100)"
fi
# ---- 2. next batch: perturbations not yet attempted, capped by disk (stop launching below 25 GB free) ---------------
mkdir -p /root/dart_logs
DEMOS=$(cat /root/factory_obs2_files.txt 2>/dev/null | sed -E "s/rac_([0-9]+)_200.npz/\1/" | sort -n); [ -z "$DEMOS" ] && DEMOS="10 160 170 180 190 20 210 250 260 270 310 320 330 340 370 390 40 50 60 70 80"
PERTS="0.05,0,0,y5 -0.05,0,0,ym5 0,0.05,0,z5 0,-0.04,0,zm4 0,0,20,yaw20 0,0,-20,yawm20 0.04,0.03,12,mix1 -0.04,0.03,-12,mix2"
n=0
for d in $DEMOS; do for pert in $PERTS; do tag=${pert##*,}
  [ -f /root/factory_clips_dart/d$(printf %03d $d)_${tag}_meta.json ] && continue
  [ -s /root/dart_logs/d${d}_${tag}.log ] && grep -qE "RESULT d|OBS_SAVED" /root/dart_logs/d${d}_${tag}.log && continue
  free=$(df -BG / | tail -1 | awk "{print \$4}" | tr -d G); if [ "$free" -lt 25 ]; then say "disk low ($free G) -> stop batch"; break 2; fi
  find /tmp -mindepth 1 -maxdepth 1 -type d -name "tmp????????" -mmin +120 -exec rm -rf {} + 2>/dev/null
  OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 OMNIGIBSON_HEADLESS=1 XDG_RUNTIME_DIR=/tmp/xdg CUDA_VISIBLE_DEVICES=0 \
    timeout 2400 $PY -u /root/factory_approach_cap_v13_dart.py --demo $d --perturb "$pert" > /root/dart_logs/d${d}_${tag}.log 2>&1
  rm -f /root/fap_tmp_$d.hdf5; n=$((n+1))
  echo "d$d $tag $(date -u +%H:%M) $(grep -oE "RESULT d[0-9]+ [A-Z_]+|OBS_SAVED rac_[0-9a-z_]+_200.npz \([0-9]+ steps\)" /root/dart_logs/d${d}_${tag}.log | tail -1)"
  [ $n -ge 14 ] && { say "batch of 14 done -> convert"; break 2; }
done; done
say "ROUND_${ROUND}_DONE renders=$(ls /root/factory_obs_dart/*.npz 2>/dev/null | wc -l)"

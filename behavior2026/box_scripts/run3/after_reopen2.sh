#!/bin/bash
# After the reopen eval (2026-09-22): (1) FK camera-pose precompute for the training mix; (2) DART corrective clips
# through the v13d factory with hand/wrist perturbations, 2 parallel sims. Usage: setsid nohup bash /root/after_reopen.sh > /root/after_reopen.out 2>&1 &
set -u; say(){ echo "[after_reopen $(date -u +%m-%dT%H:%M:%S)] $*"; }
PY=/root/miniconda3/envs/behavior/bin/python; PYO=/root/openpi_fork/.venv/bin/python
until grep -q REOPEN_EVAL_DONE /root/reopen_eval.out 2>/dev/null; do sleep 120; done
say "reopen2 done -> SYSTEM-2 STAGE eval (model stage head + 2-of-3 voting drives stage_v2/progress/target_points_v2), n=25"
rm -f /root/stage_head_log.jsonl
POLICY_CONFIG=pi05_radio_full WRAP=behavior2026_eval.gripper_reopen.GripperReopenWrapper SERVE_FORWARD_MAP_TOKENS=1 HISTORY_MODE=normal SERVE_STAGE_SOURCE=head TAG=_s2stage bash /root/run3_eval_arm.sh full 25 > /root/run3_eval_full_s2stage_driver.out 2>&1
say "S2STAGE_EVAL_DONE grasp $(grep -c "arm=full_s2stage .*grasp=True" /root/run3_eval_full.log) / $(grep -c "arm=full_s2stage " /root/run3_eval_full.log) success $(grep -c "arm=full_s2stage .*success=True" /root/run3_eval_full.log)"
say "reopen eval done: $(tail -1 /root/reopen_eval.out | cut -c1-80)"
for P in $(pgrep -f "serve_b1k.py.*--port 8000"); do kill $P; done; sleep 3
# ---- 1. FK precompute ----------------------------------------------------------------------------------------
mkdir -p /root/fk; $PYO /root/dump_proprio.py 2>&1 | tail -1
OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 OMNIGIBSON_HEADLESS=1 XDG_RUNTIME_DIR=/tmp/xdg CUDA_VISIBLE_DEVICES=0 \
  timeout 14400 $PY -u /root/fk_cam_poses.py 10 > /root/fk/fk.log 2>&1
say "FK: $(grep -E "FK_DONE|Traceback" /root/fk/fk.log | tail -1)"
# ---- 2. DART clips --------------------------------------------------------------------------------------------
mkdir -p /root/factory_clips_dart /root/factory_obs_dart /root/dart_logs
DEMOS=$(ls /root/factory_obs2/rac_*_200.npz | sed -E "s/.*rac_([0-9]+)_200.npz/\1/" | sort -n)
PERTS="0.05,0,0,y5 -0.05,0,0,ym5 0,0.05,0,z5 0,-0.04,0,zm4 0,0,20,yaw20 0,0,-20,yawm20 0.04,0.03,12,mix1 -0.04,0.03,-12,mix2"
run_queue(){ local q=$1; shift
  for job in "$@"; do d=${job%%:*}; pert=${job#*:}; tag=${pert##*,}
    [ -f /root/factory_clips_dart/d$(printf %03d $d)_${tag}_meta.json ] && continue
    find /tmp -mindepth 1 -maxdepth 1 -type d -user root -name "tmp????????" -mmin +120 -exec rm -rf {} + 2>/dev/null
    OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 OMNIGIBSON_HEADLESS=1 XDG_RUNTIME_DIR=/tmp/xdg CUDA_VISIBLE_DEVICES=0 \
      timeout 2400 $PY -u /root/factory_approach_cap_v13_dart.py --demo $d --perturb "$pert" > /root/dart_logs/d${d}_${tag}.log 2>&1
    rm -f /root/fap_tmp_$d.hdf5
    echo "[q$q] d$d $tag rc=$? $(date -u +%H:%M) $(grep -oE "RESULT d[0-9]+ [A-Z_]+|OBS_SAVED rac_[0-9a-z_]+_200.npz \([0-9]+ steps\)|PERTURB [^\n]{0,60}" /root/dart_logs/d${d}_${tag}.log | tail -2 | tr "\n" " ")"
  done; echo "[q$q] QUEUE_DONE"; }
JOBS=(); for d in $DEMOS; do for p in $PERTS; do JOBS+=("$d:$p"); done; done
A=(); B=(); i=0; for j in "${JOBS[@]}"; do if [ $((i % 2)) -eq 0 ]; then A+=("$j"); else B+=("$j"); fi; i=$((i+1)); done
say "DART: ${#JOBS[@]} jobs over $(echo $DEMOS | wc -w) demos x 8 perturbations"
run_queue A "${A[@]}" & sleep 240; run_queue B "${B[@]}" & wait
say "DART_DONE clips=$(ls /root/factory_obs_dart/*.npz 2>/dev/null | wc -l) honest=$(grep -l "\"honest_strict\": true" /root/factory_clips_dart/*_meta.json 2>/dev/null | wc -l)"

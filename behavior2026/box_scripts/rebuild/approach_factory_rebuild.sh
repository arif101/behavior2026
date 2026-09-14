#!/bin/bash
# Approach factory v13 (own grasp, human-line corridor, fixed restore) (10-DOF servo, torso joint 4 locked; BASE drive-up / corridor retreat / corridor advance) over all factory
# demos, two parallel sims. Outputs: /root/factory_obs2/rac_<demo>_200.npz (converter-ready renders,
# 10-DOF) + /root/factory_clips_approach/ cmds+meta. Markers: /root/factory_clips_approach/dDDD.attempted.
# Prereqs: /root/snapshot_bank_v21/manifest_d<demo>.json (G entry), /root/factory_clips/dDDD_meta.json,
# /root/canonical_grasp_d20.json, raw demos, flat repo layout under /root/behavior2026.
# Usage: setsid nohup bash approach_factory_rebuild.sh > /root/approach_rebuild.out 2>&1 &
set -u
PY=/root/miniconda3/envs/behavior/bin/python
mkdir -p /root/factory_obs2 /root/factory_clips_approach /root/approach_logs
DEMOS=$(ls /root/factory_clips/d*_meta.json | sed -E 's/.*d0*([0-9]+)_meta.json/\1/' | sort -n)
run_queue(){
  local q=$1; shift
  for d in "$@"; do
    if [ -f /root/factory_obs2/rac_${d}_200.npz ]; then echo "[q$q] d$d render exists, skip"; continue; fi
    if [ ! -f /root/snapshot_bank_v21/manifest_d$d.json ]; then echo "[q$q] d$d NO MANIFEST, skip"; continue; fi
    echo "[q$q] d$d start $(date -u +%H:%M:%S)"
    find /tmp -mindepth 1 -maxdepth 1 -type d -user root -name 'tmp????????' -mmin +120 -exec rm -rf {} + 2>/dev/null
    OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 OMNIGIBSON_HEADLESS=1 XDG_RUNTIME_DIR=/tmp/xdg CUDA_VISIBLE_DEVICES=0 \
      timeout 9000 $PY -u /root/factory_approach_cap_v13.py --demo $d > /root/approach_logs/d$d.log 2>&1
    rc=$?; rm -f /root/fap_tmp_$d.hdf5
    touch /root/factory_clips_approach/d$(printf %03d $d).attempted
    echo "[q$q] d$d done rc=$rc $(date -u +%H:%M:%S) $(grep -oE 'RESULT d[0-9]+ [A-Z_]+[^,]*|OBS_SAVED rac_[0-9]+_200.npz \([0-9]+ steps\)|APPROACH_HONESTY[^m]*m' /root/approach_logs/d$d.log | tail -2 | tr '\n' ' ')"
  done
  echo "[q$q] QUEUE_DONE"
}
A=(); B=(); i=0
for d in $DEMOS; do if [ $((i % 2)) -eq 0 ]; then A+=($d); else B+=($d); fi; i=$((i+1)); done
echo "queue A: ${A[*]}"; echo "queue B: ${B[*]}"
run_queue A "${A[@]}" &
sleep 300
run_queue B "${B[@]}" &
wait
echo "APPROACH_FACTORY_REBUILD_DONE renders=$(ls /root/factory_obs2/rac_*_200.npz 2>/dev/null | wc -l)"

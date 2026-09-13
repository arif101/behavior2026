#!/bin/bash
# Rebuild /root/snapshot_bank_v21 (manifest_d<demo>.json + state npz) for the factory-clip demos on
# the new sim box (old box gone 2026-09-12). Same call as snapshot_bank_v21_driver.sh, run as two
# parallel queues (one demo per process; Isaac env-reuse leak forbids reuse).
# Usage: setsid nohup bash snapshot_bank_rebuild.sh > /root/snapshot_bank_rebuild.out 2>&1 &
set -u
PY=/root/miniconda3/envs/behavior/bin/python
OUT=/root/snapshot_bank_v21; mkdir -p $OUT /root/bank_logs
DEMOS=$(ls /root/factory_clips/d*_meta.json | sed -E 's/.*d0*([0-9]+)_meta.json/\1/' | sort -n)
run_queue(){
  local q=$1; shift
  for d in "$@"; do
    if [ -f $OUT/manifest_d$d.json ]; then echo "[q$q] d$d manifest exists, skip"; continue; fi
    echo "[q$q] d$d start $(date -u +%H:%M:%S)"
    PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 OMNIGIBSON_HEADLESS=1 XDG_RUNTIME_DIR=/tmp/xdg CUDA_VISIBLE_DEVICES=0 \
      timeout 5400 $PY -u /root/build_snapshot_bank.py --demo-id $d --retries 6 \
      --bank /root/skill_start_bank_v21.json --out-dir $OUT --grip-fidelity > /root/bank_logs/d$d.log 2>&1
    rc=$?; rm -f /root/snapbank_tmp_$d.hdf5
    echo "[q$q] d$d done rc=$rc $(date -u +%H:%M:%S) manifest=$([ -f $OUT/manifest_d$d.json ] && echo yes || echo NO) $(grep -c 'BANKBUILD d'$d' s' /root/bank_logs/d$d.log) entries"
  done
  echo "[q$q] QUEUE_DONE"
}
A=(); B=(); i=0
for d in $DEMOS; do if [ $((i % 2)) -eq 0 ]; then A+=($d); else B+=($d); fi; i=$((i+1)); done
echo "queue A: ${A[*]}"; echo "queue B: ${B[*]}"
run_queue A "${A[@]}" &
sleep 240   # let the first sim boot before the second (shader cache, RAM peak)
run_queue B "${B[@]}" &
wait
echo "SNAPSHOT_BANK_REBUILD_DONE $(ls $OUT/manifest_d*.json 2>/dev/null | wc -l) manifests"

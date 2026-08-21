#!/bin/bash
# Launch N thread-capped replay processes over a split of turning_on_radio episodes.
#
# Scope rationale: this is a DIAGNOSTIC run, not a submission. The challenge scores over 100
# tasks, so a policy trained on 8-12 tasks scores ~0 on the rest — breadth cannot buy a
# submission at this data scale (winners used 10,000 demos; we have ~1k). What we need first is
# proof the architecture reaches nonzero q on ONE task, which is bought with per-task DEPTH.
#
# turning_on_radio is the right target: Comet measured plain pi0.5 at 0.30-0.60 there, so nonzero
# is demonstrably reachable, and its episodes are the shortest in the set (1957 steps vs up to
# 16208), so 200 episodes cost hours rather than days.
#
# Concurrency is CPU-bound, NOT VRAM-bound: 8 uncapped processes drove load to 373 with GPU at 0%
# and produced zero episodes in 50 minutes. Cap OMP/MKL and keep procs x threads well under nproc.
#
# Usage:  launch_radio.sh <split_json> <box1|box2> <n_procs> <out_root>
set -u
SPLIT=$1; BOX=$2; NPROC=$3; OUT=$4
PY=/root/miniconda3/envs/behavior/bin/python
OG=/root/bw/BEHAVIOR-1K/OmniGibson
OBJECTS=$($PY -c "import json;print(json.load(open('$SPLIT'))['objects'])")

# Isaac opens a lot of files and uses XDG_RUNTIME_DIR for locks/sockets. Sharing one dir between
# processes gives BlockingIOError (killed 5 of 6 on the first attempt), and the default fd limit
# is too low for several instances. Per-PROCESS runtime dir + raised ulimit.
ulimit -n 65536 2>/dev/null || echo "WARN: could not raise ulimit -n"

mkdir -p "$OUT"
for i in $(seq 0 $((NPROC-1))); do
  EPS=$($PY -c "import json;print(','.join(map(str,json.load(open('$SPLIT'))['$BOX'][$i])))")
  GPU=$((i % 4))
  mkdir -p "/tmp/xdg_${BOX}_$i"
  cd "$OG" && CUDA_VISIBLE_DEVICES=$GPU XDG_RUNTIME_DIR="/tmp/xdg_${BOX}_$i" \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
    nohup $PY /root/replay_poses_batch.py \
      --data_folder /root/replay_root --task turning_on_radio \
      --episodes "$EPS" --objects "$OBJECTS" --out_dir "$OUT" \
      > "/root/radio_p$i.log" 2>&1 &
  echo "launched P$i on gpu$GPU with $(echo $EPS | tr ',' '\n' | wc -l) episodes"
  sleep 3   # stagger: simultaneous Isaac starts contend badly
done
echo "ALL_LAUNCHED"

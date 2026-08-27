#!/usr/bin/env bash
# Full relabel_v1 sweep over every task-62 raw demo (CONTEXT_STATES_SPEC_62 §3), one sim per box.
# - waits for any running relabel_v1.py (e.g. the pilot) to exit first
# - chunks of $CHUNK demos per process so a sim crash loses at most one chunk; failed chunks are
#   logged and the sweep continues; demos with an existing ep{raw}.npz are skipped (resumable)
# - per-chunk timeout guards a hung Isaac process
# Usage: nohup setsid task62/relabel_sweep.sh > /root/step0/relabel_v1/sweep.log 2>&1 &
set -u
cd /root/behavior2026
OUT=${OUT:-/root/step0/relabel_v1}; CHUNK=${CHUNK:-5}; STRIDE=${STRIDE:-5}; TMO=${TMO:-8h}
PY=/root/miniconda3/envs/behavior/bin/python
unset DISPLAY; export XDG_RUNTIME_DIR=/tmp/xdg OMNI_KIT_ALLOW_ROOT=1 OMNIGIBSON_HEADLESS=1 PYTHONPATH=/root/behavior2026
mkdir -p "$OUT" /tmp/xdg
echo "[sweep] $(date -u +%FT%TZ) start; out=$OUT chunk=$CHUNK stride=$STRIDE"
while pgrep -f "relabel_v1.py" >/dev/null; do sleep 60; done
echo "[sweep] $(date -u +%FT%TZ) prior relabel process gone; starting"
ALL=$(ls /root/rawdemos/task-0062/episode_*.hdf5 | sed -E 's/.*episode_0*([0-9]+)\.hdf5/\1/' | sort -n)
todo=(); for ep in $ALL; do [ -f "$OUT/ep$ep.npz" ] || todo+=("$ep"); done
echo "[sweep] $(echo "$ALL" | wc -l) demos total, ${#todo[@]} to do"
i=0
while [ $i -lt ${#todo[@]} ]; do
  ids=$(IFS=,; echo "${todo[*]:$i:$CHUNK}"); i=$((i+CHUNK))
  echo "[sweep] $(date -u +%FT%TZ) chunk $ids"
  timeout -k 60 "$TMO" $PY -u task62/relabel_v1.py --demos "$ids" --stride "$STRIDE" --out "$OUT" \
     > "$OUT/log_${ids//,/_}.log" 2>&1
  rc=$?; done_n=$(ls "$OUT"/ep*.npz 2>/dev/null | wc -l)
  echo "[sweep] $(date -u +%FT%TZ) chunk $ids rc=$rc; $done_n/$(echo "$ALL" | wc -l) labeled"
  # sim leftovers from a crashed/timed-out chunk would block the next one
  pkill -f "relabel_v1.py" 2>/dev/null; sleep 5
done
echo "[sweep] $(date -u +%FT%TZ) SWEEP_DONE $(ls "$OUT"/ep*.npz | wc -l) labeled"

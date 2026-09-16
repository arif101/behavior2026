#!/bin/bash
# Gist precompute, 2-stage: (1) parallel decode-only workers (one per head-camera video file) -> frame cache memmaps
# under /root/frame_cache; (2) one tower pass per root from the cache (GPU, minutes). Writes PRECOMPUTE_ALL_DONE into
# /root/run3_logs/precompute_all.log for the downstream chains. PARAMS = the tower checkpoint (A4 by default).
# Usage: setsid nohup bash /root/run3/precompute_parallel.sh > /root/run3_logs/precompute_parallel.log 2>&1 &
set -u
P=/root/openpi_fork/.venv/bin/python; R3=/root/run3; L=/root/run3_logs; PARAMS=${PARAMS:-/root/ckpt_a4/params}
mkdir -p /root/frame_cache; cd /root/openpi_fork
say(){ echo "[precompute $(date -u +%H:%M:%S)] $*"; }
for r in /root/b1k_radio_map /root/manufactured/b1k_radio_factory /root/manufactured/b1k_radio_episodes /root/manufactured/b1k_radio_approach_v2; do
  n=$(basename $r); C=/root/frame_cache/$n.npy
  if [ -f $C.DONE ]; then say "$n: cache present, skip decode"; continue; fi
  JAX_PLATFORMS=cpu $P $R3/decode_frames.py --root $r --cache $C --init
  NF=$($P -c "import pyarrow.parquet as pq,glob; t=pq.read_table(sorted(glob.glob('$r/meta/episodes/**/*.parquet',recursive=True))[0]).to_pandas(); print(t['videos/observation.rgb.zed_link_camera_0/file_index'].nunique())")
  say "$n: decoding $NF video files in parallel"
  core=0
  for fi in $(seq 0 $((NF - 1))); do
    JAX_PLATFORMS=cpu taskset -c $((16 + (core % 96)))-$((16 + (core % 96) + 7)) $P $R3/decode_frames.py --root $r --cache $C --file-index $fi > $L/decode_${n}_$fi.log 2>&1 &
    core=$((core + 8))
  done
  wait
  grep -h DECODE_OK $L/decode_${n}_*.log | sed "s/^/  /"
  [ "$(grep -c DECODE_OK $L/decode_${n}_*.log | awk -F: '{s+=$2} END {print s}')" = "$NF" ] && touch $C.DONE || { say "$n: DECODE INCOMPLETE"; exit 1; }
done
for r in /root/b1k_radio_map /root/manufactured/b1k_radio_factory /root/manufactured/b1k_radio_episodes /root/manufactured/b1k_radio_approach_v2; do
  n=$(basename $r)
  XLA_PYTHON_CLIENT_PREALLOCATE=false $P $R3/precompute_gists.py --root $r --params $PARAMS --from-cache /root/frame_cache/$n.npy 2>&1 | grep -E "tower pass|PRECOMPUTE|Traceback|Error" | tee -a $L/precompute_all.log
done
echo PRECOMPUTE_ALL_DONE | tee -a $L/precompute_all.log

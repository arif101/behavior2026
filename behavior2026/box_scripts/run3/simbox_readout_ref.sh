#!/bin/bash
# SIM BOX: pull the readout mix, then run the STEP-0 reference paired-loss readout (FULL params = the 4dall warm start).
set -u; mkdir -p /root/run3_logs; PY=/root/openpi_fork/.venv/bin/python
say(){ echo "[readout_ref $(date -u +%m-%dT%H:%M:%S)] $*"; }
cd /root/openpi_fork
$PY /root/run3/dl_readout_mix.py --mix 2>&1 | grep -vE "it/s|%\|" | tail -6
grep -q . /root/b1k_radio_mix_readout/meta/run3_sources.json || { say "MIX_PULL_FAILED"; exit 1; }
$PY - <<'PY'
import json,glob,pyarrow.parquet as pq
r="/root/b1k_radio_mix_readout"; info=json.load(open(f"{r}/meta/info.json")); files=sorted(glob.glob(f"{r}/data/**/*.parquet",recursive=True))
rows=sum(pq.ParquetFile(f).metadata.num_rows for f in files); vids=glob.glob(f"{r}/videos/**/*.mp4",recursive=True)
print(f"SIMBOX_MIX eps={info['total_episodes']} frames={info['total_frames']} rows={rows} files={len(files)} videos={len(vids)}", flush=True)
print("SIMBOX_MIX_OK" if rows==info["total_frames"]==55095 and len(vids)==66 else "SIMBOX_MIX_FAILED", flush=True)
PY
grep -q SIMBOX_MIX_OK /root/run3_logs/readout_ref.out || { say "MIX_CHECK_FAILED"; exit 1; }
say "reference readout (FULL params) starting"
XLA_PYTHON_CLIENT_PREALLOCATE=false $PY /root/run3/paired_loss_readout.py --config pi05_radio_4d_all --params /root/run3_dl/full/params \
  --mix /root/b1k_radio_mix_readout --map /root/odart_episode_map.json --out /root/run3_logs/paired_full.json > /root/run3_logs/paired_full.out 2>&1
grep -E "PAIRED_RESULT|VERDICT|PAIRED_READOUT|Traceback|Error" /root/run3_logs/paired_full.out | tail -5
say READOUT_REF_DONE

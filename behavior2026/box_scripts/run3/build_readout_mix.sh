#!/bin/bash
# READOUT MIX (2026-09-29): factory twins + all 10 ODART roots -> /root/b1k_radio_mix_readout for paired_loss_readout.py on
# the sim box (the trainer GPU is owned by the 4dall arm). CPU-only, niced; sources already carry every derived column.
set -u
PY=/root/openpi_fork/.venv/bin/python; R3=/root/run3; M=/root/manufactured; OUT=/root/b1k_radio_mix_readout
say(){ echo "[readout_mix $(date -u +%m-%dT%H:%M:%S)] $*"; }
SRCS="$M/b1k_radio_factory $(ls -d $M/b1k_radio_odart_r* | sort)"
say "sources: $SRCS"
cd /root/openpi_fork
nice -n 10 $PY $R3/assemble_run3_mix.py --overwrite --sources $SRCS --out $OUT 2>&1 | grep -E "ASSEMBLED|Traceback|Error|assert" | tail -3
grep -q "ASSEMBLED $OUT" /root/run3_logs/readout_mix.out || { say "READOUT_ASSEMBLE_FAILED"; exit 1; }
nice -n 10 $PY $R3/deregister_depth_streams.py --root $OUT | tail -1
nice -n 10 $PY $R3/regroup_parquets.py --root $OUT --rows 16384 2>&1 | tail -2
$PY - <<'PY'
import json, glob, pyarrow.parquet as pq
root="/root/b1k_radio_mix_readout"; info=json.load(open(f"{root}/meta/info.json"))
files=sorted(glob.glob(f"{root}/data/**/*.parquet", recursive=True)); cols=set(); rows=0; maxrg=0
for f in files:
    pf=pq.ParquetFile(f); cols|=set(pf.schema_arrow.names); rows+=pf.metadata.num_rows; maxrg=max(maxrg, max(pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)))
feat=set(info["features"]); vids=sorted(glob.glob(f"{root}/videos/**/*.mp4", recursive=True))
print(f"READOUT_MIX_FEATURES eps={info['total_episodes']} frames={info['total_frames']} rows={rows} files={len(files)} videos={len(vids)} missing_in_features={sorted(cols-feat)} missing_in_parquet={sorted(feat-cols-{k for k in feat if k.startswith('observation.images') or k.startswith('observation.depth')})} max_rowgroup={maxrg}")
print("READOUT_MIX_OK" if rows==info["total_frames"] and maxrg<=16384 and vids else "READOUT_MIX_FAILED", flush=True)
PY
du -sh $OUT; say READOUT_MIX_DONE

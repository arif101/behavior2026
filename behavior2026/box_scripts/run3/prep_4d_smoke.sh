#!/bin/bash
# 4D DATA PREP, PART C (2026-09-25 03:10 UTC): part B's history tokens are correct but were written as ONE row group per file;
# the arrow parquet reader fails on the 229k-row map file ("OSError: List index overflow", int32 list offsets), so part B's
# parity/smoke are void. This part waits for part B to exit, regroups the mix parquets (<= 16384 rows per row group), verifies a
# full read of the biggest file, then parity smoke -> 40-step smoke -> PREP_4D_DONE. Log: /root/run3_logs/prep_4d_c.out
set -u
PY=/root/openpi_fork/.venv/bin/python; R3=/root/run3; L=/root/run3_logs; MIX=/root/b1k_radio_mix_4d
say(){ echo "[prep4d-c $(date -u +%m-%dT%H:%M:%S)] $*"; }
while pgrep -f "^bash /root/prep_4d_hist.sh|scripts/b1k/train_b1k.py pi05_radio_4d --exp_name=smoke4d|run3/parity_smoke_4d.py" >/dev/null; do sleep 20; done; say "part B exited"
$PY $R3/regroup_parquets.py --root $MIX --rows 16384 2>&1 | tail -9
grep -q REGROUP_OK $L/prep_4d_c.out || { say "REGROUP_FAILED"; exit 1; }
$PY - <<'PYX'
import glob, time, pyarrow.parquet as pq
t0 = time.time(); fs = sorted(glob.glob("/root/b1k_radio_mix_4d/data/**/*.parquet", recursive=True))
big = max(fs, key=lambda f: pq.ParquetFile(f).metadata.num_rows); t = pq.read_table(big, columns=["hist_tok", "index", "episode_index"])
print(f"FULL_READ_OK {big.split('/')[-1]} rows {t.num_rows} row_groups {pq.ParquetFile(big).metadata.num_row_groups} {time.time()-t0:.0f}s", flush=True)
PYX
grep -q FULL_READ_OK $L/prep_4d_c.out || { say "FULL_READ_FAILED"; exit 1; }
cd /root/openpi_fork
$PY $R3/parity_smoke_4d.py > $L/parity4d.log 2>&1; grep -E "PARITY|Traceback|Error" $L/parity4d.log | tail -5
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 B1K_STAGE_OVERSAMPLE=8 B1K_SAMPLE_WEIGHT_COL=sample_weight .venv/bin/python scripts/b1k/train_b1k.py pi05_radio_4d --exp_name=smoke4d --overwrite --num_train_steps=40 --save_interval=1000 --no-wandb-enabled > $L/smoke4d.log 2>&1
grep -oE "\[stage-oversample\][^\n]{0,60}|\[sample-weight\][^\n]{0,60}" $L/smoke4d.log | head -2
grep -oE "loss=[0-9.]+|grad_norm=[0-9.]+|Traceback|RESOURCE_EXHAUSTED" $L/smoke4d.log | tail -6 | tr "\n" " "; echo
echo "peak cgroup memory during prep+smoke: $(awk '{printf "%.0f GB", $1/2^30}' /sys/fs/cgroup/memory.peak) (cap $(awk '{printf "%.0f GB", $1/2^30}' /sys/fs/cgroup/memory.max)); oom_kill now: $(grep oom_kill /sys/fs/cgroup/memory.events)"
grep -qE "Traceback|RESOURCE_EXHAUSTED" $L/smoke4d.log || { rm -f /root/frame_cache/hist_toks_4d.npy /root/frame_cache/hist_toks_4d.done; say "token cache removed"; }
say PREP_4D_DONE

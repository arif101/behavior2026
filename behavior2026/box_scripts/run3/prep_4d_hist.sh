#!/bin/bash
# 4D DATA PREP, PART B (2026-09-25 02:00 UTC): the history-token stage of prep_4d_data.sh was cgroup-OOM-killed (the .tolist()
# parquet write of the 229k-row map file built ~7.5e9 Python floats; oom_kill=1 at the 286 GB cap), and the chain ran on
# without hist_tok -> its parity/smoke are void. This part waits for that chain to exit, then: hist tokens (fixed script,
# arrow-native, tokens memmapped so a rerun skips the tower pass) -> verify columns -> drop the frame cache -> parity -> 40-step
# smoke -> PREP_4D_DONE. Log: /root/run3_logs/prep_4d_b.out
set -u
PY=/root/openpi_fork/.venv/bin/python; R3=/root/run3; L=/root/run3_logs; MIX=/root/b1k_radio_mix_4d
say(){ echo "[prep4d-b $(date -u +%m-%dT%H:%M:%S)] $*"; }
while pgrep -f "^bash /root/prep_4d_data.sh" >/dev/null; do sleep 20; done; say "part A exited"
[ -f /root/frame_cache/mix_4d.npy ] || { say "FRAME_CACHE_MISSING"; exit 1; }
cd /root/openpi_fork
XLA_PYTHON_CLIENT_PREALLOCATE=false $PY $R3/precompute_hist_tokens.py --root $MIX --params /root/run3_dl/full/params \
  --from-cache /root/frame_cache/mix_4d.npy --toks-cache /root/frame_cache/hist_toks_4d.npy 2>&1 \
  | grep -E "tower pass|parquet:|PRECOMPUTE_HIST_TOKENS_OK|Traceback|Error|Killed" | tail -12
$PY - <<'PYX'
import glob, json, pyarrow.parquet as pq
fs = sorted(glob.glob("/root/b1k_radio_mix_4d/data/**/*.parquet", recursive=True)); bad = []
for f in fs:
    n = pq.ParquetFile(f).schema_arrow.names
    if not all(c in n for c in ("hist_tok", "hist_cellxyz", "hist_cellvalid", "odom_xyyaw")): bad.append(f.split("/data/")[1])
feats = json.load(open("/root/b1k_radio_mix_4d/meta/info.json"))["features"]
print("MIX4D_HIST files", len(fs), "missing_cols_in", bad, "features", [c for c in ("hist_tok","hist_cellxyz","hist_cellvalid","odom_xyyaw") if c not in feats] or "all registered")
print("MIX4D_HIST_OK" if not bad and all(c in feats for c in ("hist_tok","hist_cellxyz","hist_cellvalid","odom_xyyaw")) else "MIX4D_HIST_FAILED", flush=True)
PYX
grep -q MIX4D_HIST_OK $L/prep_4d_b.out || { say "HIST_STAGE_FAILED (frame cache kept)"; exit 1; }
rm -f /root/frame_cache/mix_4d.npy && say "frame cache removed (hist_toks_4d.npy kept until the smoke passes)"
[ -e /root/ckpt_4d_init/params ] || { mkdir -p /root/ckpt_4d_init && ln -sfn /root/run3_dl/full/params /root/ckpt_4d_init/params; }
md5sum /root/openpi_fork/outputs/assets/pi05_radio_4d/b1k_radio/norm_stats.json
$PY $R3/parity_smoke_4d.py 2>&1 | grep -E "PARITY|Traceback|Error" | tail -5
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 B1K_STAGE_OVERSAMPLE=8 B1K_SAMPLE_WEIGHT_COL=sample_weight .venv/bin/python scripts/b1k/train_b1k.py pi05_radio_4d --exp_name=smoke4d --overwrite --num_train_steps=40 --save_interval=1000 --no-wandb-enabled > $L/smoke4d.log 2>&1
grep -oE "\[stage-oversample\][^\n]{0,60}|\[sample-weight\][^\n]{0,60}" $L/smoke4d.log | head -2
grep -oE "loss=[0-9.]+|grad_norm=[0-9.]+|Traceback|RESOURCE_EXHAUSTED" $L/smoke4d.log | tail -6 | tr "\n" " "; echo
grep -qE "Traceback|RESOURCE_EXHAUSTED" $L/smoke4d.log || { rm -f /root/frame_cache/hist_toks_4d.npy /root/frame_cache/hist_toks_4d.done; say "token cache removed"; }
say PREP_4D_DONE

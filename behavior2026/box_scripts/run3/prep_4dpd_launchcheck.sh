#!/bin/bash
# POINTER-DROPOUT ARM pre-launch (2026-09-25): waits until the trainer GPU is free (no run3 driver / trainer / smoke),
# then parity (FULL warm start; pi05_radio_4d_pd must equal full at train=False: dropout+noise are train-only and the
# anchor-follow rule is exact when the pointer is present) -> 40-step smoke of pi05_radio_4d_pd with the memory sampler
# -> PREP_4DPD_DONE. Does NOT launch the driver. Log: /root/run3_logs/prep_4dpd.out
set -u
PY=/root/openpi_fork/.venv/bin/python; R3=/root/run3; L=/root/run3_logs
say(){ echo "[prep4dpd $(date -u +%m-%dT%H:%M:%S)] $*"; }
while pgrep -f "^bash /root/run3/run3_driver.sh|scripts/b1k/train_b1k.py|run3/parity_smoke_4d.py|run3/parity_diag" >/dev/null; do sleep 60; done; say "gpu free"
cd /root/openpi_fork
grep -q "pointer_anchor_follow" src/openpi/models/pi0.py && grep -q 'name="pi05_radio_4d_pd"' src/openpi/training/config.py && say "live fork has the pointer-dropout code" || { say "FORK_NOT_PATCHED"; exit 1; }
PARITY_EXTRA=pi05_radio_4d_pd $PY $R3/parity_smoke_4d.py > $L/parity4dpd.log 2>&1; grep -E "PARITY|Traceback|Error" $L/parity4dpd.log | tail -8
grep -q "PARITY_EXTRA pi05_radio_4d_pd" $L/prep_4dpd.out || { say "PARITY_FAILED (see parity4dpd.log)"; exit 1; }
( while true; do a=$(grep -E "^anon " /sys/fs/cgroup/memory.stat | awk '{print $2}'); c=$(cat /sys/fs/cgroup/memory.current); echo "$a $c"; sleep 5; done ) > $L/smoke4dpd_mem.samples 2>/dev/null &
SAMPLER=$!
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 B1K_STAGE_OVERSAMPLE=8 B1K_SAMPLE_WEIGHT_COL=sample_weight .venv/bin/python scripts/b1k/train_b1k.py pi05_radio_4d_pd --exp_name=smoke4dpd --overwrite --num_train_steps=40 --save_interval=1000 --no-wandb-enabled > $L/smoke4dpd.log 2>&1
kill $SAMPLER 2>/dev/null
grep -oE "\[stage-oversample\][^\n]{0,60}|\[sample-weight\][^\n]{0,60}" $L/smoke4dpd.log | head -2
grep -oE "loss=[0-9.]+|grad_norm=[0-9.]+|Traceback|RESOURCE_EXHAUSTED|Killed" $L/smoke4dpd.log | tail -6 | tr "\n" " "; echo
awk 'BEGIN{ma=0;mc=0} {if($1>ma)ma=$1; if($2>mc)mc=$2} END{printf "SMOKE_MEM max anon %.0f GB, max cgroup current %.0f GB, samples %d\n", ma/2^30, mc/2^30, NR}' $L/smoke4dpd_mem.samples
echo "oom_kill now: $(grep oom_kill /sys/fs/cgroup/memory.events)"
rm -rf /root/openpi_fork/outputs/checkpoints/pi05_radio_4d_pd/smoke4dpd
say PREP_4DPD_DONE

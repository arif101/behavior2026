#!/bin/bash
# 4D DATA PREP, PART F (2026-09-25 07:00 UTC): parity re-run against the FULL checkpoint (the real warm start; the A4-based
# check was confounded by the full config's random-init heads) + the 40-step smoke after the graphdef fix (module-level
# identity initializer). Waits for any running diag. Log: /root/run3_logs/prep_4d_f.out
set -u
PY=/root/openpi_fork/.venv/bin/python; R3=/root/run3; L=/root/run3_logs
say(){ echo "[prep4d-f $(date -u +%m-%dT%H:%M:%S)] $*"; }
while pgrep -f "run3/parity_diag2?.py|train_b1k.py|run3/parity_smoke_4d.py" >/dev/null; do sleep 15; done; say "gpu free"
cd /root/openpi_fork
grep -q "_eye_init" src/openpi/models/pi0.py && grep -q "tk tc td" src/openpi/models/model.py && say "fork has the graphdef + axis fixes" || { say "FORK_NOT_PATCHED"; exit 1; }
$PY $R3/parity_smoke_4d.py > $L/parity4d_d.log 2>&1; grep -E "PARITY|Traceback|Error" $L/parity4d_d.log | tail -7
grep -q PARITY_RESULT $L/prep_4d_f.out || { say "PARITY_FAILED (see parity4d_d.log)"; exit 1; }
( while true; do a=$(grep -E "^anon " /sys/fs/cgroup/memory.stat | awk '{print $2}'); c=$(cat /sys/fs/cgroup/memory.current); echo "$a $c"; sleep 5; done ) > $L/smoke4d_mem.samples 2>/dev/null &
SAMPLER=$!
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 B1K_STAGE_OVERSAMPLE=8 B1K_SAMPLE_WEIGHT_COL=sample_weight .venv/bin/python scripts/b1k/train_b1k.py pi05_radio_4d --exp_name=smoke4d --overwrite --num_train_steps=40 --save_interval=1000 --no-wandb-enabled > $L/smoke4d.log 2>&1
kill $SAMPLER 2>/dev/null
grep -oE "\[stage-oversample\][^\n]{0,60}|\[sample-weight\][^\n]{0,60}" $L/smoke4d.log | head -2
grep -oE "loss=[0-9.]+|grad_norm=[0-9.]+|Traceback|RESOURCE_EXHAUSTED|Killed" $L/smoke4d.log | tail -6 | tr "\n" " "; echo
awk 'BEGIN{ma=0;mc=0} {if($1>ma)ma=$1; if($2>mc)mc=$2} END{printf "SMOKE_MEM max anon %.0f GB, max cgroup current %.0f GB (cap %.0f GB), samples %d\n", ma/2^30, mc/2^30, 285999996928/2^30, NR}' $L/smoke4d_mem.samples
echo "oom_kill now: $(grep oom_kill /sys/fs/cgroup/memory.events)"
grep -qE "Traceback|RESOURCE_EXHAUSTED|Killed" $L/smoke4d.log || { rm -f /root/frame_cache/hist_toks_4d.npy /root/frame_cache/hist_toks_4d.done; say "token cache removed"; }
say PREP_4D_DONE

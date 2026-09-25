#!/bin/bash
# 4D DATA PREP, PART E (2026-09-25 04:40 UTC): the parity + 40-step smoke both failed on a jaxtyping axis-name collision
# (history_gists "*b hk hd" binds hk=9; history_tokens "*b hk hc hd" has hk=8) -> Observation.from_dict raised on every
# batch. model.py renamed the 4D history axes to tk/tc/td. This part re-runs the guarded parity smoke, then the 40-step
# smoke with a memory sampler (memory.peak is stuck at the cap since part A's OOM, so the container's live anon/current
# are sampled every 5 s and the max reported). Log: /root/run3_logs/prep_4d_e.out
set -u
PY=/root/openpi_fork/.venv/bin/python; R3=/root/run3; L=/root/run3_logs
say(){ echo "[prep4d-e $(date -u +%m-%dT%H:%M:%S)] $*"; }
cd /root/openpi_fork
$PY - <<'PYX'
import numpy as np
from openpi.models import model as _m
b = {"image": {"base_0_rgb": np.zeros((2,224,224,3), np.float32)}, "image_mask": {"base_0_rgb": np.ones(2, bool)}, "state": np.zeros((2,32), np.float32),
     "history_gists": np.zeros((2,9,2048), np.float32), "history_mask": np.ones((2,9), bool),
     "history_tokens": np.zeros((2,8,16,2048), np.float32), "history_xyz": np.zeros((2,8,16,3), np.float32), "history_valid": np.ones((2,8,16), bool), "history_dt": np.zeros((2,8), np.float32)}
o = _m.Observation.from_dict(b); print("OBS_TYPECHECK_OK gists", o.history_gists.shape, "tokens", o.history_tokens.shape, flush=True)
PYX
grep -q OBS_TYPECHECK_OK $L/prep_4d_e.out || { say "OBS_TYPECHECK_FAILED"; exit 1; }
$PY $R3/parity_smoke_4d.py > $L/parity4d_c.log 2>&1; grep -E "PARITY|Traceback|Error" $L/parity4d_c.log | tail -5
grep -q PARITY_RESULT $L/prep_4d_e.out || { say "PARITY_FAILED (see parity4d_c.log)"; exit 1; }
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

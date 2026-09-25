#!/bin/bash
# 4D DATA PREP, PART D (2026-09-25 03:55 UTC): part C's parity smoke died because parity_smoke_4d.py had no __main__ guard
# (spawned loader workers re-ran the module). Part C continues into the 40-step smoke (valid, keep it). This part waits for
# part C to exit, then runs the guarded parity smoke. Log: /root/run3_logs/prep_4d_d.out (full: parity4d_b.log)
set -u
PY=/root/openpi_fork/.venv/bin/python; R3=/root/run3; L=/root/run3_logs
say(){ echo "[prep4d-d $(date -u +%m-%dT%H:%M:%S)] $*"; }
while pgrep -f "^bash /root/prep_4d_smoke.sh|scripts/b1k/train_b1k.py pi05_radio_4d --exp_name=smoke4d|run3/parity_smoke_4d.py" >/dev/null; do sleep 20; done; say "part C exited"
cd /root/openpi_fork
$PY $R3/parity_smoke_4d.py > $L/parity4d_b.log 2>&1; grep -E "PARITY|Traceback|Error" $L/parity4d_b.log | tail -5
say PARITY_STAGE_DONE

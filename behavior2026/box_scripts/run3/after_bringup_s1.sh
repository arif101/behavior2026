#!/bin/bash
# S1 (= Run-3 arm a5 on the FIXED approach data): prep -> preflight -> GPU smoke -> driver (a5, 15k steps). Stops at the first failure.
# Usage: [START_STAGE=2] setsid nohup bash /root/run3/after_bringup_s1.sh > /root/run3_logs/after_bringup_s1.out 2>&1 &
set -eo pipefail
PY=/root/openpi_fork/.venv/bin/python; R3=/root/run3; L=/root/run3_logs; mkdir -p $L
say(){ echo "[s1 $(date -u +%m-%dT%H:%M:%S)] $*" | tee -a $L/after_bringup_s1.log; }
ST=${START_STAGE:-2}
if [ $ST -le 2 ]; then
  say "2. data prep (mix_a5 with approach_v2)"
  bash $R3/prep_s1_data.sh > $L/prep_s1.log 2>&1 || { say "PREP FAILED"; tail -25 $L/prep_s1.log; exit 1; }
  grep -E "advisory|ASSEMBLED|STAGE_WEIGHTS|DEPTH_DEREGISTERED|registered|DONE|PREP_S1|frames|episodes" $L/prep_s1.log | tail -12 | tee -a $L/after_bringup_s1.log
  grep -q PREP_S1_DATA_OK $L/prep_s1.log
fi
if [ $ST -le 3 ]; then
  say "3. preflight a5 (CPU)"
  cd /root/openpi_fork
  JAX_PLATFORMS=cpu $PY $R3/preflight_run3.py a5 > $L/preflight_s1.log 2>&1 || { say "PREFLIGHT FAILED"; grep -vE "Warning|^\s*$" $L/preflight_s1.log | tail -15; exit 1; }
  grep -E "PASS|FAIL|frames|mass|max\||PREFLIGHT" $L/preflight_s1.log | tee -a $L/after_bringup_s1.log
  grep -q PREFLIGHT_RUN3_PASS $L/preflight_s1.log
fi
if [ $ST -le 4 ]; then
  say "4. GPU smoke: a5, 20 steps, real data, checkpoint write"
  cd /root/openpi_fork; rm -rf outputs/checkpoints/pi05_radio_run3_a5/smoke
  PYTHONUNBUFFERED=1 B1K_STAGE_OVERSAMPLE=8 B1K_SAMPLE_WEIGHT_COL=sample_weight XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 \
    $PY scripts/b1k/train_b1k.py pi05_radio_run3_a5 --exp_name smoke --overwrite --num_train_steps 20 \
    --save_interval 1000 --log_interval 5 --no-wandb-enabled > $L/smoke_a5.log 2>&1 \
    || { say "SMOKE FAILED"; grep -E "Traceback|Error|RESOURCE|Killed" $L/smoke_a5.log | tail -5; exit 1; }
  grep -E "\[stage-oversample\]|\[sample-weight\]" $L/smoke_a5.log | tee -a $L/after_bringup_s1.log
  grep -iE "loss" $L/smoke_a5.log | tail -3 | tee -a $L/after_bringup_s1.log
  SM=$(ls -d outputs/checkpoints/pi05_radio_run3_a5/smoke/[0-9]* 2>/dev/null | grep -v tmp | tail -1)
  test -d "$SM/params" || { say "SMOKE: no committed checkpoint"; exit 1; }
  grep -q "\[sample-weight\]" $L/smoke_a5.log
  rm -rf outputs/checkpoints/pi05_radio_run3_a5/smoke; say "SMOKE_OK"
fi
say "5. launching driver: arm a5, 15k steps"
ARMS="a5" RUN3_STEPS=15000 setsid nohup bash $R3/run3_driver.sh > $L/driver.out 2>&1 < /dev/null &
echo $! > $L/driver.pid; say "driver pid $(cat $L/driver.pid)"
echo AFTER_BRINGUP_S1_OK | tee -a $L/after_bringup_s1.log

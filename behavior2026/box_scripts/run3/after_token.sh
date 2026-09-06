#!/bin/bash
# RUN-3: everything after /root/.hf_token lands — gated stages, stops at the first failure.
#   1 downloads (map backup, manufactured, Run-2 params+assets; parallel)
#   2 prep_run3_data.sh   (map videos, sample weights, dry-run 296/464,242, per-arm mixes, gates)
#   3 preflight_run3.py   (CPU: tree identity, sampler mass per source, one real batch per arm)
#   4 GPU smoke           (a2, 20 steps, real data, loader weighting lines, checkpoint write)
#   5 run3_driver.sh      (AUTOLAUNCH=1 default; arms a0 a2 a1 a3 a4 a5; 15k steps each)
# Usage: [START_STAGE=4] setsid nohup bash /root/run3/after_token.sh > /root/run3_logs/after_token.out 2>&1 &
set -eo pipefail
PY=/root/openpi_fork/.venv/bin/python; R3=/root/run3; L=/root/run3_logs; mkdir -p $L
test -s /root/.hf_token || { echo "no /root/.hf_token"; exit 1; }; chmod 600 /root/.hf_token
say(){ echo "[after_token $(date -u +%m-%dT%H:%M:%S)] $*" | tee -a $L/after_token.log; }

ST=${START_STAGE:-1}
if [ $ST -le 1 ]; then
say "1. downloads (parallel)"
for w in map manu ckpt; do (setsid nohup $PY $R3/dl_run3.py $w > $L/dl_$w.log 2>&1 &); done
sleep 5
for w in map manu ckpt; do
  until grep -qE "DL_OK|Traceback" $L/dl_$w.log; do sleep 20; done
  grep -q DL_OK $L/dl_$w.log || { say "download $w FAILED"; tail -5 $L/dl_$w.log; exit 1; }
  say "download $w ok"
done
du -sh /root/backup /root/manufactured /root/warmstart_run3_raw | tee -a $L/after_token.log
ls /root/manufactured | tee -a $L/after_token.log
ls /root/warmstart_run3_raw/assets/ | tee -a $L/after_token.log
fi

if [ $ST -le 2 ]; then
say "2. data prep"
bash $R3/prep_run3_data.sh > $L/prep.log 2>&1 || { say "PREP FAILED"; tail -25 $L/prep.log; exit 1; }
grep -E "MERGED|advisory|ASSEMBLED|DEPTH_DEREGISTERED|registered|DONE|PREP_RUN3" $L/prep.log | tee -a $L/after_token.log
grep -q PREP_RUN3_DATA_OK $L/prep.log
fi

if [ $ST -le 3 ]; then
say "3. preflight (CPU)"
cd /root/openpi_fork
JAX_PLATFORMS=cpu $PY $R3/preflight_run3.py a0 a1 a2 a3 a4 > $L/preflight.log 2>&1 || { say "PREFLIGHT FAILED"; grep -vE "Warning|^\s*$" $L/preflight.log | tail -15; exit 1; }
grep -E "PASS|FAIL|frames|PREFLIGHT" $L/preflight.log | tee -a $L/after_token.log
grep -q PREFLIGHT_RUN3_PASS $L/preflight.log
fi

say "4. GPU smoke: a2, 20 steps, real data, checkpoint write"
rm -rf outputs/checkpoints/pi05_radio_run3_a2/smoke
PYTHONUNBUFFERED=1 B1K_STAGE_OVERSAMPLE=8 B1K_SAMPLE_WEIGHT_COL=sample_weight XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 \
  $PY scripts/b1k/train_b1k.py pi05_radio_run3_a2 --exp_name smoke --overwrite --num_train_steps 20 \
  --save_interval 1000 --log_interval 5 --no-wandb-enabled > $L/smoke_a2.log 2>&1 \
  || { say "SMOKE FAILED"; grep -E "Traceback|Error|RESOURCE|Killed" $L/smoke_a2.log | tail -5; exit 1; }
grep -E "\[stage-oversample\]|\[sample-weight\]" $L/smoke_a2.log | tee -a $L/after_token.log
grep -E "loss|step" $L/smoke_a2.log | grep -iE "loss" | tail -3 | tee -a $L/after_token.log
SM=$(ls -d outputs/checkpoints/pi05_radio_run3_a2/smoke/[0-9]* 2>/dev/null | grep -v tmp | tail -1)
test -d "$SM/params" || { say "SMOKE: no committed checkpoint"; ls outputs/checkpoints/pi05_radio_run3_a2/smoke/; exit 1; }
du -sh $SM | tee -a $L/after_token.log
grep -q "\[sample-weight\]" $L/smoke_a2.log
rm -rf outputs/checkpoints/pi05_radio_run3_a2/smoke
df -h / | tail -1 | tee -a $L/after_token.log
say "SMOKE_OK"

if [ "${AUTOLAUNCH:-1}" = "1" ]; then
  say "5. launching driver (arms a0 a2 a1 a3 a4 a5, 15k steps)"
  ARMS="a0 a2 a1 a3 a4 a5" RUN3_STEPS=15000 setsid nohup bash $R3/run3_driver.sh > $L/driver.out 2>&1 &
  echo $! > $L/driver.pid; say "driver pid $(cat $L/driver.pid)"
fi
echo AFTER_TOKEN_OK | tee -a $L/after_token.log

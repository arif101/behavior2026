#!/bin/bash
# Serve one arm/step and run the SAME 10 instances, so results are comparable across checkpoints.
ARM=$1; STEP=$2; OUT=/root/eval_${ARM}_${STEP}_N10
cd /root/openpi_adaln
setsid env CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/b1k/serve_b1k.py \
  --robot b1k/R1Pro --task b1k/turning_on_radio --repo-id b1k_phaseA \
  --policy.config pi05_phaseA_point --policy.dir /root/phaseA_ckpts/$ARM/$STEP --port 8901 \
  > /root/serve_${ARM}_${STEP}.log 2>&1 &
for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q 8901 && break; sleep 5; done
ss -ltn | grep -q 8901 || { echo "SERVER_FAILED"; exit 1; }
echo "SERVER_UP $ARM/$STEP"
cd /workspace/BEHAVIOR-1K/OmniGibson
CUDA_VISIBLE_DEVICES=0 OMNIGIBSON_HEADLESS=1 B1K_TASK_NAME=turning_on_radio \
PYTHONPATH=/workspace/BEHAVIOR-1K/OmniGibson /root/miniconda3/envs/behavior/bin/python \
  -m omnigibson.eval.eval --task-name turning_on_radio --robot-config omnigibson/eval/r1pro.yaml \
  --mode public_test --host 127.0.0.1 --port 8901 --instance-indices 0 1 2 3 4 5 6 7 8 9 \
  --env-wrapper eval.oracle_point_wrapper.OraclePointWrapper \
  --output-dir $OUT --no-write-video
echo "EVAL_DONE $ARM $STEP"

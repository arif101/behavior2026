#!/bin/bash
# Full-budget rollout WITH video, on instance index 1 (= instance_id 302), which reached
# norm_base 0.76 (near-optimal positioning) and still scored 0 -- the most diagnostic case
# for a terminal-commit failure. Own server on 8902 so it does not contend with the running eval.
set -x
cd /root/openpi_adaln
setsid env CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/b1k/serve_b1k.py \
  --robot b1k/R1Pro --task b1k/turning_on_radio --repo-id b1k_phaseA \
  --policy.config pi05_phaseA_point --policy.dir /root/phaseA_ckpts/phaseA_weighted/30000 \
  --port 8902 > /root/serve_video.log 2>&1 &
for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q 8902 && break; sleep 5; done
ss -ltn | grep -q 8902 || { echo "SERVER_FAILED"; exit 1; }
echo "SERVER_UP 8902"
cd /workspace/BEHAVIOR-1K/OmniGibson
CUDA_VISIBLE_DEVICES=2 OMNIGIBSON_HEADLESS=1 B1K_TASK_NAME=turning_on_radio \
PYTHONPATH=/workspace/BEHAVIOR-1K/OmniGibson /root/miniconda3/envs/behavior/bin/python \
  -m omnigibson.eval.eval --task-name turning_on_radio --robot-config omnigibson/eval/r1pro.yaml \
  --mode public_test --host 127.0.0.1 --port 8902 --instance-indices 1 \
  --env-wrapper eval.oracle_point_wrapper.OraclePointWrapper \
  --output-dir /root/eval_video_fulltime --write-video
echo "VIDEO_RUN_DONE rc=$?"

#!/bin/bash
set -x
export PATH=/root/miniconda3/bin:/root/.local/bin:$PATH
CKPT=/root/g2_ckpts/pi05_b1k_ours_lora/g2_radio_lora/5000
for MODE in base gov; do
  ENV=""
  [ "$MODE" = "gov" ] && ENV="GOVERNOR=1"
  cd /workspace/openpi
  nohup env $ENV CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 uv run scripts/b1k/serve_b1k.py \
    --robot b1k/R1Pro --task b1k/turning_on_radio --repo-id turning_on_radio \
    --policy.config pi05_b1k_ours_lora --policy.dir "$CKPT" \
    --control_mode receding_horizon --action_horizon 16 --port 8000 > /workspace/serve_radio_$MODE.log 2>&1 &
  for i in $(seq 1 40); do ss -tln | grep -q 8000 && break; sleep 10; done
  ss -tln | grep -q 8000 || { echo SERVE_FAILED_$MODE; exit 1; }
  echo SERVE_UP_$MODE
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate behavior
  cd /workspace/BEHAVIOR-1K
  TRACE_PATH=/workspace/traces_radio_$MODE.jsonl python -m omnigibson.eval.eval --task-name turning_on_radio \
    --robot-config /workspace/robot_openpi.yaml --host 127.0.0.1 --port 8000 \
    --output-dir ./eval_logs/radio_$MODE --write-video > /workspace/eval_radio_$MODE.log 2>&1
  echo EVAL_DONE_$MODE
  grep -E "Result:" /workspace/eval_radio_$MODE.log | tail -1
  P=$(pgrep -f "serve_b1k" | head -1); [ -n "$P" ] && kill $P; sleep 5
done
echo RADIO_AB_DONE

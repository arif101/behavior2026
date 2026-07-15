#!/bin/bash
set -x
export PATH=/root/miniconda3/bin:/root/.local/bin:$PATH
# wait for training to finish (process exit)
while pgrep -f "python.*train_b1[k]" >/dev/null; do sleep 300; done
sleep 30
CKPT=$(ls -d /root/g2_ckpts/pi05_b1k_trash_lora/trash_lora/* | sort -n -t/ -k6 | tail -1)
echo USING_CKPT $CKPT
cd /workspace/openpi
nohup env CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 uv run scripts/b1k/serve_b1k.py \
  --robot b1k/R1Pro --task b1k/picking_up_trash --repo-id picking_up_trash \
  --policy.config pi05_b1k_trash_lora --policy.dir "$CKPT" \
  --control_mode receding_horizon --action_horizon 16 --port 8000 > /workspace/serve_trash.log 2>&1 &
for i in $(seq 1 40); do ss -tln | grep -q 8000 && break; sleep 10; done
ss -tln | grep -q 8000 || { echo SERVE_FAILED; exit 1; }
echo SERVE_UP
source /root/miniconda3/etc/profile.d/conda.sh && conda activate behavior
cd /workspace/BEHAVIOR-1K
for IDX in 0 1 2 3 4; do
  TRACE_PATH=/workspace/traces_trash_i$IDX.jsonl python -m omnigibson.eval.eval --task-name picking_up_trash \
    --robot-config /workspace/robot_openpi.yaml --mode public_test --instance-indices $IDX \
    --host 127.0.0.1 --port 8000 --output-dir ./eval_logs/trash_i$IDX --write-video > /workspace/eval_trash_i$IDX.log 2>&1
  echo INSTANCE_${IDX}_DONE
  grep -E "Result:" /workspace/eval_trash_i$IDX.log | tail -1
  /root/miniconda3/envs/behavior/bin/python /workspace/probes/analyze_rollout.py \
    --trace /workspace/traces_trash_i$IDX.jsonl --out /workspace/verdict_trash_i$IDX.json 2>/dev/null | tail -3
done
P=$(pgrep -f "serve_b1k" | head -1); [ -n "$P" ] && kill $P
echo TRASH_GATE_DONE

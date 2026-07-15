#!/bin/bash
set -x
export PATH=/root/miniconda3/bin:/root/.local/bin:$PATH
# stop training at 10k ckpt
while [ ! -d /root/g2_ckpts/pi05_b1k_wood_lora/wood_lora/10000 ]; do
  pgrep -f "python.*train_b1[k]" >/dev/null || break
  sleep 120
done
sleep 180
pkill -f "python.*train_b1[k]" && echo TRAIN_STOPPED_10K
sleep 10
CKPT=$(ls -d /root/g2_ckpts/pi05_b1k_wood_lora/wood_lora/10000 2>/dev/null || ls -d /root/g2_ckpts/pi05_b1k_wood_lora/wood_lora/5000)
echo USING_CKPT $CKPT
sed "s/robot_r1/robot/g" /workspace/BEHAVIOR-1K/OmniGibson/omnigibson/eval/r1pro.yaml > /workspace/robot_openpi.yaml
cd /workspace/openpi
nohup env CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 uv run scripts/b1k/serve_b1k.py \
  --robot b1k/R1Pro --task b1k/bringing_in_wood --repo-id bringing_in_wood \
  --policy.config pi05_b1k_wood_lora --policy.dir "$CKPT" \
  --control_mode receding_horizon --action_horizon 16 --port 8000 > /workspace/serve_wood.log 2>&1 &
for i in $(seq 1 40); do ss -tln | grep -q 8000 && break; sleep 10; done
ss -tln | grep -q 8000 || { echo SERVE_FAILED; exit 1; }
echo SERVE_UP
source /root/miniconda3/etc/profile.d/conda.sh && conda activate behavior
cd /workspace/BEHAVIOR-1K
python -m omnigibson.eval.eval --task-name bringing_in_wood --robot-config /workspace/robot_openpi.yaml \
  --host 127.0.0.1 --port 8000 --output-dir ./eval_logs/wood_base --write-video > /workspace/eval_wood_base.log 2>&1
echo BASE_EVAL_DONE
grep -E "Result:" /workspace/eval_wood_base.log | tail -1
# governor A/B: restart server with GOVERNOR=1
pkill -f "serve_b1[k]"; sleep 5
cd /workspace/openpi
nohup env GOVERNOR=1 CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 uv run scripts/b1k/serve_b1k.py \
  --robot b1k/R1Pro --task b1k/bringing_in_wood --repo-id bringing_in_wood \
  --policy.config pi05_b1k_wood_lora --policy.dir "$CKPT" \
  --control_mode receding_horizon --action_horizon 16 --port 8000 > /workspace/serve_wood_gov.log 2>&1 &
for i in $(seq 1 40); do ss -tln | grep -q 8000 && break; sleep 10; done
cd /workspace/BEHAVIOR-1K
python -m omnigibson.eval.eval --task-name bringing_in_wood --robot-config /workspace/robot_openpi.yaml \
  --host 127.0.0.1 --port 8000 --output-dir ./eval_logs/wood_gov --write-video > /workspace/eval_wood_gov.log 2>&1
echo GOV_EVAL_DONE
grep -E "Result:" /workspace/eval_wood_gov.log | tail -1
pkill -f "serve_b1[k]"
echo WOOD_GATE_DONE

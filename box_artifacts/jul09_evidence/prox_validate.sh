#!/bin/bash
set -x
export PATH=/root/miniconda3/bin:/root/.local/bin:$PATH
pgrep -f "serve_b1[k]" | xargs -r kill; sleep 10
cd /workspace/openpi
nohup env CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 uv run scripts/b1k/serve_b1k.py \
  --robot b1k/R1Pro --task b1k/turning_on_radio --repo-id turning_on_radio \
  --policy.config pi05_b1k --policy.dir /root/baseline_ckpt/pi05_turn_on_the_radio \
  --control_mode receding_horizon --action_horizon 16 --port 8000 > /workspace/serve_proxval.log 2>&1 &
for i in $(seq 1 60); do ss -tln | grep -q 8000 && break; sleep 10; done
ss -tln | grep -q 8000 || { echo SERVE_FAILED; exit 1; }
sleep 20; echo SERVE_UP_proxval
source /root/miniconda3/etc/profile.d/conda.sh && conda activate behavior; hash -r
cd /workspace/BEHAVIOR-1K
export TARGET_CATS=radio_receiver
export TRACE_PATH=/workspace/traces_proxval_i0.jsonl
rm -f $TRACE_PATH
python -m omnigibson.eval.eval --task-name turning_on_radio \
  --robot-config /workspace/robot_openpi.yaml --mode public_test --instance-indices 0 \
  --host 127.0.0.1 --port 8000 --output-dir ./eval_logs/proxval_i0 --write-video \
  > /workspace/eval_proxval_i0.log 2>&1
echo PROXVAL_INSTANCE_0_DONE
grep -E "Result:" /workspace/eval_proxval_i0.log | tail -1
/root/miniconda3/envs/behavior/bin/python /workspace/probes/analyze_rollout.py \
  --trace $TRACE_PATH --out /workspace/verdict_proxval_i0.json | tail -8
pgrep -f "serve_b1[k]" | xargs -r kill
echo PROXVAL_DONE

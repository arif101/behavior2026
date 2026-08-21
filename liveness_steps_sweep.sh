#!/bin/bash
# Denoise-step control (Qwen-RobotManip finding): is the point channel dead at 20/32 steps too?
set -x
for NS in 20 32; do
  /root/miniconda3/envs/openpi/bin/python -u /root/point_liveness_steps.py \
    --config-name pi05_radio_run2 --ckpt /root/ckpt_run2 \
    --dataset-root /root/b1k_radio_map --num-steps $NS \
    --out /root/point_liveness_run2_ns${NS}.json >> /root/point_liveness_steps.log 2>&1
  echo "LIVENESS_NS${NS}_RC=$?"
done
echo LIVENESS_STEPS_SWEEP_DONE

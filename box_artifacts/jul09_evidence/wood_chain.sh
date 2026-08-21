#!/bin/bash
set -x
export PATH=/root/.local/bin:$PATH
cd /workspace/openpi
uv run scripts/compute_norm_stats.py --config-name pi05_b1k_wood_lora --max-frames 80000 > /workspace/wood_norm.log 2>&1
[ -f outputs/assets/pi05_b1k_wood_lora/bringing_in_wood/norm_stats.json ] || { echo WOOD_NS_MISSING; exit 1; }
echo WOOD_NS_DONE
mkdir -p /root/g2_ckpts
nohup env XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/b1k/train_b1k.py pi05_b1k_wood_lora --exp_name wood_lora --overwrite --no-wandb-enabled --checkpoint-base-dir /root/g2_ckpts > /workspace/train_wood.log 2>&1 &
echo WOOD_TRAIN_LAUNCHED

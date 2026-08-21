#!/bin/bash
# Day-0 2-proc VRAM/throughput test: two concurrent proprio-only sims, staggered boot.
set -x
cd /root
PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 OMNIGIBSON_HEADLESS=1 \
  /root/miniconda3/envs/behavior/bin/python -u v2_day0_profile.py \
  --demo-id 20 --modalities proprio --out /root/day0_2proc_1.json \
  > /root/day0_2proc_1.log 2>&1 &
P1=$!
sleep 60
PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 OMNIGIBSON_HEADLESS=1 \
  /root/miniconda3/envs/behavior/bin/python -u v2_day0_profile.py \
  --demo-id 170 --modalities proprio --out /root/day0_2proc_2.json \
  > /root/day0_2proc_2.log 2>&1 &
P2=$!
for i in $(seq 1 30); do
  sleep 30
  nvidia-smi --query-gpu=memory.used --format=csv,noheader >> /root/day0_2proc_vram.log
done &
wait $P1 $P2
echo DAY0_2PROC_DONE

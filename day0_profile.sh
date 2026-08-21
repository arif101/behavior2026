#!/bin/bash
# Day-0 profiler driver (SKILL_TRAINER_V2_SPEC.md d0). Runs config A (v1 modalities)
# then config B (render-off). Sequential — each boots its own sim. ~35-45 min total.
set -x
cd /root
PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 OMNIGIBSON_HEADLESS=1 \
  /root/miniconda3/envs/behavior/bin/python -u v2_day0_profile.py \
  --demo-id 20 --modalities rgb,proprio --out /root/day0_profile_A.json \
  > /root/day0_profile_A.log 2>&1
PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 OMNIGIBSON_HEADLESS=1 \
  /root/miniconda3/envs/behavior/bin/python -u v2_day0_profile.py \
  --demo-id 20 --modalities proprio --out /root/day0_profile_B.json \
  > /root/day0_profile_B.log 2>&1
echo DAY0_PROFILE_SUITE_DONE

#!/bin/bash
# BEHAVIOR-2026 100-task grounding-label sweep launcher.
# Usage: bash sweep_100.sh [--only task_a,task_b] [--skip-upload]
# Full detached launch:
#   setsid nohup bash /root/probes/sweep_100.sh > /root/sweep_100_driver.log 2>&1 &
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=$(cat /root/.hf_token)  # do not set -x in this script: token would leak into logs
set -x
source /root/miniconda3/etc/profile.d/conda.sh && conda activate behavior
cd /root/BEHAVIOR-1K/OmniGibson
exec taskset -c 0-23 python -u /root/probes/sweep_driver.py "$@"

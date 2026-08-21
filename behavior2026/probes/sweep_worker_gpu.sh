#!/bin/bash
# usage: sweep_worker_gpu.sh <lane> <gpu> <cores> <task_list>
W=$1; GPU=$2; CORES=$3; LIST=$4
export CUDA_VISIBLE_DEVICES=$GPU
export OMNIGIBSON_APPDATA_PATH=/root/og_appdata_gpu$GPU
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=$(cat /root/.hf_token)
source /root/miniconda3/etc/profile.d/conda.sh && conda activate behavior
cd /root/BEHAVIOR-1K/OmniGibson
exec taskset -c $CORES python -u /root/probes/sweep_driver_$W.py --skip-upload --only "$LIST"

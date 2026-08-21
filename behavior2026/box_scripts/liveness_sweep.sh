#!/bin/bash
# Same seed, same frames, same noise for every checkpoint -> differences are the checkpoint.
for CK in phaseA_weighted/10000 phaseA_weighted/20000 phaseA_weighted/30000 phaseA_uniform/10000 phaseA_uniform/20000; do
  OUT=/root/liveness_$(echo $CK | tr / _).json
  [ -f $OUT ] && continue
  cd /root/openpi_adaln
  CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python /root/point_liveness.py     --ckpt /root/phaseA_ckpts/$CK --dataset-root /root/slice_labelled     --episodes-json /root/slice_labelled/labelled_episodes.json --n 32 --seed 0 --out $OUT     >> /root/liveness_sweep.log 2>&1
  echo "DONE $CK"
done
echo LIVENESS_SWEEP_DONE

#!/bin/bash
set -x
for d in 20 80 110 190 260 300 340 420; do
  PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 OMNIGIBSON_HEADLESS=1 \
    /root/miniconda3/envs/behavior/bin/python -u /root/harvest_start_states.py \
    --demo-id "$d" --budget 600 --port 8901
  echo "HARVEST_CHUNK_DONE demo=$d rc=$?"
done
echo HARVEST_PASS_DONE

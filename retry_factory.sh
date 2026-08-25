#!/bin/bash
for D in 110 170 190 200; do
  M="/root/factory_clips/d$(printf %03d "$D")_meta.json"
  for OFF in 4 8 10 12; do
    [ -f "$M" ] && grep -q '"ok": true' "$M" && break
    echo "=== RETRY d$D t0off=$OFF $(date -u +%H:%M:%S) ==="
    OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 \
      OMNIGIBSON_HEADLESS=1 timeout 3600 \
      /root/miniconda3/envs/behavior/bin/python -u \
      /root/factory_grasp_transport.py --demo "$D" --t0off "$OFF" \
      >> /root/factory_retry.log 2>&1
    rm -f "/root/fct_tmp_${D}.hdf5"
  done
done
echo RETRY_COMPLETE

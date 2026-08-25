#!/bin/bash
# Overnight factory sweep: one process per demo (og.clear forbidden), sequential.
# d30 already proven; d20 first (other focus scene), then the rest.
DEMOS="20 10 40 50 60 70 80 100 110 120 130 140 160 170 180 190 200 210 220 230 240 250 260 270 280 290 300 310 320 330 340 350 360 370 380 390 400 410 420"
for D in $DEMOS; do
  if [ -f "/root/factory_clips/d$(printf %03d "$D")_meta.json" ]; then
    echo "SKIP d$D (meta exists)"
    continue
  fi
  echo "=== FACTORY d$D start $(date -u +%H:%M:%S) ==="
  OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 \
    OMNIGIBSON_HEADLESS=1 timeout 3600 \
    /root/miniconda3/envs/behavior/bin/python -u \
    /root/factory_grasp_transport.py --demo "$D" \
    >> /root/factory_sweep.log 2>&1
  echo "=== FACTORY d$D done rc=$? $(date -u +%H:%M:%S) ==="
  rm -f "/root/fct_tmp_${D}.hdf5"
done
echo "SWEEP_COMPLETE"

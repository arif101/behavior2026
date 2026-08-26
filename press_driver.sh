#!/bin/bash
until grep -q "REEL_COMPLETE" /root/factory_reel.log 2>/dev/null || [ "$(ls /root/factory_reel/*.mp4 2>/dev/null | wc -l)" -ge 6 ]; do sleep 120; done
echo "reel done, starting press segments $(date -u +%H:%M:%S)"
for D in 30 20 10 100 190 260; do
  [ -f "/root/factory_clips_press/d$(printf %03d "$D")_meta.json" ] && continue
  echo "=== PRESS-SEG d$D $(date -u +%H:%M:%S) ==="
  OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 \
    OMNIGIBSON_HEADLESS=1 timeout 3600 \
    /root/miniconda3/envs/behavior/bin/python -u \
    /root/factory_press_segment.py --demo "$D" >> /root/factory_press.log 2>&1
  rm -f "/root/fct_tmp_${D}.hdf5"
done
echo PRESS_SEGMENTS_COMPLETE

#!/bin/bash
for M in /root/factory_clips/d*_meta.json; do
  D=$((10#$(basename "$M" | sed 's/d\([0-9]*\)_meta.json/\1/')))
  grep -q '"ok": true' "$M" || continue
  [ -f "/root/factory_reel/d$(printf %03d "$D").mp4" ] && continue
  echo "=== REEL d$D $(date -u +%H:%M:%S) ==="
  OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 \
    OMNIGIBSON_HEADLESS=1 timeout 3600 \
    /root/miniconda3/envs/behavior/bin/python -u \
    /root/render_factory_clips.py --demo "$D" >> /root/factory_reel.log 2>&1
  rm -f "/root/rr_tmp_${D}.hdf5"
done
echo REEL_ALL_COMPLETE

#!/bin/bash
# After the v5 d20 run exits: if it exported a clip, re-render it (3-camera obs,
# determinism check) and, on DELTA=0.0000 with all cameras, start the night run.
PYA='^/root/miniconda3/envs/behavior/bin/python -u /root/factory_approach.py'
while pgrep -f "$PYA" >/dev/null; do sleep 20; done
CLIP=/root/factory_clips_approach/d020_approach.npz
if [ ! -f "$CLIP" ]; then echo "RESULT verify: no d020_approach.npz exported — night run NOT started" >> /root/approach_runs.log; exit 1; fi
find /tmp -mindepth 1 -maxdepth 1 -type d -user root -name 'tmp????????' -mmin +120 -exec rm -rf {} + 2>/dev/null
OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 OMNIGIBSON_HEADLESS=1 \
  timeout 5400 /root/miniconda3/envs/behavior/bin/python -u /root/obs_render_factory2.py --demo 20 \
  --clipdir /root/factory_clips_approach --suffix approach --tag a > /root/obs2_approach.log 2>&1
rm -f /root/or2_tmp_20.hdf5
L=$(grep -a "OBS2_DONE" /root/obs2_approach.log | tail -1)
echo "RESULT render d20 approach: ${L:-NO OBS2_DONE LINE}" >> /root/approach_runs.log
if echo "$L" | grep -q "left_ok=True right_ok=True DELTA=0.0000"; then
  echo "RESULT render verified — launching approach night run" >> /root/approach_runs.log
  exec /root/approach_night.sh
else
  echo "RESULT render check FAILED — night run NOT started" >> /root/approach_runs.log
fi

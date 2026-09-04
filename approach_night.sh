#!/bin/bash
# Approach factory across all factory-clip demos (v5 recipe). Single sim; anchored waits.
PY='^/root/miniconda3/envs/behavior/bin/python -u /root/(factory_approach_cap|relay_episode_factory).py'
while pgrep -f "$PY|^/root/miniconda3/envs/behavior/bin/python -u /root/obs_render_factory2.py" >/dev/null; do sleep 60; done
for f in /root/factory_clips/d*_grasp_transport.npz; do
  D=$((10#$(basename "$f" | sed 's/d\([0-9]*\)_.*/\1/')))
  [ -f "/root/factory_obs2/rac_${D}_200.npz" ] && continue
  find /tmp -mindepth 1 -maxdepth 1 -type d -user root -name 'tmp????????' -mmin +120 -exec rm -rf {} + 2>/dev/null
  echo "=== APPROACH-NIGHT d$D $(date -u +%H:%M:%S) ===" >> /root/approach_runs.log
  OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 OMNIGIBSON_HEADLESS=1 \
    timeout 5400 /root/miniconda3/envs/behavior/bin/python -u /root/factory_approach_cap.py --demo "$D" \
    >> /root/approach_runs.log 2>&1
  rm -f "/root/fap_tmp_${D}.hdf5"
done
echo "APPROACH_NIGHT_COMPLETE" >> /root/approach_runs.log

#!/bin/bash
# Run the approach-factory smoke on d20 once the episode converter releases the
# 50GB cgroup (sim + converter cannot coexist).
while pgrep -f "convert_clips_to_parque[t]|convert_episodes_afte[r]" >/dev/null; do sleep 120; done
sleep 30
find /tmp -mindepth 1 -maxdepth 1 -type d -user root -name 'tmp????????' -mmin +120 -exec rm -rf {} + 2>/dev/null
echo "=== APPROACH d20 $(date -u +%H:%M:%S) ===" >> /root/approach_runs.log
OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 OMNIGIBSON_HEADLESS=1 \
  timeout 5400 /root/miniconda3/envs/behavior/bin/python -u /root/factory_approach.py --demo 20 \
  >> /root/approach_runs.log 2>&1
rm -f /root/fap_tmp_20.hdf5
echo "APPROACH_SMOKE_DONE" >> /root/approach_runs.log

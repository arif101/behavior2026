#!/bin/bash
# d20 approach clip with orient-first (single pass) -> relay x6 on it (filmed) -> night run
E="OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 OMNIGIBSON_HEADLESS=1"
echo "=== APPROACH-CAP d20 v6 (orient-first) $(date -u +%H:%M:%S) ===" >> /root/approach_runs.log
env $E timeout 5400 /root/miniconda3/envs/behavior/bin/python -u /root/factory_approach_cap.py --demo 20 >> /root/approach_runs.log 2>&1
rm -f /root/fap_tmp_20.hdf5
if [ -f /root/factory_clips_approach/d020_approach.npz ]; then
  echo "=== FULLCHAIN d20 v6 x6 $(date -u +%H:%M:%S) ===" >> /root/ef_runs.log
  env $E timeout 9000 /root/miniconda3/envs/behavior/bin/python -u /root/relay_episode_factory.py --demo 20 \
    --clipdir /root/factory_clips_approach --suffix approach --tag 300 --tries 6 --budget 600 --film >> /root/ef_runs.log 2>&1
  rm -f /root/ref_tmp_20.hdf5
else
  echo "RESULT v6 d20 produced no clip — night run starts on v6 recipe anyway" >> /root/approach_runs.log
fi
exec /root/approach_night.sh

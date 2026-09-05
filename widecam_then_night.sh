#!/bin/bash
E="OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 OMNIGIBSON_HEADLESS=1"
echo "=== WIDECAM d20 $(date -u +%H:%M:%S) ===" >> /root/ef_runs.log
env $E timeout 9000 /root/miniconda3/envs/behavior/bin/python -u /root/relay_episode_factory.py --demo 20 \
  --clipdir /root/factory_clips_approach --suffix approach --tag 400 --tries 4 --budget 600 --film --widecam >> /root/ef_runs.log 2>&1
rm -f /root/ref_tmp_20.hdf5
echo "WIDECAM_DONE" >> /root/ef_runs.log
# resume the approach night (bonus A5 data) and its second pass
export APPROACH_PASS=1; nohup /root/approach_night.sh >> /root/approach_night.out 2>&1 &
nohup /root/approach_second_pass.sh >> /root/approach_second_pass.out 2>&1 &

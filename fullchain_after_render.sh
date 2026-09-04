#!/bin/bash
# When the d20 approach render check exits, run the policy relay ON THE APPROACH CLIP
# (full episode: honest approach -> closure -> carry -> transport -> learned press), filmed.
while pgrep -f '^/root/miniconda3/envs/behavior/bin/python -u /root/obs_render_factory2.py' >/dev/null; do sleep 5; done
echo "=== FULLCHAIN d20 (relay on approach clip) $(date -u +%H:%M:%S) ===" >> /root/ef_runs.log
OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 OMNIGIBSON_HEADLESS=1 \
  timeout 7200 /root/miniconda3/envs/behavior/bin/python -u /root/relay_episode_factory.py --demo 20 \
  --clipdir /root/factory_clips_approach --suffix approach --tag 300 --tries 3 --budget 600 --film \
  >> /root/ef_runs.log 2>&1
rm -f /root/ref_tmp_20.hdf5
echo "FULLCHAIN_DONE" >> /root/ef_runs.log

#!/bin/bash
# Night sequencer v2 (single sim at all times):
#   1. wait for the running d260 rerun (raycast-era margin) to finish
#   2. retry d190 and d260 with the AABB support check + radio-half-extent margin
#   3. hand off to the overnight v2 obs batch
clean_scratch() { find /tmp -mindepth 1 -maxdepth 1 -type d -user root -name 'tmp????????' -mmin +120 -exec rm -rf {} + 2>/dev/null; }
while pgrep -f "chain_fix_rerun.sh|rt_replay_test51" >/dev/null; do sleep 60; done
for D in 190 260; do
  clean_scratch
  echo "=== RERUN2 d$D (AABB + half-extent margin) $(date -u +%H:%M:%S) ===" >> /root/chain_sweep2.log
  OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 \
    OMNIGIBSON_HEADLESS=1 timeout 9000 \
    /root/miniconda3/envs/behavior/bin/python -u \
    /root/rt_replay_test51.py --demo "$D" >> /root/chain_sweep2.log 2>&1
  rm -f /root/rtt51_tmp.hdf5
done
echo "RERUN2_COMPLETE" >> /root/chain_sweep2.log
exec /root/obs2_batch.sh

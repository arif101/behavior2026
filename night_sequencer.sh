#!/bin/bash
# Night sequencer (single sim at all times):
#   1. wait for the d190/d260 rerun driver to finish
#   2. re-run d190 once more with the AABB support fix (its raycast-era run
#      placed the radio on the floor)
#   3. hand off to the overnight v2 obs batch (obs2_batch.sh has its own
#      process-wait as double insurance)
clean_scratch() { find /tmp -mindepth 1 -maxdepth 1 -type d -user root -name 'tmp????????' -mmin +120 -exec rm -rf {} + 2>/dev/null; }
while pgrep -f "chain_fix_rerun.sh|rt_replay_test51" >/dev/null; do sleep 60; done
clean_scratch
echo "=== RERUN2 d190 (AABB support fix) $(date -u +%H:%M:%S) ===" >> /root/chain_sweep2.log
OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 \
  OMNIGIBSON_HEADLESS=1 timeout 9000 \
  /root/miniconda3/envs/behavior/bin/python -u \
  /root/rt_replay_test51.py --demo 190 >> /root/chain_sweep2.log 2>&1
rm -f /root/rtt51_tmp.hdf5
echo "RERUN2_COMPLETE" >> /root/chain_sweep2.log
exec /root/obs2_batch.sh

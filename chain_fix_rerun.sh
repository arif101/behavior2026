#!/bin/bash
# Post-sweep addendum: wait for the v2 obs validation to free the sim, then
# re-run d190 (anchor fix) and d260 (support-surface fix) with the patched
# rt_replay_test51.py. Appends to chain_sweep2.log so the armed monitor fires.
clean_scratch() { find /tmp -mindepth 1 -maxdepth 1 -type d -user root -name 'tmp????????' -mmin +120 -exec rm -rf {} + 2>/dev/null; }
while pgrep -f "obs_render_factory2|rt_replay_test51" >/dev/null; do sleep 60; done
for D in 190 260; do
  clean_scratch
  echo "=== RERUN d$D (fixed anchor+support) $(date -u +%H:%M:%S) ===" >> /root/chain_sweep2.log
  OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 \
    OMNIGIBSON_HEADLESS=1 timeout 9000 \
    /root/miniconda3/envs/behavior/bin/python -u \
    /root/rt_replay_test51.py --demo "$D" >> /root/chain_sweep2.log 2>&1
  rm -f /root/rtt51_tmp.hdf5
done
clean_scratch
echo "RERUN_COMPLETE" >> /root/chain_sweep2.log

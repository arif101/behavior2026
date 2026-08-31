#!/bin/bash
clean_scratch() { find /tmp -mindepth 1 -maxdepth 1 -type d -user root -name 'tmp????????' -mmin +120 -exec rm -rf {} + 2>/dev/null; }
for D in 20 100 190 260 310; do
  clean_scratch
  echo "=== CHAIN d$D $(date -u +%H:%M:%S) ==="
  OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 \
    OMNIGIBSON_HEADLESS=1 timeout 9000 \
    /root/miniconda3/envs/behavior/bin/python -u \
    /root/rt_replay_test50.py --demo "$D" >> /root/chain_sweep.log 2>&1
  rm -f /root/rtt50_tmp.hdf5
done
clean_scratch
echo CHAIN_SWEEP_COMPLETE

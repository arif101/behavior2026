#!/bin/bash
clean_scratch() { find /tmp -mindepth 1 -maxdepth 1 -type d -user root -name 'tmp????????' -mmin +120 -exec rm -rf {} + 2>/dev/null; }
for D in 100 190 260 310; do
  clean_scratch
  echo "=== TCHAIN d$D $(date -u +%H:%M:%S) ==="
  OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 \
    OMNIGIBSON_HEADLESS=1 timeout 9000 \
    /root/miniconda3/envs/behavior/bin/python -u \
    /root/rt_replay_test51.py --demo "$D" >> /root/chain_sweep2.log 2>&1
  rm -f /root/rtt51_tmp.hdf5
done
clean_scratch
OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 OMNIGIBSON_HEADLESS=1 timeout 5400 /root/miniconda3/envs/behavior/bin/python -u /root/obs_render_factory2.py --demo 10 >> /root/obs2_test.log 2>&1
rm -f /root/or2_tmp_10.hdf5
echo SWEEP2_COMPLETE

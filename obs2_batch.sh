#!/bin/bash
# Overnight v2 obs batch: after the chain reruns free the sim, render every
# factory clip (3 cameras + base_pose) that doesn't already have a v2 capture.
# ~1h/clip => runs into tomorrow; each clip logs OBS2_DONE to obs2_test.log
# (monitored). Single sim process at all times.
clean_scratch() { find /tmp -mindepth 1 -maxdepth 1 -type d -user root -name 'tmp????????' -mmin +120 -exec rm -rf {} + 2>/dev/null; }
while pgrep -f "rt_replay_test51|obs_render_factory2" >/dev/null; do sleep 120; done
for f in /root/factory_clips/d*_grasp_transport.npz; do
  D=$((10#$(basename "$f" | sed 's/d\([0-9]*\)_.*/\1/')))
  [ -f "/root/factory_obs2/rac_${D}_0.npz" ] && continue
  clean_scratch
  echo "=== OBS2 d$D $(date -u +%H:%M:%S) ===" >> /root/obs2_test.log
  OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 \
    OMNIGIBSON_HEADLESS=1 timeout 9000 \
    /root/miniconda3/envs/behavior/bin/python -u \
    /root/obs_render_factory2.py --demo "$D" >> /root/obs2_test.log 2>&1
  rm -f "/root/or2_tmp_${D}.hdf5"
done
clean_scratch
echo "OBS2_BATCH_COMPLETE" >> /root/obs2_test.log

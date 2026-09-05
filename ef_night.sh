#!/bin/bash
# Overnight relay episode factory: after the v2 obs batch completes, run the
# policy-only episode factory across every factory-clip demo — 3 stochastic
# draws each (snapshot restores make draws 2-3 cheap), budget 600 steps.
# Saves rac_<demo>_r<try>.npz per success; logs a toggle/no-toggle verdict per
# draw for the success-rate table.
clean_scratch() { find /tmp -mindepth 1 -maxdepth 1 -type d -user root -name 'tmp????????' -mmin +120 -exec rm -rf {} + 2>/dev/null; }
while pgrep -f "obs_render_factor[y]2|obs2_batc[h].sh|rt_replay_test5[12]" >/dev/null; do sleep 120; done
for f in /root/factory_clips/d*_grasp_transport.npz; do
  D=$((10#$(basename "$f" | sed 's/d\([0-9]*\)_.*/\1/')))
  ls /root/factory_obs2/rac_${D}_r*.npz >/dev/null 2>&1 && continue
  clean_scratch
  echo "=== EFN d$D $(date -u +%H:%M:%S) ===" >> /root/ef_runs.log
  OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 \
    OMNIGIBSON_HEADLESS=1 timeout 9000 \
    /root/miniconda3/envs/behavior/bin/python -u \
    /root/relay_episode_factory.py --demo "$D" --tries 3 --budget 600 \
    >> /root/ef_runs.log 2>&1
  rm -f "/root/ref_tmp_${D}.hdf5"
done
clean_scratch
echo "EF_NIGHT_COMPLETE" >> /root/ef_runs.log

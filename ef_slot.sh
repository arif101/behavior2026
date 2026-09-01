#!/bin/bash
# Boundary-triggered slot: wait for the NEXT OBS2_DONE (current render finishes),
# then preempt the batch, run the relay episode factory smoke on d20 (2 tries),
# and relaunch the batch.
N0=$(grep -ac "OBS2_DONE" /root/obs2_test.log)
while [ "$(grep -ac "OBS2_DONE" /root/obs2_test.log)" -le "$N0" ]; do sleep 30; done
BP=$(pgrep -f "obs2_batc[h].sh" | head -1)
[ -n "$BP" ] && kill "$BP" 2>/dev/null
RP=$(pgrep -f "obs_render_factor[y]2" | head -1)
if [ -n "$RP" ]; then
  kill "$RP" 2>/dev/null
  sleep 8
  kill -9 "$RP" 2>/dev/null
fi
sleep 3
rm -f /root/or2_tmp_*.hdf5
echo "=== EF d20 $(date -u +%H:%M:%S) ===" >> /root/ef_runs.log
OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 \
  OMNIGIBSON_HEADLESS=1 timeout 7200 \
  /root/miniconda3/envs/behavior/bin/python -u \
  /root/relay_episode_factory.py --demo 20 --tries 2 >> /root/ef_runs.log 2>&1
rm -f /root/ref_tmp_20.hdf5
nohup /root/obs2_batch.sh >> /root/obs2_batch.out 2>&1 &
echo "EF_SLOT_DONE batch relaunched" >> /root/ef_runs.log

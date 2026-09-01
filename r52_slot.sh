#!/bin/bash
# R52 validation slot: preempt the batch at the current boundary, run the
# scripted in-hand press on d30, relaunch the batch.
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
echo "=== R52 d30 $(date -u +%H:%M:%S) ===" >> /root/r52_runs.log
OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 \
  OMNIGIBSON_HEADLESS=1 timeout 9000 \
  /root/miniconda3/envs/behavior/bin/python -u \
  /root/rt_replay_test52.py --demo 30 >> /root/r52_runs.log 2>&1
rm -f /root/rtt52_tmp_30.hdf5
nohup /root/obs2_batch.sh >> /root/obs2_batch.out 2>&1 &
echo "R52_SLOT_DONE batch relaunched" >> /root/r52_runs.log

#!/bin/bash
# Preemption slot: pause the obs2 batch at a clip boundary, run the policy relay
[ -f /root/factory_clips/d020_grasp_transport.npz ] || { echo "RELAY_SLOT_ABORT no clip" >> /root/relay_d20.log; exit 1; }
# on d30, then relaunch the batch (idempotent — it skips existing captures).
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
echo "=== RELAY d20 $(date -u +%H:%M:%S) ===" >> /root/relay_d20.log
OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 \
  OMNIGIBSON_HEADLESS=1 timeout 5400 \
  /root/miniconda3/envs/behavior/bin/python -u \
  /root/rt_relay_test1.py --demo 20 >> /root/relay_d20.log 2>&1
rm -f /root/relay_tmp_20.hdf5
nohup /root/obs2_batch.sh >> /root/obs2_batch.out 2>&1 &
echo "RELAY_SLOT_DONE batch relaunched" >> /root/relay_d20.log

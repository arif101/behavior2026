#!/bin/bash
# Preemption slot: pause the obs2 batch at a clip boundary, run instrumented
# policy relays (d20 toggle-mechanism confirm + d100 generalization), then
# relaunch the batch (idempotent — it skips existing captures).
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
for D in 20 100; do
  [ -f "/root/factory_clips/d$(printf %03d "$D")_grasp_transport.npz" ] || continue
  echo "=== RELAY d$D $(date -u +%H:%M:%S) ===" >> /root/relay_runs.log
  OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 \
    OMNIGIBSON_HEADLESS=1 timeout 5400 \
    /root/miniconda3/envs/behavior/bin/python -u \
    /root/rt_relay_test1.py --demo "$D" >> /root/relay_runs.log 2>&1
  rm -f "/root/relay_tmp_${D}.hdf5"
done
nohup /root/obs2_batch.sh >> /root/obs2_batch.out 2>&1 &
echo "RELAY_SLOT_DONE batch relaunched" >> /root/relay_runs.log

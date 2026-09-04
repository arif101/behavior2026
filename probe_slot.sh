#!/bin/bash
# At the next night RESULT boundary: pause, probe d20 d40 d50 geometry (~5 min each), resume.
CAP='^/root/miniconda3/envs/behavior/bin/python -u /root/factory_approach_cap'
N0=$(grep -ac "^RESULT" /root/approach_runs.log)
while [ "$(grep -ac '^RESULT' /root/approach_runs.log)" -le "$N0" ]; do sleep 30; done
kill $(pgrep -f '^/bin/bash /root/approach_night.sh') 2>/dev/null
P=$(pgrep -f "$CAP"); [ -n "$P" ] && kill $P && sleep 8 && kill -9 $P 2>/dev/null; rm -f /root/fap_tmp_*.hdf5
for D in 20 40 50; do
  echo "=== PROBE d$D $(date -u +%H:%M:%S) ===" >> /root/approach_runs.log
  OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 OMNIGIBSON_HEADLESS=1 \
    timeout 1800 /root/miniconda3/envs/behavior/bin/python -u /root/probe_demo_geometry.py --demos $D >> /root/approach_runs.log 2>&1
  rm -f /root/pdg_tmp_$D.hdf5
done
echo "RESULT probes done (d20 d40 d50)" >> /root/approach_runs.log
exec /root/approach_night.sh

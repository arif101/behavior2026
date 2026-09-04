#!/bin/bash
# At the next night RESULT boundary: pause the night, dump d20's canonical grasp geometry
# (~5 min), resume the night. Anchored patterns only.
CAP='^/root/miniconda3/envs/behavior/bin/python -u /root/factory_approach_cap'
N0=$(grep -ac "^RESULT" /root/approach_runs.log)
while [ "$(grep -ac '^RESULT' /root/approach_runs.log)" -le "$N0" ]; do sleep 30; done
kill $(pgrep -f '^/bin/bash /root/approach_night.sh') 2>/dev/null
P=$(pgrep -f "$CAP"); [ -n "$P" ] && kill $P && sleep 8 && kill -9 $P 2>/dev/null; rm -f /root/fap_tmp_*.hdf5
echo "=== CANON dump d20 $(date -u +%H:%M:%S) ===" >> /root/approach_runs.log
OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 OMNIGIBSON_HEADLESS=1 \
  timeout 1800 /root/miniconda3/envs/behavior/bin/python -u /root/dump_canonical_grasp.py --demo 20 >> /root/approach_runs.log 2>&1
rm -f /root/dcg_tmp_20.hdf5
[ -f /root/canonical_grasp_d20.json ] && echo "RESULT canon dump OK" >> /root/approach_runs.log || echo "RESULT canon dump FAILED" >> /root/approach_runs.log
exec /root/approach_night.sh

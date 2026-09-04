#!/bin/bash
# Validate v10 (adaptive K + retreat-first staging + canonical grasp) on d40 at the next boundary.
CAP='^/root/miniconda3/envs/behavior/bin/python -u /root/(factory_approach_cap|probe_demo_geometry)'
N0=$(grep -ac "^RESULT" /root/approach_runs.log)
while [ "$(grep -ac '^RESULT' /root/approach_runs.log)" -le "$N0" ]; do sleep 20; done
kill $(pgrep -f '^/bin/bash /root/approach_night.sh') 2>/dev/null
P=$(pgrep -f "$CAP"); [ -n "$P" ] && kill $P && sleep 8 && kill -9 $P 2>/dev/null; rm -f /root/fap_tmp_*.hdf5 /root/pdg_tmp_*.hdf5
echo "=== APPROACH-CAP d40 v10 (retreat-first, adaptive K) $(date -u +%H:%M:%S) ===" >> /root/approach_runs.log
OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 OMNIGIBSON_HEADLESS=1 \
  timeout 9000 /root/miniconda3/envs/behavior/bin/python -u /root/factory_approach_cap_v10.py --demo 40 >> /root/approach_runs.log 2>&1
rm -f /root/fap_tmp_40.hdf5
if [ -f /root/factory_obs2/rac_40_200.npz ]; then cp /root/factory_approach_cap_v10.py /root/factory_approach_cap.py; echo "RESULT v10 d40 SAVED — v10 promoted to the night/second-pass recipe" >> /root/approach_runs.log; fi
exec /root/approach_night.sh

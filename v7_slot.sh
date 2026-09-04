#!/bin/bash
# Tomorrow: at the next approach-night demo boundary, pause the night, validate v7 on d20,
# relay x6 on the v7 clip (filmed), then resume the night. Anchored patterns only.
CAP='^/root/miniconda3/envs/behavior/bin/python -u /root/factory_approach_cap.py'
N0=$(grep -ac "^RESULT" /root/approach_runs.log)
while [ "$(grep -ac '^RESULT' /root/approach_runs.log)" -le "$N0" ]; do sleep 30; done
kill $(pgrep -f '^/bin/bash /root/approach_night.sh') 2>/dev/null
P=$(pgrep -f "$CAP"); [ -n "$P" ] && kill $P && sleep 8 && kill -9 $P 2>/dev/null; rm -f /root/fap_tmp_*.hdf5
mkdir -p /root/factory_clips_approach/v7_d20; mv /root/factory_clips_approach/d020_* /root/factory_clips_approach/v7_d20/ 2>/dev/null; mv /root/factory_obs2/rac_20_200.npz /root/factory_clips_approach/v7_d20/rac_20_200_v5.npz 2>/dev/null
E="OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 OMNIGIBSON_HEADLESS=1"
echo "=== APPROACH-CAP d20 v7 (gentle orient) $(date -u +%H:%M:%S) ===" >> /root/approach_runs.log
env $E timeout 9000 /root/miniconda3/envs/behavior/bin/python -u /root/factory_approach_cap_v7.py --demo 20 >> /root/approach_runs.log 2>&1
rm -f /root/fap_tmp_20.hdf5
if [ -f /root/factory_clips_approach/d020_approach.npz ]; then
  echo "=== FULLCHAIN d20 v7 x6 $(date -u +%H:%M:%S) ===" >> /root/ef_runs.log
  env $E timeout 9000 /root/miniconda3/envs/behavior/bin/python -u /root/relay_episode_factory.py --demo 20 \
    --clipdir /root/factory_clips_approach --suffix approach --tag 300 --tries 6 --budget 600 --film >> /root/ef_runs.log 2>&1
  rm -f /root/ref_tmp_20.hdf5
fi
exec /root/approach_night.sh

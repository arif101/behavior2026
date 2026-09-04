#!/bin/bash
# wait for the current v7 d20 run to exit, rerun v7 d20 with the corrected honesty
# gate, relay x6 on it (filmed), then night on v7 if d20 certified (else v5)
CAP='^/root/miniconda3/envs/behavior/bin/python -u /root/factory_approach_cap'
while pgrep -f "$CAP" >/dev/null; do sleep 10; done
rm -f /root/fap_tmp_20.hdf5; mkdir -p /root/factory_clips_approach/v7a_d20; mv /root/factory_clips_approach/d020_* /root/factory_clips_approach/v7a_d20/ 2>/dev/null
E="OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 OMNIGIBSON_HEADLESS=1"
echo "=== APPROACH-CAP d20 v7b (precontact gate) $(date -u +%H:%M:%S) ===" >> /root/approach_runs.log
env $E timeout 5400 /root/miniconda3/envs/behavior/bin/python -u /root/factory_approach_cap_v7.py --demo 20 >> /root/approach_runs.log 2>&1
rm -f /root/fap_tmp_20.hdf5
if [ -f /root/factory_clips_approach/d020_approach.npz ]; then
  cp /root/factory_approach_cap_v7.py /root/factory_approach_cap.py   # v7 becomes the night recipe
  echo "=== FULLCHAIN d20 v7 x6 $(date -u +%H:%M:%S) ===" >> /root/ef_runs.log
  env $E timeout 9000 /root/miniconda3/envs/behavior/bin/python -u /root/relay_episode_factory.py --demo 20 \
    --clipdir /root/factory_clips_approach --suffix approach --tag 300 --tries 6 --budget 600 --film >> /root/ef_runs.log 2>&1
  rm -f /root/ref_tmp_20.hdf5
else
  echo "RESULT v7b d20 no clip — night continues on v5 recipe" >> /root/approach_runs.log
fi
exec /root/approach_night.sh

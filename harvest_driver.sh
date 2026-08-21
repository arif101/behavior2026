#!/bin/bash
# State-harvest driver: bring up the run-2 policy server (port 8901; provider nginx squats
# 8000), wait for the port, then harvest a representative subset of train demos, one
# process per demo. If the sim OOMs against the server, restart the server with
# JAX_PLATFORMS=cpu per harvest_start_states.py's fallback note.
set -x
PY=/root/miniconda3/envs/openpi/bin/python
cd /root/openpi_fork
setsid nohup $PY scripts/b1k/serve_b1k.py \
  --robot b1k/R1Pro --task b1k/turning_on_radio --repo-id b1k_radio \
  --policy.config pi05_radio_run2 --policy.dir /root/ckpt --port 8901 \
  > /root/serve_harvest.log 2>&1 < /dev/null &
for i in $(seq 1 90); do ss -ltn 2>/dev/null | grep -q 8901 && break; sleep 5; done
ss -ltn | grep -q 8901 || { echo "HARVEST_SERVER_FAILED"; exit 1; }
echo "HARVEST_SERVER_UP"

for d in 20 80 110 190 260 300 340 420; do
  PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 OMNIGIBSON_HEADLESS=1 \
    /root/miniconda3/envs/behavior/bin/python -u /root/harvest_start_states.py \
    --demo-id "$d" --budget 600 --port 8901
  echo "HARVEST_CHUNK_DONE demo=$d rc=$?"
done
echo HARVEST_PASS_DONE

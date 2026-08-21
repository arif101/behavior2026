#!/bin/bash
# Day-0: commit-probe re-confirm on radio_run2@49999. Server on 8000 (probe's native port),
# left running afterwards for the key-causality baseline.
set -x
PY=/root/miniconda3/envs/openpi/bin/python
cd /root/openpi_fork
setsid nohup $PY scripts/b1k/serve_b1k.py \
  --robot b1k/R1Pro --task b1k/turning_on_radio --repo-id b1k_radio \
  --policy.config pi05_radio_run2 --policy.dir /root/ckpt --port 8000 \
  > /root/serve_day0.log 2>&1 < /dev/null &
for i in $(seq 1 90); do ss -ltn 2>/dev/null | grep -q ":8000" && break; sleep 5; done
ss -ltn | grep ":8000" || { echo SERVER_FAILED; exit 1; }
echo SERVER_UP
cd /root
OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 OMNIGIBSON_HEADLESS=1 \
  /root/miniconda3/envs/behavior/bin/python -u /root/probe_action_dist.py \
  > /root/commit_probe_day0.log 2>&1
echo PROBE_RC=$?
grep "PROBE2" /root/commit_probe_day0.log | tail -8
echo DAY0_COMMIT_PROBE_DONE

#!/bin/bash
# Bring up the policy server for one Run-3 arm on port 8000 (the server block of run3_eval_arm.sh, standalone).
# Usage: bash serve_arm.sh <arm>   env POLICY_CONFIG (pi05_radio_run2) HISTORY_MODE (normal) SERVE_FORWARD_MAP_TOKENS (0)
#        EXTRA_SERVE_ARGS (e.g. "--action-horizon 32")
set -u
ARM=${1:?arm}; PORT=8000
PYO=/root/miniconda3/envs/openpi/bin/python; [ -x $PYO ] || PYO=/root/openpi_fork/.venv/bin/python
POLICY_CONFIG=${POLICY_CONFIG:-pi05_radio_run2}; HISTORY_MODE=${HISTORY_MODE:-normal}; SERVE_FORWARD_MAP_TOKENS=${SERVE_FORWARD_MAP_TOKENS:-0}
EXTRA_SERVE_ARGS=${EXTRA_SERVE_ARGS:-}
CK=/root/ckpt_$ARM
say(){ echo "[serve_arm $(date -u +%m-%dT%H:%M:%S)] $*"; }
[ -d $CK/params ] || { say "NO_CKPT $CK"; exit 1; }
for P in $(pgrep -f "serve_b1k.py.*--port $PORT"); do say "killing stale server $P"; kill $P; done
sleep 5
cd /root/openpi_fork
setsid nohup env XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 HISTORY_MODE=$HISTORY_MODE SERVE_FORWARD_MAP_TOKENS=$SERVE_FORWARD_MAP_TOKENS \
  $PYO scripts/b1k/serve_b1k.py --policy.config $POLICY_CONFIG --policy.dir $CK \
  --robot b1k/R1Pro --task b1k/turning_on_radio --repo-id b1k_radio --port $PORT $EXTRA_SERVE_ARGS \
  > /root/serve_$ARM.log 2>&1 < /dev/null &
port_open(){ (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; }
for i in $(seq 1 90); do port_open && break; sleep 5; done
port_open || { say "SERVER_FAILED_TO_START"; tail -20 /root/serve_$ARM.log; exit 1; }
say "SERVER_UP $ARM cfg=$POLICY_CONFIG hist=$HISTORY_MODE map=$SERVE_FORWARD_MAP_TOKENS extra='$EXTRA_SERVE_ARGS'"

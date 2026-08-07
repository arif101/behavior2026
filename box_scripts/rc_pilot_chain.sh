#!/bin/bash
# Reverse-curriculum pilot chain: ONE demo per Isaac process (env-reuse leaks: viewer-width
# assert on 2nd create_from_hdf5 in-process — same class of leak we chunk around on LIBERO).
# Usage: bash rc_pilot_chain.sh "10 20" <attempts> <offset> <horizon>
set -u
DEMOS=${1:-"10 20"}
ATTEMPTS=${2:-2}
OFFSET=${3:-150}
HORIZON=${4:-600}

cd /root/bw/BEHAVIOR-1K
for D in $DEMOS; do
  echo "=== RC_CHAIN demo $D ==="
  env PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 XDG_RUNTIME_DIR=/tmp/xdg \
    /root/miniconda3/envs/behavior/bin/python /root/reverse_curriculum_collect.py \
    --demo-id "$D" --attempts "$ATTEMPTS" --offset "$OFFSET" --horizon "$HORIZON"
  echo "=== RC_CHAIN demo $D exit=$? ==="
done

python3 - <<'PYEOF'
import json
runs = [json.loads(l) for l in open("/root/rc_stats.jsonl")]
n = len(runs)
s = sum(1 for r in runs if r["success"])
print(f"RC_CHAIN_DONE attempts={n} successes={s} rate={s/max(n,1):.2f}")
PYEOF

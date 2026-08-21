#!/bin/bash
# V2 D4: build validated snapshot bank for ALL 40 bank demos (held-out included — their
# snapshots serve sim-gate eval; training exclusion is enforced by the trainer, not here).
# ~6-9 min/demo render-off => ~5-6 h total. d30/d50 get the extended-retry adjudication.
set -x
DEMOS=$(python3 -c "
import json
b = json.load(open('/root/skill_start_bank.json'))
print(' '.join(map(str, sorted({e['demo'] for e in b['entries']}))))")
echo "SNAPBANK_DEMOS $DEMOS"
for d in $DEMOS; do
  PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 OMNIGIBSON_HEADLESS=1 \
    /root/miniconda3/envs/behavior/bin/python -u /root/build_snapshot_bank.py \
    --demo-id "$d" --retries 6 >> /root/snapshot_bank.log 2>&1
  echo "SNAPBANK_DEMO_DONE d=$d rc=$?"
done
echo SNAPSHOT_BANK_DONE

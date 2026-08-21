#!/bin/bash
# V2.1 snapshot bank: new start bank (press-50 + grasp anchors) + grip-fidelity restores.
# Runs AFTER the training run truncates (needs the sim). ~6-8 h for 40 demos x 6 entries.
set -x
DEMOS=$(python3 -c "
import json
b = json.load(open('/root/skill_start_bank_v21.json'))
print(' '.join(map(str, sorted({e['demo'] for e in b['entries']}))))")
for d in $DEMOS; do
  PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 OMNIGIBSON_HEADLESS=1 \
    /root/miniconda3/envs/behavior/bin/python -u /root/build_snapshot_bank.py \
    --demo-id "$d" --retries 6 --bank /root/skill_start_bank_v21.json \
    --out-dir /root/snapshot_bank_v21 --grip-fidelity \
    >> /root/snapshot_bank_v21.log 2>&1
  echo "V21_BANK_DEMO_DONE d=$d rc=$?"
done
echo SNAPSHOT_BANK_V21_DONE

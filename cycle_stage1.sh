#!/bin/bash
# Stage-1 curriculum pass: one train_skill chunk per train demo (Isaac env-reuse leaks ->
# process per demo), each resuming the newest checkpoint. Holdout demos are refused by
# train_skill's assert. Chunks stop early at rolling20 >= 0.95 or 3000 env steps.
set -x
CK=/root/skill_ckpts
DEMOS=$(python3 -c "
import json
b = json.load(open('/root/skill_start_bank.json'))
ho = set(b['held_out_demos'])
ds = sorted({e['demo'] for e in b['entries'] if e['stage'] == 1 and e['demo'] not in ho})
print(' '.join(map(str, ds)))")
echo "STAGE1_DEMOS $DEMOS"
for d in $DEMOS; do
  LATEST=$(ls -t $CK/*.pt 2>/dev/null | head -1)
  PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 OMNIGIBSON_HEADLESS=1 \
    /root/miniconda3/envs/behavior/bin/python -u /root/train_skill.py \
    --demo-id "$d" --stage 1 ${LATEST:+--resume "$LATEST"} \
    --max-env-steps 3000 --stop-at 0.95
  echo "CHUNK_DONE demo=$d rc=$?"
done
echo STAGE1_PASS_DONE

#!/bin/bash
# Curriculum pass driver: STAGE env selects the rung (bank stages 0=press-5, 1=~-10,
# 2=~-25, 3=~-100; stages >=1 quasi-static-anchored). One train_skill chunk per train
# demo, resuming the newest checkpoint. Usage: STAGE=1 bash cycle_stage.sh
set -x
STAGE=${STAGE:-1}
CK=/root/skill_ckpts
DEMOS=$(python3 -c "
import json
b = json.load(open('/root/skill_start_bank.json'))
ho = set(b['held_out_demos'])
ds = sorted({e['demo'] for e in b['entries'] if e['stage'] == $STAGE and e['demo'] not in ho})
print(' '.join(map(str, ds)))")
echo "STAGE${STAGE}_DEMOS $DEMOS"
for d in $DEMOS; do
  if [ -n "$SKIP_UNTIL" ] && [ "$d" -le "$SKIP_UNTIL" ]; then continue; fi
  LATEST=$(ls -t $CK/*.pt 2>/dev/null | head -1)
  PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 OMNIGIBSON_HEADLESS=1 \
    /root/miniconda3/envs/behavior/bin/python -u /root/train_skill.py \
    --demo-id "$d" --stage "$STAGE" ${LATEST:+--resume "$LATEST"} \
    --max-env-steps 4500 --max-episodes 25 --stop-at 0.95
  echo "CHUNK_DONE stage=$STAGE demo=$d rc=$?"
done
echo "STAGE${STAGE}_PASS_DONE"

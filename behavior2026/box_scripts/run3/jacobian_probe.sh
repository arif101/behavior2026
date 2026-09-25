#!/bin/bash
# CORRECTIVE-FIELD MEASUREMENT (sim box, 2026-09-21). full ckpt served with the v2 stack.
#  1. harvest near-grasp states (right EE within 0.25 m of the button) on train layouts 0..7 -> /root/near_states/near_tr<k>.npz
#  2. for up to 4 states: 15 conditions x 3 samples, one executed chunk (16 steps) each -> /root/jacobian_probe/<tag>.jsonl
# Usage: setsid nohup bash /root/jacobian_probe.sh > /root/jacobian_probe.out 2>&1 &
set -u
say(){ echo "[jac $(date -u +%m-%dT%H:%M:%S)] $*"; }
PYB=/root/miniconda3/envs/behavior/bin/python; S=/root/near_states; O=/root/jacobian_probe; mkdir -p $S $O
POLICY_CONFIG=pi05_radio_full SERVE_FORWARD_MAP_TOKENS=1 HISTORY_MODE=normal bash /root/serve_arm.sh full || { say SERVER_FAILED; exit 1; }
for k in 0 1 2 3 4 5 6 7; do
  [ $(ls $S/near_tr*.npz 2>/dev/null | wc -l) -ge 4 ] && break
  say "HARVEST near_tr$k"
  FREEZE_OUT=$S FREEZE_TAG=near_tr$k FREEZE_NEAR_R=0.25 FREEZE_CAP=1550 bash /root/freeze_rollout.sh behavior2026_eval.freeze_harvest.FreezeHarvestV2Wrapper train $k /root/freeze_diag/near_tr$k 1600 | tee -a $O/harvest.log
  [ -f $S/near_tr$k.json ] && say "  -> $(python3 -c "import json; d=json.load(open('$S/near_tr$k.json')); print('near=%s stationary=%s step=%s' % (d.get('near'), d.get('stationary'), d.get('step')))")"
done
say "harvest done: $(ls $S/near_tr*.npz | wc -l) states"
for f in $(ls $S/near_tr*.npz | head -4); do
  tag=$(basename $f .npz); k=${tag#near_tr}
  say "PROBE $tag (45 rollouts)"
  cd /root/bw/BEHAVIOR-1K
  env PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 XDG_RUNTIME_DIR=/tmp/xdg OMNIGIBSON_HEADLESS=1 MAP_ARM=B B1K_TASK_TARGETS=/root/task_targets.json B1K_TASK_NAME=turning_on_radio \
    JAC_STATE=$f JAC_TAG=$tag JAC_OUT=$O JAC_STEPS=16 JAC_SAMPLES=3 \
    $PYB -m omnigibson.eval.eval --task-name turning_on_radio --host 127.0.0.1 --port 8000 \
    --env-wrapper behavior2026_eval.jacobian_probe.JacobianProbeWrapper --instance-indices $k --num-rollouts 45 --mode train \
    --output-dir $O/eval_$tag --max-steps 6000 --headless > $O/eval_$tag.log 2>&1
  say "  -> $(grep -c JAC_RESULT $O/eval_$tag.log) results; tracebacks: $(grep -c Traceback $O/eval_$tag.log)"
done
say "JAC_DONE"

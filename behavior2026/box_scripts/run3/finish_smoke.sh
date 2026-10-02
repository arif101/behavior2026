#!/bin/bash
# SMOKE of the finish factory on ONE stall state (h4d_tr0) with 2 specs: unperturbed (00) + radio +4 cm lateral / hand +2 cm up (01).
# Runs beside the harvest (one extra Isaac process, no policy server: --policy local). ~10 min.
set -u
say(){ echo "[finish_smoke $(date -u +%m-%dT%H:%M:%S)] $*"; }
W=behavior2026_eval.finish_factory.FinishFactoryWrapper
mkdir -p /root/finish_smoke
FF_STATE=/root/stall_states/h4d_tr0.npz FF_ID=0 FF_PERTURBS="0,0,0,0,0,0,00;0.04,0,0,0,0.02,0,01" AFF_TAU=2 \
  bash /root/freeze_rollout.sh $W train 0 /root/finish_smoke/tr0 50 --policy local
grep -hE "OBJPERTURB|HANDPERTURB|PRE |RETREAT|ORIENT ok|STAGE ok|APPROACH ok|NATIVE_AG|VERIFIED_WELD|WELD_FAILED|LIFT|OBS_SAVED|FINISH_|Traceback|Error" /root/finish_smoke/tr0/eval.log | tail -30 | cut -c1-200
say "FINISH_SMOKE_DONE"

#!/bin/bash
# ORACLE-POINTER DIAGNOSTIC for the 4dall FINAL params (2026-10-01, operator: "replace stages 2 and 3 with the oracle test").
# Lets stage 1 of eval_4dall_chain.sh (final params, pointer off, 25 rollouts) finish untouched, then runs the 09-17 P1
# convention: StageV2AffordanceWrapper, map tokens forwarded, HISTORY_MODE=normal, tracker stage (exact pointer -> exact
# tracker), DIAG_ORACLE_POINT=1 = the exact sim-state target injected on every step. DIAGNOSTIC ONLY (privileged read).
# Decides perception vs execution: grasps with the oracle -> the trained finish exists and the gap is the visual read;
# 0 with the oracle -> the finish motion itself is missing (full ckpt 09-18: oracle 0/10, A4 oracle 0/10).
# Usage: ( setsid nohup bash /root/oracle_followup.sh > /root/oracle_followup.out 2>&1 < /dev/null & )
set -u
say(){ echo "[oracle $(date -u +%m-%dT%H:%M:%S)] $*"; }
summarize(){ local arm=$1 tag=$2; local L=/root/run3_eval_$arm.log; local N G S
  N=$(grep -c "arm=$arm$tag " $L); G=$(grep -c "arm=$arm$tag .*grasp=True" $L); S=$(grep -c "arm=$arm$tag .*success=True" $L)
  mkdir -p /root/run3_eval/$arm$tag; { echo "EVAL_SUMMARY arm=$arm$tag rollouts=$N grasp=$G success=$S ($(date -u +%m-%dT%H:%M))"; grep "arm=$arm$tag " $L; } | tee /root/run3_eval/$arm$tag/SUMMARY | head -1; }
stop_server(){ for P in $(pgrep -f "^/root/openpi_fork/.venv/bin/python scripts/b1k/serve_b1k.py"); do kill $P; done; sleep 5; }
# 1. retire the chain driver (stages 2/3 cancelled) and the held-out factory waiter (re-armed later behind the finish work);
#    the stage-1 eval (bash /root/run3_eval_arm.sh 4dall_final 25) is NOT touched
for P in $(pgrep -f "^bash /root/eval_4dall_chain.sh"); do kill $P; done
for P in $(pgrep -f "^bash /root/odart_heldout.sh"); do kill $P; done
sleep 2; say "chain driver procs left: $(pgrep -cf '^bash /root/eval_4dall_chain.sh'); heldout waiter: $(pgrep -cf '^bash /root/odart_heldout.sh'); stage-1 eval procs: $(pgrep -cf '^bash /root/run3_eval_arm.sh 4dall_final 25')"
# 2. wait for stage 1 to finish on its own
until [ "$(pgrep -cf '^bash /root/run3_eval_arm.sh 4dall_final 25')" = "0" ]; do sleep 60; done
say "stage 1 finished: $(summarize 4dall_final _ptroff)"
stop_server
# 3. oracle pointer, n=10
rm -f /root/stage_head_log.jsonl
say "=== ORACLE: 4dall_final, DIAG_ORACLE_POINT=1, P1 convention, 10 rollouts on instance 301 ==="
env POLICY_CONFIG=pi05_radio_4d_all WRAP=behavior2026_eval.stage_v2_wrapper.StageV2AffordanceWrapper SERVE_FORWARD_MAP_TOKENS=1 HISTORY_MODE=normal DIAG_ORACLE_POINT=1 TAG=_oracle \
  bash /root/run3_eval_arm.sh 4dall_final 10 > /root/run3_eval_4dall_final_oracle_driver.out 2>&1
stop_server
say "oracle: $(summarize 4dall_final _oracle)"
say "ORACLE_DONE"

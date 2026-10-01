#!/bin/bash
# n=25 EVAL CHAIN for the all-corrective arm on the frozen held-out instance 301 (2026-10-01, operator go-ahead).
# Serving = the pre-registered 4D convention (RUN3_CONFIG 09-24/25): GripperReopenWrapper (-> StageV2AffordanceWrapper ->
# AffordanceMapFullRes), SERVE_STAGE_SOURCE=head + REOPEN_STAGE_SOURCE=head (System-2 stage head drives stage_v2/progress/
# target_points_v2 and the reopen rule), map tokens forwarded, HISTORY_MODE=normal, POLICY_CONFIG=pi05_radio_4d_all.
# Order: (1) FINAL params, pointer OFF (AFF_TAU=2: the affordance head never injects a point) -> (2) if 0 grasps: FINAL params,
# pointer ON (AFF_TAU default 0.5) -> (3) if still 0: ckpt-5000 params, pointer OFF. Reference: A4 2/25 grasps.
# Usage: ( setsid nohup bash /root/eval_4dall_chain.sh > /root/eval_4dall_chain.out 2>&1 < /dev/null & )
set -u
say(){ echo "[eval4dall $(date -u +%m-%dT%H:%M:%S)] $*"; }
export POLICY_CONFIG=pi05_radio_4d_all WRAP=behavior2026_eval.gripper_reopen.GripperReopenWrapper SERVE_FORWARD_MAP_TOKENS=1 HISTORY_MODE=normal SERVE_STAGE_SOURCE=head REOPEN_STAGE_SOURCE=head
summarize(){ local arm=$1 tag=$2; local L=/root/run3_eval_$arm.log; local N G S
  N=$(grep -c "arm=$arm$tag " $L); G=$(grep -c "arm=$arm$tag .*grasp=True" $L); S=$(grep -c "arm=$arm$tag .*success=True" $L)
  mkdir -p /root/run3_eval/$arm$tag; { echo "EVAL_SUMMARY arm=$arm$tag rollouts=$N grasp=$G success=$S ($(date -u +%m-%dT%H:%M))"; grep "arm=$arm$tag " $L; } | tee /root/run3_eval/$arm$tag/SUMMARY
  echo $G; }
stop_server(){ for P in $(pgrep -f "^/root/openpi_fork/.venv/bin/python scripts/b1k/serve_b1k.py"); do kill $P; done; sleep 5; }
run_eval(){ local arm=$1 tag=$2 tau=$3; rm -f /root/stage_head_log.jsonl
  say "=== $arm$tag: AFF_TAU=$tau, 25 rollouts on instance 301 ==="
  AFF_TAU=$tau TAG=$tag bash /root/run3_eval_arm.sh $arm 25 > /root/run3_eval_${arm}${tag}_driver.out 2>&1
  stop_server; summarize $arm $tag | tail -1; }
[ -d /root/ckpt_4dall_final/params ] && [ -f /root/ckpt_4dall_final/assets/b1k_radio/norm_stats.json ] || { say "NO_FINAL_CKPT"; exit 1; }
[ -d /root/ckpt_4dall/params ] || { say "NO_5000_CKPT"; exit 1; }
G1=$(run_eval 4dall_final _ptroff 2 | tail -1); say "STAGE1 final/pointer-off grasps=$G1"
if [ "$G1" = "0" ]; then G2=$(run_eval 4dall_final _pton 0.5 | tail -1); say "STAGE2 final/pointer-on grasps=$G2"
  if [ "$G2" = "0" ]; then G3=$(run_eval 4dall _ptroff 2 | tail -1); say "STAGE3 ckpt5000/pointer-off grasps=$G3"; fi; fi
say "EVAL_4DALL_CHAIN_DONE"

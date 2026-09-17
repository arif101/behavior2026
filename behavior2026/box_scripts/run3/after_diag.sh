#!/bin/bash
# ROOT-CAUSE PROBES (sim box, 2026-09-17), run after freeze_diag.sh prints CHAIN_DONE. Serving-only; DIAGNOSTIC arms.
#  P1 oracle pointer, full ckpt   (DIAG_ORACLE_POINT=1: exact target from sim state -> ceiling if the pointer were perfect)
#  P2 pointer OFF, full ckpt      (AFF_TAU=2: null embedding always; in-distribution via 20 % modality dropout)
#  P3 oracle pointer, A4 ckpt     (same probe on the pre-registered reference)
#  P4 A4 with map tokens forwarded (SERVE_FORWARD_MAP_TOKENS=1: was the A-arm freeze/reach profile a serving artefact?)
# n=10 each (~3.5 h each). Usage: setsid nohup bash /root/after_diag.sh > /root/after_diag.out 2>&1 &
set -u
say(){ echo "[probes $(date -u +%m-%dT%H:%M:%S)] $*"; }
until grep -q CHAIN_DONE /root/freeze_diag.out 2>/dev/null; do sleep 300; done
say "freeze diagnostic chain done -> root-cause probes"
FULL="POLICY_CONFIG=pi05_radio_full WRAP=behavior2026_eval.stage_v2_wrapper.StageV2AffordanceWrapper SERVE_FORWARD_MAP_TOKENS=1 HISTORY_MODE=normal"
tally(){ echo "PROBE_SUMMARY $1: grasp $(grep -c "arm=$1 .*grasp=True" $2) / $(grep -c "arm=$1 " $2) success $(grep -c "arm=$1 .*success=True" $2)"; }
say "P1 oracle pointer, full ckpt, n=10"
env $FULL DIAG_ORACLE_POINT=1 TAG=_oracle bash /root/run3_eval_arm.sh full 10 > /root/run3_eval_full_oracle_driver.out 2>&1
tally full_oracle /root/run3_eval_full.log
say "P2 pointer off, full ckpt, n=10"
env $FULL AFF_TAU=2.0 TAG=_ptoff bash /root/run3_eval_arm.sh full 10 > /root/run3_eval_full_ptoff_driver.out 2>&1
tally full_ptoff /root/run3_eval_full.log
say "P3 oracle pointer, A4 ckpt, n=10 (parity serving otherwise)"
env SERVE_FORWARD_MAP_TOKENS=0 DIAG_ORACLE_POINT=1 TAG=_oracle bash /root/run3_eval_arm.sh a4 10 > /root/run3_eval_a4_oracle_driver.out 2>&1
tally a4_oracle /root/run3_eval_a4.log
say "P4 A4 with map tokens forwarded, n=10"
env SERVE_FORWARD_MAP_TOKENS=1 TAG=_map bash /root/run3_eval_arm.sh a4 10 > /root/run3_eval_a4_map_driver.out 2>&1
tally a4_map /root/run3_eval_a4.log
say "PROBES_DONE"

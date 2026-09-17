#!/bin/bash
# FREEZE DIAGNOSTIC CHAIN (RTX sim box, 2026-09-17). Replaces chain_evals_0917.sh from the S1 record eval onward.
#  0. wait for the S1 record eval (run3_eval_arm.sh a5 20 6) to exit; write a5 SUMMARY.
#  1. SMOKE without a server (--policy local = zero actions): harvest wrapper must dump a state, continue wrapper must
#     restore + orient. On failure: DIAG_SMOKE_FAILED, skip to the full-stack eval.
#  2. serve S1 (parity: SERVE_FORWARD_MAP_TOKENS=0), HARVEST freeze states: public_test idx 0 (=301, diagnostic only,
#     never a training seed) + train ids 0..5, step cap 1600.
#  3. CONTINUE from every harvested state: FREEZE_ORIENT=1 (wrist to canonical grasp attitude + 0.25 m advance, mirrors
#     S1 run 7's own escape) and FREEZE_ORIENT=0 (control: restore only). Readout = grasp/weld + minL/minR + success.
#  4. POINTS-OFF: S1 on 301 with AFF_TAU=2 (never inject target points) x2, harvest wrapper -> does the freeze persist?
#  5. COMMIT-TO-CHUNK: S1 on 301 with --action-horizon 32 x3 (serving knob only).
#  YIELD: before every rollout, if the full-stack params are on HF and its eval has not run, run it (n=25, full serving
#  stack) + SUMMARY, then re-serve S1 and resume. After everything: full eval (if still pending), then HISTORY_MODE=off x10.
# Usage: setsid nohup bash /root/freeze_diag.sh > /root/freeze_diag.out 2>&1 &
set -u
say(){ echo "[diag $(date -u +%m-%dT%H:%M:%S)] $*"; }
PYB=/root/miniconda3/envs/behavior/bin/python
D=/root/freeze_diag; S=/root/freeze_states; mkdir -p $D $S
HARV=behavior2026_eval.freeze_harvest.FreezeHarvestWrapper
CONT=behavior2026_eval.freeze_continue.FreezeContinueWrapper
summarize(){ local G N S_; G=$(grep -c "grasp=True" "$2"); N=$(grep -c "RUN3_EVAL_RESULT" "$2"); S_=$(grep -c "success=True" "$2")
  { echo "EVAL_SUMMARY arm=$1 rollouts=$N grasp=$G success=$S_ ($(date -u +%m-%dT%H:%M))"; grep "RUN3_EVAL_RESULT" "$2"; } | tee /root/run3_eval/$1/SUMMARY; }
full_on_hf(){ /root/openpi_fork/.venv/bin/python - <<'PY' 2>/dev/null | grep -q FULL_ON_HF
from huggingface_hub import HfApi
tok = open("/root/.hf_token").read().strip()
fs = HfApi(token=tok).list_repo_files("arif101/b26-run3-params", repo_type="model")
if any(f.startswith("full/provenance/") for f in fs) and any(f.startswith("full/params/") for f in fs) and any(f.startswith("full/assets/") for f in fs): print("FULL_ON_HF")
PY
}
run_full_eval(){
  say "full-stack params on HF -> n=25 eval, full serving stack"
  mkdir -p /root/run3_eval/full
  POLICY_CONFIG=pi05_radio_full WRAP=behavior2026_eval.stage_v2_wrapper.StageV2AffordanceWrapper SERVE_FORWARD_MAP_TOKENS=1 HISTORY_MODE=normal \
    bash /root/run3_eval_arm.sh full 25 > /root/run3_eval_full_driver.out 2>&1
  summarize full /root/run3_eval_full.log
  touch $D/FULL_DONE
  say "FULL_EVAL_DONE grasp $(grep -c grasp=True /root/run3_eval_full.log) / $(grep -c RUN3_EVAL_RESULT /root/run3_eval_full.log) (A4 reference 2/25)"
}
serve_s1(){ SERVE_FORWARD_MAP_TOKENS=0 EXTRA_SERVE_ARGS="${1:-}" bash /root/serve_arm.sh a5 || { say "S1_SERVER_FAILED"; return 1; }; }
check_yield(){ # $1 = server args to restore afterwards
  if [ ! -f $D/FULL_DONE ] && full_on_hf; then run_full_eval; serve_s1 "${1:-}" || exit 1; fi
}

# ---- 0. S1 record eval ---------------------------------------------------------------------------------------------
while pgrep -f "run3_eval_arm.sh a5 20 6" >/dev/null; do sleep 60; done
summarize a5 /root/run3_eval_a5.log
say "S1 record complete: grasp $(grep -c grasp=True /root/run3_eval_a5.log) / $(grep -c RUN3_EVAL_RESULT /root/run3_eval_a5.log)"
for P in $(pgrep -f "serve_b1k.py.*--port 8000"); do say "stopping S1 server $P for the smoke"; kill $P; done; sleep 5

# ---- 1. smoke (no server) -------------------------------------------------------------------------------------------
say "SMOKE harvest (zero-action policy)"
FREEZE_OUT=$S FREEZE_TAG=smoke_tr0 FREEZE_MIN=150 FREEZE_WIN=60 bash /root/freeze_rollout.sh $HARV train 0 $D/smoke_harvest 700 --policy local | tee -a $D/smoke.log
if [ ! -f $S/smoke_tr0.npz ]; then say "DIAG_SMOKE_FAILED (no harvest npz)"; SMOKE_OK=0; else
  say "SMOKE continue+orient"
  FREEZE_STATE=$S/smoke_tr0.npz FREEZE_ORIENT=1 bash /root/freeze_rollout.sh $CONT train 0 $D/smoke_continue 60 --policy local | tee -a $D/smoke.log
  if grep -q FREEZE_CONTINUE_READY $D/smoke.log; then SMOKE_OK=1; say "DIAG_SMOKE_OK"; else SMOKE_OK=0; say "DIAG_SMOKE_FAILED (continue)"; fi
fi

if [ "$SMOKE_OK" = "1" ]; then
  serve_s1 || exit 1
  # ---- 2. harvest --------------------------------------------------------------------------------------------------
  for spec in "public_test 0 pt301" "train 0 tr0" "train 1 tr1" "train 2 tr2" "train 3 tr3" "train 4 tr4" "train 5 tr5"; do
    set -- $spec; check_yield
    say "HARVEST $3 ($1 idx $2)"
    FREEZE_OUT=$S FREEZE_TAG=$3 bash /root/freeze_rollout.sh $HARV $1 $2 $D/harvest_$3 1600 | tee -a $D/harvest.log
    [ -f $S/$3.npz ] || echo "NO_FREEZE $3" | tee -a $D/harvest.log
  done
  say "HARVEST done: $(ls $S/*.npz 2>/dev/null | grep -v smoke | wc -l) freeze states; $(grep -c NO_FREEZE $D/harvest.log) no-freeze"
  # ---- 3. continue-from-freeze ------------------------------------------------------------------------------------
  for f in $(ls $S/*.npz | grep -v smoke); do
    tag=$(basename $f .npz); mode=train; idx=${tag#tr}; [ "$tag" = "pt301" ] && { mode=public_test; idx=0; }
    for orient in 1 0; do
      check_yield
      say "CONTINUE $tag orient=$orient"
      FREEZE_STATE=$f FREEZE_ORIENT=$orient FREEZE_ADVANCE=0.25 bash /root/freeze_rollout.sh $CONT $mode $idx $D/continue_${tag}_o$orient | tee -a $D/continue.log
    done
  done
  say "CONTINUE done: orient=1 grasps $(grep DIAG_RESULT $D/continue.log | grep -c "_o1 .*grasp=True") / $(grep DIAG_RESULT $D/continue.log | grep -c "_o1 "); orient=0 grasps $(grep DIAG_RESULT $D/continue.log | grep -c "_o0 .*grasp=True") / $(grep DIAG_RESULT $D/continue.log | grep -c "_o0 ")"
  # ---- 4. points-off ----------------------------------------------------------------------------------------------
  for i in 1 2; do check_yield; say "POINTS_OFF 301 #$i"
    AFF_TAU=2.0 FREEZE_OUT=$S FREEZE_TAG=ptoff_$i bash /root/freeze_rollout.sh $HARV public_test 0 $D/pointsoff_$i 1600 | tee -a $D/pointsoff.log
    [ -f $S/ptoff_$i.npz ] || echo "NO_FREEZE ptoff_$i" | tee -a $D/pointsoff.log
  done
  # ---- 5. commit-to-chunk ------------------------------------------------------------------------------------------
  serve_s1 "--action-horizon 32" || exit 1
  for i in 1 2 3; do check_yield "--action-horizon 32"; say "COMMIT32 301 #$i"
    FREEZE_OUT=$S FREEZE_TAG=commit32_$i bash /root/freeze_rollout.sh $HARV public_test 0 $D/commit32_$i 1600 | tee -a $D/commit32.log
    [ -f $S/commit32_$i.npz ] || echo "NO_FREEZE commit32_$i" | tee -a $D/commit32.log
  done
  say "DIAG_DONE"
fi

# ---- full-stack eval (if not yet run) + history-off liveness arm ---------------------------------------------------
if [ ! -f $D/FULL_DONE ]; then until full_on_hf; do sleep 300; done; run_full_eval; fi
say "HISTORY_MODE=off x10 on the full checkpoint"
POLICY_CONFIG=pi05_radio_full WRAP=behavior2026_eval.stage_v2_wrapper.StageV2AffordanceWrapper SERVE_FORWARD_MAP_TOKENS=1 HISTORY_MODE=off TAG=_hist_off \
  bash /root/run3_eval_arm.sh full 10 > /root/run3_eval_full_hist_off_driver.out 2>&1
say "CHAIN_DONE hist_off grasp $(grep -c "arm=full_hist_off.*grasp=True" /root/run3_eval_full.log) / $(grep -c "arm=full_hist_off" /root/run3_eval_full.log)"

#!/bin/bash
# RTX sim box chain (2026-09-17): (1) wait for the A4 validity rollouts (run3_eval_arm.sh a4 5) to exit;
# (2) resume the S1 record eval a5 runs 6..25 (parity serving, SERVE_FORWARD_MAP_TOKENS=0) and rewrite its SUMMARY;
# (3) wait for the full-stack arm on HF (full/params + full/provenance, uploaded by the trainer's finalize) and run
#     its n=25 eval with the full serving stack (pi05_radio_full + StageV2AffordanceWrapper + map tokens forwarded +
#     HISTORY_MODE=normal), then SUMMARY. Usage: setsid nohup bash /root/chain_evals_0917.sh > /root/chain_evals_0917.out 2>&1 &
set -u
say(){ echo "[chain $(date -u +%m-%dT%H:%M:%S)] $*"; }
summarize(){ # $1 arm-dir (a5 / full), $2 log
  local G N S; G=$(grep -c "grasp=True" "$2"); N=$(grep -c "RUN3_EVAL_RESULT" "$2"); S=$(grep -c "success=True" "$2")
  { echo "EVAL_SUMMARY arm=$1 rollouts=$N grasp=$G success=$S ($(date -u +%m-%dT%H:%M))"; grep "RUN3_EVAL_RESULT" "$2"; } | tee /root/run3_eval/$1/SUMMARY
}
while pgrep -f "run3_eval_arm.sh a4 5" >/dev/null; do sleep 60; done
say "A4 validity finished: $(grep -c RUN3_EVAL_RESULT /root/run3_eval_a4.log) results, grasp=$(grep -c grasp=True /root/run3_eval_a4.log)"
say "resuming S1 (a5) runs 6..25, parity serving"
SERVE_FORWARD_MAP_TOKENS=0 bash /root/run3_eval_arm.sh a5 20 6 > /root/run3_eval_a5_driver2.out 2>&1
summarize a5 /root/run3_eval_a5.log
say "S1 record complete"
until /root/openpi_fork/.venv/bin/python - <<'PY' 2>/dev/null | grep -q FULL_ON_HF
from huggingface_hub import HfApi
tok = open("/root/.hf_token").read().strip()
fs = HfApi(token=tok).list_repo_files("arif101/b26-run3-params", repo_type="model")
if any(f.startswith("full/provenance/") for f in fs) and any(f.startswith("full/params/") for f in fs) and any(f.startswith("full/assets/") for f in fs): print("FULL_ON_HF")
PY
do sleep 300; done
say "full-stack params on HF -> n=25 eval, full serving stack"
mkdir -p /root/run3_eval/full
POLICY_CONFIG=pi05_radio_full WRAP=behavior2026_eval.stage_v2_wrapper.StageV2AffordanceWrapper SERVE_FORWARD_MAP_TOKENS=1 HISTORY_MODE=normal \
  bash /root/run3_eval_arm.sh full 25 > /root/run3_eval_full_driver.out 2>&1
summarize full /root/run3_eval_full.log
say "CHAIN_DONE full: grasp $(grep -c grasp=True /root/run3_eval_full.log) / $(grep -c RUN3_EVAL_RESULT /root/run3_eval_full.log) (A4 reference 2/25)"

#!/bin/bash
# RTX sim box: run the S1 (arm a5) n=25 eval automatically. Waits for (1) the approach-factory rerun to finish and
# (2) S1's params on HF (the trainer's finalize uploads a5/params, a5/assets, then a5/provenance/*). Then
# run3_eval_arm.sh a5 25 (parity serving: SERVE_FORWARD_MAP_TOKENS=0 like the A0-A4 evals) and a SUMMARY with the
# grasp-completion count (assisted-grasp weld telemetry).
# Usage: setsid nohup bash /root/chain_s1_eval.sh > /root/chain_s1_eval.out 2>&1 &
set -u
say(){ echo "[s1_eval $(date -u +%m-%dT%H:%M:%S)] $*"; }
until grep -q APPROACH_FACTORY_REBUILD_DONE /root/approach_rebuild.out 2>/dev/null; do sleep 120; done
say "rerun finished: renders=$(ls /root/factory_obs2 | wc -l)"
until /root/openpi_fork/.venv/bin/python - <<'PY' 2>/dev/null | grep -q A5_ON_HF
from huggingface_hub import HfApi
tok = open("/root/.hf_token").read().strip()
fs = HfApi(token=tok).list_repo_files("arif101/b26-run3-params", repo_type="model")
if any(f.startswith("a5/provenance/") for f in fs) and any(f.startswith("a5/params/") for f in fs): print("A5_ON_HF")
PY
do sleep 300; done
say "S1 params on HF — starting the n=25 eval (parity serving)"
SERVE_FORWARD_MAP_TOKENS=0 bash /root/run3_eval_arm.sh a5 25 > /root/run3_eval_a5_driver.out 2>&1
G=$(grep -c "grasp=True" /root/run3_eval_a5_driver.out); N=$(grep -c "RUN3_EVAL_RESULT" /root/run3_eval_a5_driver.out); S=$(grep -c "success=True" /root/run3_eval_a5_driver.out)
{ echo "S1_EVAL_SUMMARY arm=a5 rollouts=$N grasp=$G success=$S ($(date -u +%m-%dT%H:%M))"; grep "RUN3_EVAL_RESULT" /root/run3_eval_a5_driver.out; } | tee /root/run3_eval/a5/SUMMARY
say "done: grasp $G / $N (A4 reference 2/25)"

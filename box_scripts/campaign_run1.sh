#!/bin/bash
# RUN-1 EVAL CAMPAIGN (RUN1_EVAL_PROTOCOL.md + Amendment 1) — arms via MAP_ARM env.
# Usage: bash campaign_run1.sh <ARM> <N_RUNS> [START_IDX]
#   ARM in A|B|B0|C. Server must be up (serve_run1b.sh). Archives per run under
#   /root/campaign/<ARM>/run_N/: eval json, video, action log, wrapper stats (incl. I1/I4).
set -u
ARM=${1:?arm}; N=${2:?n_runs}; START=${3:-1}
WRAP=behavior2026_eval.affordance_map_fullres.AffordanceMapFullRes
[ "$ARM" = "C" ] && WRAP=behavior2026_eval.scaffold_collect.ScaffoldCollectWrapper

for i in $(seq $START $((START + N - 1))); do
  OUT=/root/campaign/$ARM/run_$i
  mkdir -p $OUT
  echo "=== ARM $ARM RUN $i ==="
  rm -f /root/action_log.jsonl /root/affordance_wrapper_stats.json /root/scaffold_stats.json
  rm -rf /root/map_live && mkdir -p /root/map_live
  cd /root/bw/BEHAVIOR-1K
  env PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 XDG_RUNTIME_DIR=/tmp/xdg OMNIGIBSON_HEADLESS=1 \
    MAP_ARM=$ARM \
    B1K_TASK_TARGETS=/root/task_targets.json B1K_TASK_NAME=turning_on_radio \
    /root/miniconda3/envs/behavior/bin/python -m omnigibson.eval.eval \
    --task-name turning_on_radio --host 127.0.0.1 --port 8000 \
    --env-wrapper $WRAP \
    --instance-indices 0 --num-rollouts 1 --mode public_test \
    --output-dir $OUT --write-video --headless > $OUT/eval.log 2>&1
  cp /root/action_log.jsonl $OUT/ 2>/dev/null
  cp /root/affordance_wrapper_stats.json $OUT/ 2>/dev/null
  cp /root/scaffold_stats.json $OUT/ 2>/dev/null
  cp -r /root/map_live $OUT/map_live 2>/dev/null
  /root/miniconda3/envs/behavior/bin/python - <<PY 2>/dev/null
import glob, json
js = glob.glob("$OUT/json/*.json")
st = {}
try:
    st = json.load(open("$OUT/affordance_wrapper_stats.json"))
except Exception:
    pass
if js:
    d = json.load(open(js[0]))
    print("CAMPAIGN_RESULT arm=$ARM run=$i success=%s q=%s steps=%s inject=%s/%s conf=%s "
          "wristL=%s maperr=%s afferr=%s" % (
          d["success"], d["q_score"]["final"], d["steps"],
          st.get("n_inject"), st.get("n_steps"), st.get("conf_p50"),
          st.get("wrist_angL_p50"), st.get("map_err_p50"), st.get("aff_err_p50")))
else:
    print("CAMPAIGN_RESULT arm=$ARM run=$i NO_JSON")
PY
done
echo "CAMPAIGN_ARM_${ARM}_DONE"

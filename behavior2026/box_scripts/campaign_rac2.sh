#!/bin/bash
# RaC v2 COLLECTION CAMPAIGN — arm-C scaffold engine + training-format obs capture.
# Usage: bash campaign_rac2.sh <N_RUNS> [START_IDX] [MAX_STEPS]
#   MAX_STEPS is for SMOKE runs only (real runs use the eval default ~3225-step timeout).
#   Smoke env knobs (set them on the command line, never in real collection):
#     RAC2_FORCE_KICK=<step> RAC2_ACCEPT_ANY=1 bash campaign_rac2.sh 1 900 400
# Server must be up (serve_run1b.sh, port 8000 — the same serving config arm C used).
# Clips -> /root/rac2_clips/rac_<epid>_<k>.npz ; per-episode stats -> /root/rac2_stats/.
set -u
N=${1:?n_runs}; START=${2:-1}; MAXSTEPS=${3:-}
WRAP=behavior2026_eval.scaffold_collect_v2.ScaffoldCollectV2Wrapper

MS_ARG=""
[ -n "$MAXSTEPS" ] && MS_ARG="--max-steps $MAXSTEPS"

for i in $(seq $START $((START + N - 1))); do
  FREE_G=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
  if [ "$FREE_G" -lt 30 ]; then
    echo "RAC2_ABORT run=$i: only ${FREE_G}G free disk (<30G guard)"; break
  fi
  OUT=/root/campaign_rac2/run_$i
  mkdir -p $OUT
  echo "=== RAC2 RUN $i ==="
  rm -f /root/action_log.jsonl /root/affordance_wrapper_stats.json /root/scaffold_stats.json
  rm -rf /root/map_live && mkdir -p /root/map_live
  cd /root/bw/BEHAVIOR-1K
  env PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 XDG_RUNTIME_DIR=/tmp/xdg OMNIGIBSON_HEADLESS=1 \
    MAP_ARM=C \
    RAC2_TASK_INSTANCE_ID=301 \
    B1K_TASK_TARGETS=/root/task_targets.json B1K_TASK_NAME=turning_on_radio \
    /root/miniconda3/envs/behavior/bin/python -m omnigibson.eval.eval \
    --task-name turning_on_radio --host 127.0.0.1 --port 8000 \
    --env-wrapper $WRAP \
    --instance-indices 0 --num-rollouts 1 --mode public_test $MS_ARG \
    --output-dir $OUT --write-video --headless > $OUT/eval.log 2>&1
  cp /root/action_log.jsonl $OUT/ 2>/dev/null
  cp /root/affordance_wrapper_stats.json $OUT/ 2>/dev/null
  cp /root/scaffold_stats.json $OUT/ 2>/dev/null
  /root/miniconda3/envs/behavior/bin/python - <<PY 2>/dev/null
import glob, json
js = glob.glob("$OUT/json/*.json")
st = {}
try:
    st = json.load(open("$OUT/scaffold_stats.json"))
except Exception:
    pass
if js:
    d = json.load(open(js[0]))
    print("RAC2_RESULT run=$i success=%s q=%s steps=%s interventions=%s clips_kept=%s" % (
          d["success"], d["q_score"]["final"], d["steps"],
          st.get("interventions"), st.get("clips_kept")))
else:
    print("RAC2_RESULT run=$i NO_JSON interventions=%s clips_kept=%s" % (
          st.get("interventions"), st.get("clips_kept")))
PY
done
echo "RAC2_CAMPAIGN_DONE n_clips=$(ls /root/rac2_clips 2>/dev/null | wc -l)"

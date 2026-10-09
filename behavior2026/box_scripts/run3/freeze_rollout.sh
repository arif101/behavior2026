#!/bin/bash
# One diagnostic rollout through the eval harness. Usage:
#   bash freeze_rollout.sh <wrapper-class-path> <mode train|public_test> <index> <outdir> [max_steps] [extra eval args]
# env passthrough: FREEZE_* (harvest/continue), AFF_TAU (affordance injection gate). Prints DIAG_RESULT + FREEZE_* lines.
set -u
WRAP=${1:?wrapper}; MODE=${2:?mode}; IDX=${3:?index}; OUT=${4:?outdir}; MAXS=${5:-}; shift 5 2>/dev/null || shift $#
EXTRA="$*"
PYB=/root/miniconda3/envs/behavior/bin/python
mkdir -p $OUT; rm -f /root/action_log.jsonl /root/affordance_wrapper_stats.json /root/freeze_continue_last.json
rm -rf /root/map_live && mkdir -p /root/map_live
cd /root/bw/BEHAVIOR-1K
MS=""; [ -n "$MAXS" ] && MS="--max-steps $MAXS"
env PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 XDG_RUNTIME_DIR=/tmp/xdg OMNIGIBSON_HEADLESS=1 \
  MAP_ARM=B B1K_TASK_TARGETS=/root/task_targets.json B1K_TASK_NAME=turning_on_radio \
  $PYB -m omnigibson.eval.eval --task-name turning_on_radio --host 127.0.0.1 --port 8000 \
  --env-wrapper $WRAP --instance-indices $IDX --num-rollouts 1 --mode $MODE \
  --output-dir $OUT --write-video --headless $MS $EXTRA > $OUT/eval.log 2>&1
cp /root/action_log.jsonl $OUT/ 2>/dev/null; cp /root/affordance_wrapper_stats.json $OUT/ 2>/dev/null; cp /root/freeze_continue_last.json $OUT/ 2>/dev/null
grep -hE "FREEZE_HARVESTED|FREEZE_DUMP_FAILED|FREEZE_CONTINUE_READY|FREEZE_CONTINUE_NO_STATE|ORIENT (reached|budget)" $OUT/eval.log | tail -4
grep -qE "Traceback" $OUT/eval.log && { echo "DIAG_TRACEBACK $OUT"; grep -B2 -A12 "Traceback" $OUT/eval.log | tail -16; }
$PYB - <<PY 2>/dev/null
import glob, json
js = glob.glob("$OUT/json/*.json"); st = {}
try: st = json.load(open("$OUT/affordance_wrapper_stats.json"))
except Exception: pass
dl = st.get("dist_L_series") or []; dr = st.get("dist_R_series") or []
if js:
    d = json.load(open(js[0]))
    print("DIAG_RESULT out=$OUT mode=$MODE idx=$IDX success=%s q=%s steps=%s grasp=%s weld=%s/%s inject=%s/%s conf=%s minL=%s minR=%s" % (
        d["success"], d["q_score"]["final"], d["steps"], st.get("grasp"), st.get("ag_weld_right_step"), st.get("ag_weld_left_step"),
        st.get("n_inject"), st.get("n_steps"), st.get("conf_p50"), st.get("dist_L_min"), st.get("dist_R_min")))
else:
    print("DIAG_RESULT out=$OUT mode=$MODE idx=$IDX NO_JSON")
PY

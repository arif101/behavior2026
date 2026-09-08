#!/bin/bash
# RUN-3 EVAL DRIVER (RT-core sim box) — serve one Run-3 arm through the Run-2 primary serving
# stack and run N rollouts on the frozen held-out instance. Assembled from the committed Run-2
# pieces: rate_legal.sh (serve line) + campaign_run1.sh (eval loop + archive).
#
# Usage:  bash run3_eval_arm.sh <arm a0..a5|run2ref> <N> [START_IDX]
# Prereqs (serve_run2.sh layout): /root/openpi_fork = tarball + fork_snapshot + 5 run-2 patches +
#   patch_b1k_robot_name; OmniGibson evaluator patched with patch_point_passthrough.py THEN
#   patch_map_passthrough2.py (+ patch_action_logger.py); /root/behavior2026_eval/ = eval/*.py with
#   patch_wrapper_arms.py applied; /root/foveated_map.py, /root/aff_out/, /root/task_targets.json,
#   /root/*.json keys present; /root/.hf_token.
# Serving convention (identical for EVERY arm; pre-registered):
#   wrapper  behavior2026_eval.affordance_map_fullres.AffordanceMapFullRes (LEGAL affordance-head
#            points -> AdaLN; online FoveatedMap -> map tokens; NO sim-state reads feed the policy)
#   MAP_ARM  B (live map tokens; matches the training-time observation convention — tokens present
#            with live content; the frozen prefix route contributes through map_alpha=0.036 exactly
#            as it did during training; map_geo AdaLN reads the same tokens live)
#   points   target_points = obj_base - ee_base (add_target_points.py convention; the wrapper
#            computes it from the affordance head + proprio EE, same frame)
#   config   pi05_radio_run2 (architecturally identical to pi05_radio_run3_a*; only model flags,
#            transforms and asset_id matter at serve time)  --repo-id b1k_radio (asset id of the
#            norm stats shipped in <arm>/assets/b1k_radio/norm_stats.json — Run-2's stats, md5
#            a6053883b29666925566ec02c433e562, identical across arms)
#   timeout  NO --max-steps (task default = 3,225 steps, as Run-2)
set -u
ARM=${1:?arm}; N=${2:?n_runs}; START=${3:-1}
PORT=8000
PYO=/root/miniconda3/envs/openpi/bin/python
PYB=/root/miniconda3/envs/behavior/bin/python
WRAP=behavior2026_eval.affordance_map_fullres.AffordanceMapFullRes
CK=/root/ckpt_$ARM
say(){ echo "[run3_eval $(date -u +%m-%dT%H:%M:%S)] $*"; }

# ---- checkpoint: <arm>/params + <arm>/assets/b1k_radio/norm_stats.json from HF ------------------
if [ ! -f $CK/assets/b1k_radio/norm_stats.json ] || [ ! -d $CK/params ]; then
  $PYO - <<PY
import os; os.environ["HF_XET_HIGH_PERFORMANCE"]="1"
from huggingface_hub import snapshot_download
tok=open("/root/.hf_token").read().strip()
if "$ARM" == "run2ref":
    snapshot_download("arif101/b26-run2-params", token=tok, local_dir="$CK", allow_patterns=["params/**","assets/**"])
else:
    snapshot_download("arif101/b26-run3-params", token=tok, local_dir="/root/run3_dl", allow_patterns=["$ARM/params/**","$ARM/assets/**"])
    import shutil, pathlib
    pathlib.Path("$CK").mkdir(exist_ok=True)
    for sub in ("params","assets"):
        dst=pathlib.Path("$CK")/sub
        if not dst.exists(): shutil.copytree(f"/root/run3_dl/$ARM/{sub}", dst)
print("CKPT_OK")
PY
fi
md5sum $CK/assets/b1k_radio/norm_stats.json | tee -a /root/run3_eval_$ARM.log

# ---- server (one per arm; kill stale by explicit pid) -------------------------------------------
for P in $(pgrep -f "serve_b1k.py.*--port $PORT"); do say "killing stale server $P"; kill $P; done
sleep 5
cd /root/openpi_fork
setsid nohup env XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 \
  $PYO scripts/b1k/serve_b1k.py \
  --policy.config pi05_radio_run2 --policy.dir $CK \
  --robot b1k/R1Pro --task b1k/turning_on_radio --repo-id b1k_radio --port $PORT \
  > /root/serve_$ARM.log 2>&1 < /dev/null &
for i in $(seq 1 90); do ss -ltn 2>/dev/null | grep -q ":$PORT" && break; sleep 5; done
ss -ltn | grep -q ":$PORT" || { say "SERVER_FAILED_TO_START"; tail -20 /root/serve_$ARM.log; exit 1; }
grep -m1 "Using norm stats for repo" /root/serve_$ARM.log | tee -a /root/run3_eval_$ARM.log
say "SERVER_UP $ARM"

# ---- rollouts (instance 301 = public_test index 0; one rollout per run dir, as Run-2) -----------
for i in $(seq $START $((START + N - 1))); do
  OUT=/root/run3_eval/$ARM/run_$i; mkdir -p $OUT
  say "=== ARM $ARM RUN $i ==="
  rm -f /root/action_log.jsonl /root/affordance_wrapper_stats.json
  rm -rf /root/map_live && mkdir -p /root/map_live
  cd /root/bw/BEHAVIOR-1K
  env PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 XDG_RUNTIME_DIR=/tmp/xdg OMNIGIBSON_HEADLESS=1 \
    MAP_ARM=B B1K_TASK_TARGETS=/root/task_targets.json B1K_TASK_NAME=turning_on_radio \
    $PYB -m omnigibson.eval.eval \
    --task-name turning_on_radio --host 127.0.0.1 --port $PORT \
    --env-wrapper $WRAP \
    --instance-indices 0 --num-rollouts 1 --mode public_test \
    --output-dir $OUT --write-video --headless > $OUT/eval.log 2>&1
  cp /root/action_log.jsonl $OUT/ 2>/dev/null; cp /root/affordance_wrapper_stats.json $OUT/ 2>/dev/null
  cp -r /root/map_live $OUT/map_live 2>/dev/null
  $PYB - <<PY 2>/dev/null
import glob, json
js = glob.glob("$OUT/json/*.json"); st = {}
try: st = json.load(open("$OUT/affordance_wrapper_stats.json"))
except Exception: pass
if js:
    d = json.load(open(js[0]))
    dl = st.get("dist_L_series") or []; dr = st.get("dist_R_series") or []
    print("RUN3_EVAL_RESULT arm=$ARM run=$i success=%s q=%s steps=%s inject=%s/%s conf=%s minL=%s minR=%s" % (
        d["success"], d["q_score"]["final"], d["steps"], st.get("n_inject"), st.get("n_steps"), st.get("conf_p50"),
        min(dl) if dl else None, min(dr) if dr else None))
else:
    print("RUN3_EVAL_RESULT arm=$ARM run=$i NO_JSON")
PY
done | tee -a /root/run3_eval_$ARM.log
say "RUN3_EVAL_ARM_${ARM}_DONE"

#!/bin/bash
# Rate estimation for the winning config (oracle points + execute-full-chunk-32), n=5.
#
# WHY: the q=1.0 is n=1, and BEHAVIOR-1K run-to-run variance is documented >27% on single tasks.
# One success is an existence proof; the leaderboard pays rates. Five sequential rollouts on
# instance 301 (the ONLY public_test instance for turning_on_radio — the harness resolved
# public_test to [301], so cross-instance generalization needs other tasks, not this one).
#
# PER-RUN ARCHIVING — the lesson from clearing the success run's action log before analyzing it:
# every rollout keeps its own action log, result JSON, video, and injection stats under
# /root/rate/run_N/. The action logs feed the JITTER ANALYSIS the user's observation motivates:
# the successful run was visibly jittery during grasp/button-press; the logs let us test whether
# manipulation-phase jitter aligns with the 32-step replan seams (-> winner-style generation-time
# seam inpainting is the fix) or lives within chunks (-> flow-sample mode switching, needs a
# different treatment).
#
# Waits for any current eval to finish before starting; restarts the server at h32 once.
set -u
cd /root

echo "waiting for any running eval..."
while pgrep -f "omnigibson.eval.eval" > /dev/null; do sleep 30; done

kill $(cat /root/serve.pid) 2>/dev/null; sleep 5
cd /root/openpi_fork
setsid nohup env XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 \
  /root/miniconda3/envs/openpi/bin/python scripts/b1k/serve_b1k.py \
  --policy.config pi05_radio_gate --policy.dir /root/ckpt \
  --robot b1k/R1Pro --task b1k/turning_on_radio --repo-id b1k_radio \
  --action-horizon 32 --port 8000 \
  > /root/serve_rate.log 2>&1 < /dev/null &
echo $! > /root/serve.pid
sleep 85

mkdir -p /root/rate
for N in 1 2 3 4 5; do
  echo "=== RATE RUN $N/5 ==="
  rm -f /root/action_log.jsonl /root/oracle_wrapper_stats.json
  mkdir -p /root/rate/run_$N
  cd /root/bw/BEHAVIOR-1K
  env PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 XDG_RUNTIME_DIR=/tmp/xdg OMNIGIBSON_HEADLESS=1 \
    B1K_TASK_TARGETS=/root/g3_pipeline/task_targets.json B1K_TASK_NAME=turning_on_radio \
    /root/miniconda3/envs/behavior/bin/python -m omnigibson.eval.eval \
    --task-name turning_on_radio --host 127.0.0.1 --port 8000 \
    --env-wrapper behavior2026_eval.oracle_point_fullres.OraclePointFullRes \
    --instance-indices 0 --num-rollouts 1 --mode public_test \
    --output-dir /root/rate/run_$N --write-video --headless \
    > /root/rate/run_$N/eval.log 2>&1
  cp /root/action_log.jsonl /root/rate/run_$N/ 2>/dev/null
  cp /root/oracle_wrapper_stats.json /root/rate/run_$N/ 2>/dev/null
  # one-line verdict per run so the summary is greppable
  /root/miniconda3/envs/openpi/bin/python - <<PY 2>/dev/null
import glob, json
js = glob.glob("/root/rate/run_$N/json/*.json")
if js:
    d = json.load(open(js[0]))
    print(f"RATE_RESULT run=$N success={d['success']} q={d['q_score']['final']} steps={d['steps']}")
else:
    print(f"RATE_RESULT run=$N NO_JSON (crashed?)")
PY
done
echo "RATE_RUNS_DONE"

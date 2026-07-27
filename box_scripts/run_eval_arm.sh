#!/bin/bash
# Closed-loop eval of ONE Phase-A arm over a fixed instance set.
# Usage: run_eval_arm.sh <arm> <step> <task> <n_instances> [gpu]
# The instance list and seeds are FIXED, so both arms are scored on identical episodes --
# that is what makes the A/B a comparison rather than two unrelated samples.
set -x
ARM=${1:-phaseA_weighted}; STEP=${2:-10000}; TASK=${3:-turning_on_radio}; N=${4:-10}; GPU=${5:-0}
CK=/root/phaseA_ckpts/$ARM/$STEP
OUT=/root/eval_${ARM}_${STEP}_${TASK}
PORT=8901

# one server per arm; kill any previous by explicit PID
for P in $(pgrep -f "serve_b1k.py.*--port $PORT"); do echo "killing stale server $P"; kill $P; done
sleep 5
cd /root/openpi_adaln
setsid nohup env CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/b1k/serve_b1k.py \
  --robot b1k/R1Pro --task b1k/$TASK --repo-id b1k_phaseA \
  --policy.config pi05_phaseA_point --policy.dir $CK --port $PORT \
  > /root/serve_${ARM}_${STEP}.log 2>&1 < /dev/null &
# wait for the port, not a fixed sleep
for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ":$PORT" && break; sleep 5; done
ss -ltn | grep -q ":$PORT" || { echo "SERVER_FAILED_TO_START"; exit 1; }
echo "SERVER_UP $ARM/$STEP"

IDX=$(seq 0 $((N-1)) | tr "\n" " ")
cd /workspace/BEHAVIOR-1K/OmniGibson
CUDA_VISIBLE_DEVICES=$GPU OMNIGIBSON_HEADLESS=1 /root/miniconda3/envs/behavior/bin/python \
  -m omnigibson.eval.eval --task-name $TASK --robot-config omnigibson/eval/r1pro.yaml \
  --mode public_test --host 127.0.0.1 --port $PORT --instance-indices $IDX \
  --max-steps 500 --output-dir $OUT --write-video
echo "EVAL_ARM_DONE $ARM $STEP rc=$?"
/root/miniconda3/envs/behavior/bin/python - "$OUT" <<'PY'
import glob, json, sys, statistics
fs = glob.glob(sys.argv[1] + "/**/*.json", recursive=True)
qs = []
for f in fs:
    try:
        d = json.load(open(f))
    except Exception:
        continue
    q = d.get("q_score", d.get("q", None))
    if q is not None:
        qs.append(float(q))
print(f"RESULTS n={len(qs)} mean_q={statistics.mean(qs):.4f} nonzero={sum(1 for q in qs if q>0)}" if qs
      else f"RESULTS none parsed from {len(fs)} json files")
PY

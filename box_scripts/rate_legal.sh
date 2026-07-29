#!/bin/bash
# LEGAL-BASELINE rate campaign: the q=1.0 serving config with the AFFORDANCE point source
# (rate_runs.sh with exactly ONE variable changed: oracle wrapper -> AffordanceMapFullRes).
#
# WHY: ckpt 49999 scored q=1.0 with oracle points whose effective error vs the metalink is
# 6.3 cm; the affordance head is at 2.6 cm. If point QUALITY was the binding factor, the legal
# arm should perform in the same band as the oracle arm (1/6-1/5 with one q=1.0). This is the
# first fully-legal closed-loop measurement AND the battle test of the entire Run-1 serve path
# (affordance + odometry + online map + map_tokens passthrough, ignored at K=0).
#
# VERIFY, DO NOT ASSUME: check affordance_wrapper_stats.json per run (n_inject > 0, conf stats,
# ms budgets) before believing any number.
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
  > /root/serve_legal.log 2>&1 < /dev/null &
echo $! > /root/serve.pid
sleep 85

mkdir -p /root/rate_legal
for N in 1 2 3 4 5; do
  echo "=== LEGAL RUN $N/5 ==="
  rm -f /root/action_log.jsonl /root/affordance_wrapper_stats.json
  mkdir -p /root/rate_legal/run_$N
  cd /root/bw/BEHAVIOR-1K
  env PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 XDG_RUNTIME_DIR=/tmp/xdg OMNIGIBSON_HEADLESS=1 \
    B1K_TASK_TARGETS=/root/g3_pipeline/task_targets.json B1K_TASK_NAME=turning_on_radio \
    /root/miniconda3/envs/behavior/bin/python -m omnigibson.eval.eval \
    --task-name turning_on_radio --host 127.0.0.1 --port 8000 \
    --env-wrapper behavior2026_eval.affordance_map_fullres.AffordanceMapFullRes \
    --instance-indices 0 --num-rollouts 1 --mode public_test \
    --output-dir /root/rate_legal/run_$N --write-video --headless \
    > /root/rate_legal/run_$N/eval.log 2>&1
  cp /root/action_log.jsonl /root/rate_legal/run_$N/ 2>/dev/null
  cp /root/affordance_wrapper_stats.json /root/rate_legal/run_$N/ 2>/dev/null
  /root/miniconda3/envs/openpi/bin/python - <<PY 2>/dev/null
import glob, json
js = glob.glob("/root/rate_legal/run_$N/json/*.json")
if js:
    d = json.load(open(js[0]))
    print(f"LEGAL_RESULT run=$N success={d['success']} q={d['q_score']['final']} steps={d['steps']}")
else:
    print(f"LEGAL_RESULT run=$N NO_JSON (crashed?)")
st = glob.glob("/root/rate_legal/run_$N/affordance_wrapper_stats.json")
if st:
    print("  stats:", open(st[0]).read().replace(chr(10), " "))
PY
done
echo "RATE_LEGAL_DONE"

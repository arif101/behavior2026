#!/bin/bash
set -u
exec >> /root/n25_chain.log 2>&1
echo "===== N25 CHAIN START $(date -u) ====="

serve_arm() {  # $1 = ckpt dir
  local dir="$1"
  local spid
  spid=$(pgrep -f 'serve_b1k.py' | head -1)
  if [ -n "$spid" ]; then kill "$spid"; for i in $(seq 1 20); do pgrep -f 'serve_b1k.py' >/dev/null || break; sleep 2; done; fi
  sleep 5
  rm -f /root/serve_n25.log
  ( cd /root/openpi_fork && XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 \
    /root/miniconda3/envs/openpi/bin/python scripts/b1k/serve_b1k.py \
    --policy.config pi05_radio_run2 --policy.dir "$dir" --robot b1k/R1Pro \
    --task b1k/turning_on_radio --repo-id b1k_radio --port 8000 > /root/serve_n25.log 2>&1 & )
  for i in $(seq 1 40); do grep -q "server listening" /root/serve_n25.log 2>/dev/null && { echo "SERVE UP: $dir"; return 0; }; sleep 6; done
  echo "SERVE FAILED: $dir"; return 1
}

run_arm() {  # $1=arm  $2=ckptdir
  serve_arm "$2" || { echo "ABORT $1: serve failed"; return 1; }
  echo "===== EVAL $1 n=25 START $(date -u) ====="
  mkdir -p /root/eval_$1_n25
  ( cd /root/bw/BEHAVIOR-1K && PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 XDG_RUNTIME_DIR=/tmp/xdg \
    OMNIGIBSON_HEADLESS=1 MAP_ARM=B B1K_TASK_TARGETS=/root/task_targets.json B1K_TASK_NAME=turning_on_radio \
    /root/miniconda3/envs/behavior/bin/python -u /root/eval_run3_arm.py \
    --arm $1 --num-rollouts 25 --port 8000 --output-dir /root/eval_$1_n25 --write-video --resume )
  echo "===== EVAL $1 n=25 DONE $(date -u) ====="
}

run_arm a0 /root/ckpt_a0/a0
run_arm a4 /root/ckpt_a4/a4
run_arm a2 /root/ckpt_a2/a2

echo "===== ALL N25 EVALS DONE $(date -u) ====="
for A in a0 a4 a2; do /root/miniconda3/envs/behavior/bin/python /root/reconstruct_summary.py /root/eval_${A}_n25 $A; done
echo "----- PAIRED A0 vs A4 (max-effect) -----"
/root/miniconda3/envs/behavior/bin/python /root/compare_arms.py /root/eval_a0_n25 /root/eval_a4_n25 A0 A4
echo "----- PAIRED A0 vs A2 (primary) -----"
/root/miniconda3/envs/behavior/bin/python /root/compare_arms.py /root/eval_a0_n25 /root/eval_a2_n25 A0 A2
echo "CHAIN_COMPLETE $(date -u)"

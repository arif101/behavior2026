#!/bin/bash
# Serve the run1b K=8 checkpoint (pi05_radio_map) at h32 — first K=8 serving anywhere,
# on 24GB (the submission-class fit test). Evaluator patches must be applied first.
set -ex
kill $(cat /root/serve.pid 2>/dev/null) 2>/dev/null || true
sleep 3
cd /root/openpi_fork
setsid nohup env XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 \
  LD_LIBRARY_PATH=/root/miniconda3/envs/openpi/lib \
  /root/miniconda3/envs/openpi/bin/python scripts/b1k/serve_b1k.py \
  --policy.config pi05_radio_map --policy.dir /root/ckpt \
  --robot b1k/R1Pro --task b1k/turning_on_radio --repo-id b1k_radio \
  --action-horizon 32 --port 8000 \
  > /root/serve_run1b.log 2>&1 &
echo $! > /root/serve.pid
echo "server launching (pid $(cat /root/serve.pid)); wait ~90s then check:"
echo "  grep -aiE 'ready|listening|error' /root/serve_run1b.log | tail -3"

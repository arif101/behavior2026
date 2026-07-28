#!/bin/bash
# Runs the moment training exits. Two measurements, both cheap, both GPU-only (no sim needed).
# Waits first because the trainer holds ~92% of every GPU; running concurrently would OOM.
#
# 1. PREFIX PROBE -- the architectural question. Point conditioning enters ONLY the action expert
#    (use_adarms=[False,True]; adarms_cond=[None,cond]), so the VLM -- where images and language are
#    attended together -- may never learn WHICH object matters. Ridge-decode the target from prefix
#    activations vs from the action expert, each against its own shuffled-target noise floor.
#      prefix clears its floor      -> VLM already infers the target; AdaLN is sufficient
#      prefix does NOT clear it     -> VLM is target-blind, and a SOFT TOKEN in the KV cache is the
#                                      fix -- the same mechanism as SERF's map tokens (40.7 -> 63.5)
#    The evidence usually cited against prefix injection (2606.27663: AdaLN 77.5 vs "prefix" 38.5)
#    injected the point as TEXT COORDINATES, so it tests an encoding, not a site.
#
# 2. VAL LOSS on the 20 held-out episodes, per checkpoint. 3B params over 180 demos of one task can
#    memorize. val(25800) vs val(50000) tells us whether 1.94 epochs is undertrained (still falling)
#    or converged (flat) -- which decides whether extending the run is worth any money at all.
set -u
cd /root/openpi
. .venv/bin/activate

echo "waiting for training to exit…"
while pgrep -f "train_b1k.py pi05" > /dev/null; do sleep 60; done
echo "training exited"

CKPT_DIR=/root/openpi/outputs/checkpoints/pi05_radio_gate/radio_gate
echo "=== checkpoints ==="
ls "$CKPT_DIR"

for step in $(ls "$CKPT_DIR" 2>/dev/null | sort -n); do
  P="$CKPT_DIR/$step/params"
  [ -d "$P" ] || continue
  echo ""
  echo "=========== step $step ==========="

  echo "--- val loss ---"
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 python /root/val_loss.py \
      --ckpt "$P" --config pi05_radio_gate \
      --out "/root/val_loss_$step.json" 2>&1 \
    | grep -avE "^W0|^I0|external/|warn" | tail -12

  echo "--- prefix probe ---"
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 python /root/prefix_probe.py \
      --ckpt "$P" --config pi05_radio_gate --n 1024 \
      --out "/root/prefix_probe_$step.json" 2>&1 \
    | grep -avE "^W0|^I0|external/|warn" | tail -30
done

echo ""
echo "POST_ANALYSIS_DONE"

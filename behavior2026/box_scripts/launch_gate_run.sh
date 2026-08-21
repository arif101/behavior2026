#!/bin/bash
# Wait for norm stats, then launch the radio gate run. Idempotent and safe to re-run.
#
# THE GATE: can this architecture reach nonzero q on ONE task?
#   turning_on_radio, 180 train / 20 held-out episodes, ~387k frames
#   ABSOLUTE joint actions (Comet measured delta 0.00 vs absolute 0.30 on this exact task;
#     PI's own pi05_libero uses absolute; our traces show ~65% command delivery under delta)
#   point conditioning on 100% of frames (Phase A was 2.67%, and the action-expert probe read
#     R2 0.369 for decoding the point but 0.046 for the future action -- present, never intent)
#
# Pre-registered read:
#   nonzero q, or fingertip < 8 cm on HELD-OUT instances  -> gate passes; SERF and RL unlock
#   flat after 10+ epochs, fingertip still 13-40 cm       -> undertraining REFUTED; the
#                                                            architecture work becomes primary
#
# batch 30 = 10/GPU across 3 A100-80GB (32 would not shard: 32 % 3 != 0).
# 12,900 steps ~= 1 epoch over 180 episodes; save_interval matches so every checkpoint is an epoch.
set -u

cd /root/openpi
. .venv/bin/activate

echo "waiting for norm stats…"
while pgrep -f compute_norm_stats > /dev/null; do sleep 60; done

STATS=$(find /root/openpi -name "norm_stats.json" 2>/dev/null | head -1)
if [ -z "$STATS" ]; then
  echo "ABORT: compute_norm_stats exited but wrote no norm_stats.json"
  tail -20 /root/normstats.log
  exit 1
fi
echo "norm stats present: $STATS"
python - <<'PY'
import json, glob
p = glob.glob("/root/openpi/**/norm_stats.json", recursive=True)[0]
d = json.load(open(p))
n = d.get("norm_stats", d)
for k in list(n)[:4]:
    v = n[k]
    if isinstance(v, dict) and "mean" in v:
        print(f"  {k}: dim {len(v['mean'])}")
PY

# Checkpoints are ~12 GB each; 300 GB disk with ~45 GB used leaves room for ~15. The run caps at
# 50k steps (~4 epochs) so this fits, but prune if the cap is ever raised.
echo "launching gate run…"
# NOTE: the config is a POSITIONAL arg (tyro subcommand), not --config.
nohup python scripts/b1k/train_b1k.py pi05_radio_gate \
  --exp-name radio_gate \
  > /root/train.log 2>&1 &

sleep 120
echo "--- first log lines ---"
grep -avE "^W0|^I0|external/" /root/train.log | tail -20
echo "GATE_LAUNCHED"

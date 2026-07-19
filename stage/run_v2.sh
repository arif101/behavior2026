#!/bin/bash
# Stage head v2 end-to-end on the box: cache -> split -> train -> eval.
# Prereq: prep_data.py has written /root/data/manifest.json.
set -euo pipefail
cd /root/stage
export B2026_CACHE=/root/cache_v05
export B2026_TASK_TARGETS=/root/task_targets.json

STEPS="${STEPS:-8000}"
BS="${BS:-32}"

python3 build_cache.py --manifest /root/data/manifest.json \
    --task_targets /root/task_targets.json

# Split: hold out the highest episode index per task (held-out EPISODES).
python3 - <<'EOF'
import json
from collections import defaultdict
eps = json.load(open("/root/data/manifest.json"))
by_task = defaultdict(list)
for e in eps:
    by_task[e["task"]].append(e["file_idx"])
train, val = [], []
for task, fis in by_task.items():
    fis = sorted(set(fis))
    for fi in fis[:-1]:
        train.append([task, fi])
    val.append([task, fis[-1]])
json.dump({"train": train, "val": val}, open("/root/data/splits.json", "w"), indent=1)
json.dump(val, open("/root/data/val_eps.json", "w"))
print(f"train {len(train)} eps, val {len(val)} eps")
EOF

python3 train.py --episodes /root/data/splits.json --out /root/stage_ckpt \
    --steps "$STEPS" --bs "$BS"

python3 eval.py --episodes /root/data/val_eps.json \
    --ckpt /root/stage_ckpt/best.pt --out /root/stage_ckpt/eval_heldout_eps.json

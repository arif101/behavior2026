#!/bin/bash
# Stage head v3 end-to-end: cache -> split (held-out episodes AND held-out
# tasks) -> train -> dual eval.  Prereq: prep_data.py wrote /root/data/manifest.json.
set -euo pipefail
cd /root/stage
export B2026_CACHE=/root/cache_v05
export B2026_TASK_TARGETS=/root/task_targets.json

STEPS="${STEPS:-12000}"
BS="${BS:-32}"
export HOLDOUT_TASKS="${HOLDOUT_TASKS:-putting_away_toys tidying_living_room}"

python3 build_cache.py --manifest /root/data/manifest.json \
    --task_targets /root/task_targets.json

# Split: held-out tasks contribute ALL their episodes to val_tasks (never
# trained); remaining tasks hold out their highest episode index (val_eps).
python3 - <<'EOF'
import json, os
from collections import defaultdict
holdout = set(os.environ["HOLDOUT_TASKS"].split())
eps = json.load(open("/root/data/manifest.json"))
by_task = defaultdict(list)
for e in eps:
    by_task[e["task"]].append(e["file_idx"])
train, val, val_tasks = [], [], []
for task, fis in sorted(by_task.items()):
    fis = sorted(set(fis))
    if task in holdout:
        val_tasks += [[task, fi] for fi in fis]
        continue
    for fi in fis[:-1]:
        train.append([task, fi])
    val.append([task, fis[-1]])
json.dump({"train": train, "val": val}, open("/root/data/splits.json", "w"), indent=1)
json.dump(val, open("/root/data/val_eps.json", "w"))
json.dump(val_tasks, open("/root/data/val_tasks.json", "w"))
print(f"train {len(train)} eps, val {len(val)} eps, holdout-task {len(val_tasks)} eps")
EOF

python3 train.py --episodes /root/data/splits.json --out /root/stage_ckpt \
    --steps "$STEPS" --bs "$BS"

python3 eval.py --episodes /root/data/val_eps.json \
    --ckpt /root/stage_ckpt/best.pt --out /root/stage_ckpt/eval_heldout_eps.json
python3 eval.py --episodes /root/data/val_tasks.json \
    --ckpt /root/stage_ckpt/best.pt --out /root/stage_ckpt/eval_heldout_tasks.json
echo RUN_V3_DONE

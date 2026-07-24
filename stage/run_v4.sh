#!/bin/bash
# Stage head v4 end-to-end: real literals -> per-arm extractor labels ->
# cache -> split -> train + dual eval -> kill-ablation (posbins) + dual eval.
# Prereqs: prep_data.py wrote /root/data/manifest.json; run_pred_sweep.sh
# finished (predicates + sweep_labels_v4); BEHAVIOR-1K clone for bddl3.
# Run detached:  setsid nohup bash run_v4.sh > /root/run_v4.log 2>&1 < /dev/null &
set -euo pipefail
cd /root/stage
export B2026_CACHE=/root/cache_v4
export B2026_TASK_TARGETS=/root/task_targets.json
export B2026_TASK_LITERALS=/root/task_literals.json
export B2026_PREDICATES=/root/predicates

STEPS="${STEPS:-12000}"
BS="${BS:-32}"
export HOLDOUT_TASKS="${HOLDOUT_TASKS:-putting_away_toys tidying_living_room}"

# --- 0. real goal literals from BDDL (idempotent, CPU) ----------------------
python3 - <<'EOF'
import json, os, subprocess
tasks = sorted({e["task"] for e in json.load(open("/root/data/manifest.json"))})
if not os.path.exists("/root/task_literals.json"):
    subprocess.run(["python3", "goal_literals.py", "--tasks", *tasks,
                    "--out", "/root/task_literals.json"], check=True, cwd="/root/stage")
EOF
python3 -c "import json; from common import FX,FY,CX,CY; \
    json.dump(dict(fx=FX,fy=FY,cx=CX,cy=CY), open('/root/intrinsics.json','w'))"

# --- 1. per-arm extractor labels over the sweep_labels_v4 replays -----------
python3 - <<'EOF'
import json, os, subprocess
man = json.load(open("/root/data/manifest.json"))
for e in man:
    task, fi = e["task"], e["file_idx"]
    labels = f"/root/sweep_labels_v4/{task}/labels_{fi:03d}.jsonl"
    done = f"/root/predicates/{task}/ep{fi:03d}.json.done"
    out = f"/root/perframe/{task}/ep{fi:03d}.jsonl"
    e["extractor_perframe"] = out
    # Only use extractor labels from a COMPLETED replay (.done). Timed-out
    # replays leave truncated label files that would give bad per-arm labels on
    # their tail -- those fall back to official-only arm attribution.
    if not os.path.exists(done):
        print(f"NO_COMPLETE_REPLAY {task}/ep{fi:03d} (official-only fallback)")
        e["extractor_perframe"] = None
        continue
    if os.path.exists(out):
        continue
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cmd = ["python3", "extract_stages.py", "--labels", labels,
           "--parquet", e["parquet"], "--task", task,
           "--task_targets", "/root/task_targets.json",
           "--intrinsics", "/root/intrinsics.json",
           "--out_json", out.replace(".jsonl", "_timeline.json"),
           "--out_perframe", out,
           "--episode_index", str(e["episode_index"])]
    if e.get("prange"):
        cmd += ["--prange", str(e["prange"][0]), str(e["prange"][1])]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd="/root/stage")
json.dump(man, open("/root/data/manifest_v4.json", "w"), indent=1)
print("PERFRAME_OK")
EOF

# --- 2. cache (real literals + real ledger + per-arm labels + posbins) ------
python3 build_cache.py --manifest /root/data/manifest_v4.json \
    --task_targets /root/task_targets.json

# --- 3. split: identical policy to v3 ---------------------------------------
python3 - <<'EOF'
import json, os
from collections import defaultdict
holdout = set(os.environ["HOLDOUT_TASKS"].split())
eps = json.load(open("/root/data/manifest_v4.json"))
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

# --- 4. semantic head (the real v4) -----------------------------------------
python3 train.py --episodes /root/data/splits.json --out /root/stage_ckpt_v4 \
    --steps "$STEPS" --bs "$BS"
python3 eval.py --episodes /root/data/val_eps.json \
    --ckpt /root/stage_ckpt_v4/best.pt --out /root/stage_ckpt_v4/eval_heldout_eps.json
python3 eval.py --episodes /root/data/val_tasks.json \
    --ckpt /root/stage_ckpt_v4/best.pt --out /root/stage_ckpt_v4/eval_heldout_tasks.json

# --- 5. kill ablation (spec eval 4): identical head, temporal-position bins -
python3 train.py --episodes /root/data/splits.json --out /root/stage_ckpt_v4_posbins \
    --steps "$STEPS" --bs "$BS" --label_file stage_labels_posbins.npz
# 5a. self-scored (its own bin labels) -- upper bound for the bin head
python3 eval.py --episodes /root/data/val_tasks.json \
    --ckpt /root/stage_ckpt_v4_posbins/best.pt --label_file stage_labels_posbins.npz \
    --out /root/stage_ckpt_v4_posbins/eval_heldout_tasks_selflabels.json
# 5b. scored against SEMANTIC labels -- stage acc is meaningless here by
# construction; ledger F1 / progress MAE / boundary metrics are the comparable
# rows of the kill table (semantic must win, spec 6.4)
python3 eval.py --episodes /root/data/val_tasks.json \
    --ckpt /root/stage_ckpt_v4_posbins/best.pt \
    --out /root/stage_ckpt_v4_posbins/eval_heldout_tasks_semlabels.json
python3 eval.py --episodes /root/data/val_eps.json \
    --ckpt /root/stage_ckpt_v4_posbins/best.pt \
    --out /root/stage_ckpt_v4_posbins/eval_heldout_eps_semlabels.json

echo RUN_V4_DONE

#!/bin/bash
set -x
export PATH=/root/miniconda3/bin:/root/.local/bin:$PATH
export HF_HUB_DISABLE_XET=1
# 1. stage data (RGB-only)
/root/miniconda3/envs/behavior/bin/python - << 'PY'
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import snapshot_download
import pandas as pd, glob
files = sorted(glob.glob("/workspace/2026-challenge-demos/meta/episodes/**/*.parquet", recursive=True))
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
sel = df[df["tasks"].apply(lambda t: "picking_up_trash" in list(t))]
pats = set()
cams = ["observation.rgb.zed_link_camera_0","observation.rgb.left_realsense_link_camera_0","observation.rgb.right_realsense_link_camera_0"]
for _, r in sel.iterrows():
    pats.add("data/chunk-%03d/file-%03d.parquet" % (r["data/chunk_index"], r["data/file_index"]))
    for cam in cams:
        pats.add("videos/%s/chunk-%03d/file-%03d.mp4" % (cam, r[f"videos/{cam}/chunk_index"], r[f"videos/{cam}/file_index"]))
print(len(sel), "eps,", len(pats), "files")
snapshot_download("behavior-1k/2026-challenge-demos", repo_type="dataset", local_dir="/workspace/2026-challenge-demos", allow_patterns=sorted(pats))
print("TRASH_DATA_OK")
PY
# 2. config entry + task registry
cd /workspace/openpi && python3 - << 'PY'
p = "src/openpi/training/config.py"
s = open(p).read()
if "pi05_b1k_trash_lora" not in s:
    i = s.index('        name="pi05_b1k_wood_lora"')
    start = s.rindex("    TrainConfig(", 0, i)
    d = 0; j = start
    while True:
        if s[j] == "(": d += 1
        elif s[j] == ")":
            d -= 1
            if d == 0: break
        j += 1
    end = s.index("\n", j) + 1
    block = s[start:end].replace("pi05_b1k_wood_lora", "pi05_b1k_trash_lora").replace("bringing_in_wood", "picking_up_trash")
    s = s[:end] + block + s[end:]
    open(p, "w").write(s)
import ast; ast.parse(open(p).read()); print("TRASH_CONFIG_OK")
p2 = "src/openpi/configs/tasks/b1k.py"
s2 = open(p2).read()
if "picking_up_trash" not in s2:
    s2 = s2.replace('"bringing_in_wood": "bringing_in_wood",', '"bringing_in_wood": "bringing_in_wood",\n    "picking_up_trash": "picking_up_trash",')
    open(p2, "w").write(s2)
import ast; ast.parse(open(p2).read()); print("REGISTRY_OK")
PY
# 3. norm stats then train LONG (30k, no early stop — testing the more-steps hypothesis)
uv run scripts/compute_norm_stats.py --config-name pi05_b1k_trash_lora --max-frames 80000 > /workspace/trash_norm.log 2>&1
[ -f outputs/assets/pi05_b1k_trash_lora/picking_up_trash/norm_stats.json ] || { echo NS_MISSING; exit 1; }
echo TRASH_NS_DONE
nohup env XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/b1k/train_b1k.py pi05_b1k_trash_lora --exp_name trash_lora --overwrite --no-wandb-enabled --checkpoint-base-dir /root/g2_ckpts > /workspace/train_trash.log 2>&1 &
echo TRASH_TRAIN_LAUNCHED

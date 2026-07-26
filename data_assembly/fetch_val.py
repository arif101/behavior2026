"""Fetch raw demos for the held-out val set. Downloads only files not already present:
the heldout-EPISODE stratum needs just video chunks (its data parquets are already local),
the heldout-TASK stratum needs both."""
import glob, json, os
import pandas as pd
from huggingface_hub import hf_hub_download

tok = open("/root/.hf_token").read().strip()
REPO = "behavior-1k/2026-challenge-demos"
ROOT = "/root/valA/demos"
md = "/root/phaseA/demos_meta"
os.makedirs(ROOT, exist_ok=True)

eps = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{md}/meta/episodes/*/*.parquet"))])
vp = json.load(open("/root/phaseA/val_plan.json"))
want = set(vp["heldout_episode"]) | set(vp["heldout_task"])
sel = eps[eps.episode_index.isin(want)]
print(f"resolving files for {len(sel)} episodes", flush=True)

RGB = ["observation.rgb.left_realsense_link_camera_0",
       "observation.rgb.right_realsense_link_camera_0",
       "observation.rgb.zed_link_camera_0"]
files = set()
for _, r in sel.iterrows():
    files.add(f"data/chunk-{int(r['data/chunk_index']):03d}/file-{int(r['data/file_index']):03d}.parquet")
    for vk in RGB:
        files.add(f"videos/{vk}/chunk-{int(r[f'videos/{vk}/chunk_index']):03d}/"
                  f"file-{int(r[f'videos/{vk}/file_index']):03d}.mp4")
files = sorted(files)
print(f"{len(files)} distinct files needed", flush=True)

done = 0
for i, f in enumerate(files):
    dst = os.path.join(ROOT, f)
    if os.path.exists(dst):
        done += 1
        continue
    try:
        hf_hub_download(REPO, f, repo_type="dataset", token=tok, local_dir=ROOT)
    except Exception as e:
        print(f"  SKIP {f}: {str(e)[:90]}", flush=True)
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(files)}", flush=True)
# copy the meta tree so the val set is a self-contained LeRobot dataset
os.system(f"mkdir -p {ROOT}/meta && cp -r {md}/meta/* {ROOT}/meta/ 2>/dev/null")
print(f"VAL_FETCH_DONE already-had={done} total={len(files)}", flush=True)

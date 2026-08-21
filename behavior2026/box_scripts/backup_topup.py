"""Final backup top-up before sim-box spin-down: everything that changed after the delta push.

Adds: refreshed code bundles (anti-shortcut commits), norm stats (training-critical, tiny),
live-map snapshots (debug-console source data), viz-eval result json/stats, demos list.
Then VERIFIES the whole repo state (file count + size) — the spin-down gate.
"""

import os

os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
from huggingface_hub import HfApi

REPO = "arif101/b26-foveated-backup-20260730"
tok = open("/root/.hf_token").read().strip()
api = HfApi(token=tok)

files = [
    ("/root/behavior2026.bundle", "code/behavior2026.bundle"),
    ("/root/openpi_factored.bundle", "code/openpi_factored.bundle"),
    ("/root/ckpt/assets/b1k_radio/norm_stats.json", "keys/norm_stats.json"),
    ("/root/demos.txt", "keys/demos.txt"),
]
for src, dst in files:
    if os.path.exists(src):
        api.upload_file(path_or_fileobj=src, path_in_repo=dst, repo_id=REPO, repo_type="dataset")
        print(f"  {dst} ({os.path.getsize(src)/1e6:.2f} MB)", flush=True)
    else:
        print(f"  MISSING: {src}", flush=True)

for d, dst in (("/root/map_live", "map_live"), ("/root/viz_eval/json", "campaigns/viz_eval_json")):
    if os.path.isdir(d):
        api.upload_folder(folder_path=d, path_in_repo=dst, repo_id=REPO, repo_type="dataset")
        print(f"  dir {dst}", flush=True)

info = api.repo_info(REPO, repo_type="dataset", files_metadata=True)
n = len(info.siblings)
sz = sum(s.size or 0 for s in info.siblings) / 1e9
print(f"TOPUP_DONE files={n} total={sz:.2f}GB", flush=True)

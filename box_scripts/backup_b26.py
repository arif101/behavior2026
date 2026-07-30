"""Backup the foveated-memory milestone to HF (private). Token FILE-BASED only (/root/.hf_token).

Ordering: cheapest-most-critical first, so a mid-upload box death still preserves the keys to
recreate everything else. b1k_radio_map (5.1 G) SUPERSEDES b1k_radio2 (it contains it + all
label/token columns). aff_data and rawdemos are skipped (recreatable from what IS backed up:
scripts live in the git bundles).
"""

import os
import time

os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
from huggingface_hub import HfApi

REPO = "arif101/b26-foveated-backup-20260730"
tok = open("/root/.hf_token").read().strip()
api = HfApi(token=tok)
api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
print(f"repo ready: {REPO}", flush=True)


def up_file(path, dest):
    t0 = time.time()
    api.upload_file(path_or_fileobj=path, path_in_repo=dest, repo_id=REPO, repo_type="dataset")
    print(f"  file {dest} ({os.path.getsize(path)/1e6:.1f} MB, {time.time()-t0:.0f}s)", flush=True)


def up_dir(path, dest):
    t0 = time.time()
    api.upload_folder(folder_path=path, path_in_repo=dest, repo_id=REPO, repo_type="dataset")
    print(f"  dir  {dest} ({time.time()-t0:.0f}s)", flush=True)


# 1. tiny critical keys
for f in ("episode_map.json", "metalink_offset.json", "camera_intrinsics.json"):
    up_file(f"/root/{f}", f"keys/{f}")

# 2. labels + tokens (the compute-expensive derivations)
up_dir("/root/metalink_labels", "metalink_labels")
up_dir("/root/map_tokens", "map_tokens")

# 3. affordance checkpoints + metrics
up_dir("/root/aff_out", "aff_out")
if os.path.isdir("/root/aff_out_728"):
    up_dir("/root/aff_out_728", "aff_out_728")

# 4. git bundles (all code incl. fork_snapshot + patch scripts)
up_file("/root/behavior2026.bundle", "code/behavior2026.bundle")
up_file("/root/openpi_factored.bundle", "code/openpi_factored.bundle")

# 5. campaign evidence (legal + oracle + blend: logs, stats, videos)
for d, dest in (("/root/rate_legal", "campaigns/rate_legal"),
                ("/root/rate", "campaigns/rate_oracle_h32"),
                ("/root/rate_blend", "campaigns/rate_blend")):
    if os.path.isdir(d):
        up_dir(d, dest)

# 6. the augmented training dataset (largest, last)
up_dir("/root/b1k_radio_map", "b1k_radio_map")

info = api.repo_info(REPO, repo_type="dataset", files_metadata=True)
n = len(info.siblings)
sz = sum(s.size or 0 for s in info.siblings) / 1e9
print(f"BACKUP_DONE files={n} total={sz:.2f}GB", flush=True)

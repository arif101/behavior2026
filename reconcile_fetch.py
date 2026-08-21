"""Reconcile the 43 fetch 404s: the episode ids are NOT uniformly 10..2000 step 10.

fetch_direct.py assumed deterministic ids and landed 157/200 with 43 404s. repo_info succeeded
earlier on this repo (it's snapshot_download that stalls), so list the ACTUAL task-0000 filenames
and fetch exactly the difference.
"""

import os

os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
from huggingface_hub import HfApi, hf_hub_download

tok = open("/root/.hf_token").read().strip()
api = HfApi(token=tok)
files = [f for f in api.list_repo_files("behavior-1k/2026-challenge-rawdata", repo_type="dataset")
         if f.startswith("task-0000/") and f.endswith(".hdf5")]
print(f"repo has {len(files)} hdf5 under task-0000/", flush=True)

d = "/root/rawdemos/task-0000"
have = {f"task-0000/{fn}" for fn in os.listdir(d)} if os.path.isdir(d) else set()
missing = sorted(set(files) - have)
print(f"have {len(have)}, missing {len(missing)}", flush=True)

ok = fail = 0
for fn in missing:
    try:
        hf_hub_download("behavior-1k/2026-challenge-rawdata", fn, repo_type="dataset",
                        local_dir="/root/rawdemos", token=tok)
        ok += 1
    except Exception as e:  # noqa: BLE001
        fail += 1
        print(f"FAIL {fn}: {str(e)[:80]}", flush=True)
print(f"DONE ok={ok} fail={fail} total_on_disk={len(have) + ok}", flush=True)

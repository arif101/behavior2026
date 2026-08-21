"""Fetch raw demos for the 4 expansion tasks (user sign-off 2026-08-07).

task-0001 picking_up_trash | task-0007 picking_up_toys | task-0015 bringing_in_wood |
task-0016 moving_boxes_to_storage  (indices from BEHAVIOR-1K docs/challenge/task_data.json,
verified radio == task-0000). One repo listing (episode ids are NOT assumed to follow the
radio 10..2000-step-10 pattern), then per-file hf_hub_download like fetch_direct.py
(snapshot_download stalls on the 20k-file repo). ~1.4G/task expected.
"""

import os

os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
from huggingface_hub import HfApi, hf_hub_download

TASKS = [1, 7, 15, 16]
tok = open("/root/.hf_token").read().strip()
api = HfApi(token=tok)

print("listing repo files...", flush=True)
fs = api.list_repo_files("behavior-1k/2026-challenge-rawdata", repo_type="dataset")
for tidx in TASKS:
    pre = f"task-{tidx:04d}/"
    files = sorted(f for f in fs if f.startswith(pre))
    print(f"{pre}: {len(files)} files", flush=True)
    ok = fail = skip = 0
    for fn in files:
        dst = f"/root/rawdemos/{fn}"
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            skip += 1
            continue
        try:
            hf_hub_download("behavior-1k/2026-challenge-rawdata", fn, repo_type="dataset",
                            local_dir="/root/rawdemos", token=tok)
            ok += 1
            if ok % 25 == 0:
                print(f"  {ok}/{len(files)}", flush=True)
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"  FAIL {fn}: {str(e)[:80]}", flush=True)
    print(f"TASK_DONE task-{tidx:04d} ok={ok} skip={skip} fail={fail}", flush=True)
print("FETCH_EXPANSION_DONE", flush=True)

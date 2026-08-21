"""Direct per-file fetch of the 200 turning_on_radio raw demos.

snapshot_download stalled indefinitely before its first byte on the 20,001-file
2026-challenge-rawdata repo (empty log, no files, 10+ minutes). The episode filenames are
deterministic — task-0000/episode_%08d.hdf5, ids 10..2000 step 10, matching our dataset's
raw_episode_id column exactly — so hf_hub_download per file needs no repo tree resolution.
"""

import os

os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
from huggingface_hub import hf_hub_download

tok = open("/root/.hf_token").read().strip()
ok = fail = 0
for i in range(10, 2001, 10):
    fn = f"task-0000/episode_{i:08d}.hdf5"
    try:
        hf_hub_download("behavior-1k/2026-challenge-rawdata", fn, repo_type="dataset",
                        local_dir="/root/rawdemos", token=tok)
        ok += 1
        if ok % 25 == 0:
            print(f"{ok} done", flush=True)
    except Exception as e:  # noqa: BLE001
        fail += 1
        print(f"FAIL {fn}: {str(e)[:80]}", flush=True)
print(f"DONE ok={ok} fail={fail}", flush=True)

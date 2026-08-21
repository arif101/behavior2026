#!/usr/bin/env python
"""Resilient sweep offloader. Fixes HF 500 on large multi-file commits by uploading videos
per-file with retries (preserves path structure). Decoupled label/video ledgers: labels ALWAYS
backed up even if a video fails. Deletes local mp4s only after confirmed video upload."""
import glob, json, os, sys, time
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import HfApi

REPO = "arif101/behavior2026-artifacts"
LBL, OUT, ST = "/root/sweep_labels", "/root/sweep_out", "/root/sweep_status"
LEDGER = "/root/sweep_status/_offload_ledger.json"
api = HfApi(token=open("/root/.hf_token").read().strip())

def load_ledger():
    return json.load(open(LEDGER)) if os.path.exists(LEDGER) else {"labels": [], "videos": []}
def save_ledger(l): json.dump(l, open(LEDGER, "w"))

def done_tasks():
    out = []
    for f in glob.glob(f"{ST}/*.json"):
        if "_uploaded" in f or "_offload" in f: continue
        d = json.load(open(f))
        if d.get("status") == "ok": out.append(d.get("task", os.path.basename(f)[:-5]))
    return out

def retry(fn, tries=4):
    for i in range(tries):
        try: return fn()
        except Exception as e:
            if i == tries - 1: raise
            # exponential backoff + deterministic jitter (no random import): 10,20,40,80,160s
            time.sleep(min(10 * (2 ** i) + (i * 7 % 11), 200))

def offload_pass():
    led = load_ledger()
    for t in done_tasks():
        if t not in led["labels"]:
            ld = f"{LBL}/{t}"
            if os.path.isdir(ld):
                try:
                    retry(lambda: api.upload_folder(folder_path=ld, repo_id=REPO, repo_type="model",
                          path_in_repo=f"sweep_100/sweep_labels/{t}"))
                    led["labels"].append(t); save_ledger(led); print(f"LABELS_OK {t}", flush=True)
                except Exception as e:
                    print(f"LABELS_FAIL {t}: {type(e).__name__} {str(e)[:80]}", flush=True)
        if t not in led["videos"]:
            od = f"{OUT}/{t}"
            mp4s = glob.glob(f"{od}/**/*.mp4", recursive=True)
            if os.path.isdir(od) and mp4s:
                # ONE commit per task (upload_folder), not per-file: per-file = commit storm
                # (~30 commits/task x 96 tasks x 2 boxes) that HF rate-limits. upload_folder
                # hashes+skips already-uploaded files, so retry is resumable.
                try:
                    retry(lambda: api.upload_folder(folder_path=od, repo_id=REPO, repo_type="model",
                          path_in_repo=f"sweep_100/sweep_out/{t}"), tries=6)
                    led["videos"].append(t); save_ledger(led)
                    for m in mp4s: os.remove(m)
                    print(f"VIDEOS_OK {t} ({len(mp4s)} mp4s, freed local)", flush=True)
                except Exception as e:
                    print(f"VIDEOS_FAIL {t}: {type(e).__name__} {str(e)[:80]}", flush=True)
    print(f"PASS_DONE labels={len(led['labels'])} videos={len(led['videos'])}", flush=True)

if __name__ == "__main__":
    once = "--once" in sys.argv
    while True:
        try: offload_pass()
        except Exception as e: print(f"PASS_ERR {type(e).__name__} {e}", flush=True)
        if once: break
        time.sleep(7200)

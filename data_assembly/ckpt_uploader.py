"""Upload each completed Phase-A checkpoint's params to HF as it appears.

The A/B is ~65h of GPU time; the train script only uploads an arm's FINAL checkpoint, so a
box failure at hour 50 would lose everything. A step dir is treated as complete once a NEWER
step dir exists beside it (orbax has moved on) or it has been untouched for 20 minutes.
Uploaded steps are recorded so restarts don't re-push.
"""
import json, pathlib, time, shutil
from huggingface_hub import HfApi

ROOT = pathlib.Path("/root/openpi/outputs/checkpoints/pi05_phaseA_point")
DONE = pathlib.Path("/root/ckpt_uploaded.json")
REPO = "arif101/behavior2026-artifacts"
api = HfApi(token=open("/root/.hf_token").read().strip())
done = set(json.load(open(DONE))) if DONE.exists() else set()

while True:
    try:
        for arm_dir in sorted(p for p in ROOT.iterdir() if p.is_dir()) if ROOT.exists() else []:
            steps = sorted([p for p in arm_dir.iterdir() if p.is_dir() and p.name.isdigit()],
                           key=lambda p: int(p.name))
            for i, s in enumerate(steps):
                key = f"{arm_dir.name}/{s.name}"
                if key in done:
                    continue
                params = s / "params"
                if not params.exists() or not any(params.iterdir()):
                    continue
                newer = i < len(steps) - 1
                stale = (time.time() - params.stat().st_mtime) > 1200
                if not (newer or stale):
                    continue
                api.upload_folder(folder_path=str(params),
                                  path_in_repo=f"ckpts/phaseA/{arm_dir.name}/{s.name}/params",
                                  repo_id=REPO, repo_type="model")
                done.add(key)
                json.dump(sorted(done), open(DONE, "w"))
                print(f"uploaded {key}", flush=True)
                # Drop optimizer state only once a NEWER checkpoint exists. Pruning the
                # latest would save disk at the cost of the ability to resume after a crash,
                # which is the whole point of uploading during a 65h run.
                if newer:
                    shutil.rmtree(s / "train_state", ignore_errors=True)
        # Standing sweep: prune optimizer state from ANY non-newest checkpoint already on HF.
        # Doing this only at upload time misses checkpoints that were newest when uploaded (the
        # stale-rule path) and never revisited -- which filled the disk to 52GB free mid-run.
        for arm_dir in sorted(p for p in ROOT.iterdir() if p.is_dir()) if ROOT.exists() else []:
            steps = sorted([p for p in arm_dir.iterdir() if p.is_dir() and p.name.isdigit()],
                           key=lambda p: int(p.name))
            for s in steps[:-1]:                       # never the newest: it is the resume point
                ts = s / "train_state"
                if ts.exists() and f"{arm_dir.name}/{s.name}" in done:
                    shutil.rmtree(ts, ignore_errors=True)
                    print(f"swept train_state from {arm_dir.name}/{s.name}", flush=True)
    except Exception as e:
        print(f"uploader error (continuing): {e}", flush=True)
    time.sleep(600)

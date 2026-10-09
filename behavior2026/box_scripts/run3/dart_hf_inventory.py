"""Inventory every version of every DART LeRobot root ever pushed to HF `arif101/b26-radio-manufactured` (the round script
re-used path_in_repo b1k_radio_dart_r<k> across restarts, overwriting earlier pushes; old versions live in the revision
history). For each dart-related commit: which b1k_radio_dart_r* roots changed, and their total_episodes/total_frames at that
revision. Read-only. Run on a box with /root/.hf_token: .venv/bin/python dart_hf_inventory.py"""
import json, os
from huggingface_hub import HfApi, hf_hub_download
tok = open("/root/.hf_token").read().strip(); api = HfApi(token=tok); REPO = "arif101/b26-radio-manufactured"
commits = list(api.list_repo_commits(REPO, repo_type="dataset"))
print(f"{len(commits)} commits total")
seen = {}
for c in reversed(commits):                       # oldest first
    try:
        files = api.list_repo_files(REPO, repo_type="dataset", revision=c.commit_id)
    except Exception as e:
        print(c.commit_id[:8], "list ERR", str(e)[:60]); continue
    roots = sorted({f.split("/")[0] for f in files if f.startswith("b1k_radio_dart_r") and f.endswith("meta/info.json")})
    changed = []
    for r in roots:
        try:
            p = hf_hub_download(REPO, f"{r}/meta/info.json", repo_type="dataset", token=tok, revision=c.commit_id, local_dir=f"/root/hf_probe/{c.commit_id[:8]}")
            info = json.load(open(p)); key = (r, info.get("total_episodes"), info.get("total_frames"))
            if seen.get(r) != key[1:]:
                changed.append(f"{r}={info.get('total_episodes')}eps/{info.get('total_frames')}fr"); seen[r] = key[1:]
        except Exception as e:
            changed.append(f"{r}=ERR {str(e)[:40]}")
    if changed:
        print(c.created_at.strftime("%m-%d %H:%M"), c.commit_id[:8], "|", " ".join(changed), "|", c.title[:50])
cur = api.list_repo_files(REPO, repo_type="dataset")
print("CURRENT roots:", sorted({f.split("/")[0] for f in cur if f.startswith("b1k_radio_dart_r")}))
print("factory_clips_dart meta files on HF:", len([f for f in cur if f.startswith("factory_clips_dart/") and f.endswith("_meta.json")]))
print("INVENTORY_DONE")

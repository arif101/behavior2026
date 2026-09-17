"""Back up the assembled Run-3 training tables (parquet + meta; videos are on HF as sources) before spinning the trainer down.
Target: arif101/b26-run3-mixes (dataset): mix_full/ (gist_head, hist_geo, stage_v2, progress, toggled, target_points_v2,
sample_weight), mix_a5_v2/, run3_logs/ (small)."""
import os, pathlib
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
from huggingface_hub import HfApi
tok = open("/root/.hf_token").read().strip(); api = HfApi(token=tok); REPO = "arif101/b26-run3-mixes"
api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
for name, root in (("mix_full", "/root/b1k_radio_mix_full"), ("mix_a5_v2", "/root/b1k_radio_mix_a5_v2")):
    api.upload_folder(folder_path=root, path_in_repo=name, repo_id=REPO, repo_type="dataset",
                      allow_patterns=["data/**", "meta/**"], ignore_patterns=["videos/**", "*.mp4"])
    print("UPLOADED", name, flush=True)
api.upload_folder(folder_path="/root/run3_logs", path_in_repo="run3_logs", repo_id=REPO, repo_type="dataset",
                  allow_patterns=["*.log", "*.out", "*.json"], ignore_patterns=["*.pid"])
print("UPLOADED run3_logs", flush=True)
fs = api.list_repo_files(REPO, repo_type="dataset"); print("BACKUP_DONE files=%d" % len(fs), flush=True)

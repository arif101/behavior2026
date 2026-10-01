"""Back up everything still box-only on the 4dall trainer (2026-10-01) to HF b26-run3-mixes:
  /root/run3_logs/**  + /root/bringup_all.log  -> run3_logs_4dall_0930/   (distinct prefix: run3_logs/ holds earlier boxes' logs)
  /root/b1k_radio_mix_all data/** meta/**     -> mix_all/                 (same data+meta pattern as mix_4d; videos rebuild from sources)
Verifies by listing the repo and comparing file counts + total bytes per prefix. Token from /root/.hf_token (never inline)."""
import os, glob, time; os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
from huggingface_hub import HfApi
tok = open("/root/.hf_token").read().strip(); api = HfApi(token=tok); REPO = "arif101/b26-run3-mixes"; t0 = time.time()
def local(root, pats):
    fs = [f for p in pats for f in glob.glob(os.path.join(root, p), recursive=True) if os.path.isfile(f)]
    return len(fs), sum(os.path.getsize(f) for f in fs)
def remote(prefix):
    info = api.repo_info(REPO, repo_type="dataset", files_metadata=True)
    fs = [s for s in info.siblings if s.rfilename.startswith(prefix + "/")]
    return len(fs), sum((s.size or 0) for s in fs)
# 1. logs
api.upload_folder(folder_path="/root/run3_logs", path_in_repo="run3_logs_4dall_0930", repo_id=REPO, repo_type="dataset")
api.upload_file(path_or_fileobj="/root/bringup_all.log", path_in_repo="run3_logs_4dall_0930/bringup_all.log", repo_id=REPO, repo_type="dataset")
ln, lb = local("/root/run3_logs", ["**/*"]); ln += 1; lb += os.path.getsize("/root/bringup_all.log")
rn, rb = remote("run3_logs_4dall_0930")
print(f"LOGS local {ln} files {lb} B | remote {rn} files {rb} B | {'OK' if rn >= ln and rb == lb else 'MISMATCH'}", flush=True)
# 2. mix_all tables + meta
api.upload_folder(folder_path="/root/b1k_radio_mix_all", path_in_repo="mix_all", repo_id=REPO, repo_type="dataset",
                  allow_patterns=["data/**", "meta/**"], ignore_patterns=["videos/**", "*.mp4"])
mn, mb = local("/root/b1k_radio_mix_all", ["data/**/*", "meta/**/*"]); rn2, rb2 = remote("mix_all")
print(f"MIX_ALL local {mn} files {mb} B | remote {rn2} files {rb2} B | {'OK' if rn2 >= mn and rb2 == mb else 'MISMATCH'}", flush=True)
ok = (rn >= ln and rb == lb) and (rn2 >= mn and rb2 == mb)
print(f"{'BACKUP_TRAINER_DONE' if ok else 'BACKUP_TRAINER_MISMATCH'} {time.time()-t0:.0f}s", flush=True)

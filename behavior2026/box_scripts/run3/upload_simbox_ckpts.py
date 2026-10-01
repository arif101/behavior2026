"""Sim box -> HF b26-run3-params: the mid-run 4dall params that exist only here (the trainer's finalize slimmed its copies and the
keep-newest-1 push rolled them off HF). Plain upload_folder per step; does NOT use run3_hf.py (its ckpt command deletes older
mid-run folders). Verifies file count + bytes per step. Token from /root/.hf_token."""
import os, glob, sys, time; os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
from huggingface_hub import HfApi
tok = open("/root/.hf_token").read().strip(); api = HfApi(token=tok); REPO = "arif101/b26-run3-params"; t0 = time.time()
steps = [int(s) for s in (sys.argv[1:] or ["5000", "7500", "10000"])]
ok_all = True
for s in steps:
    d = f"/root/run3_dl/4dall_{s}/4dall/ckpt_{s}/params"
    fs = [f for f in glob.glob(d + "/**/*", recursive=True) if os.path.isfile(f)]; lb = sum(os.path.getsize(f) for f in fs)
    api.upload_folder(folder_path=d, path_in_repo=f"4dall/ckpt_{s}/params", repo_id=REPO, repo_type="model")
    info = api.repo_info(REPO, repo_type="model", files_metadata=True)
    rs = [x for x in info.siblings if x.rfilename.startswith(f"4dall/ckpt_{s}/params/")]; rb = sum((x.size or 0) for x in rs)
    ok = len(rs) >= len(fs) and rb == lb; ok_all &= ok
    print(f"CKPT_{s} local {len(fs)} files {lb} B | remote {len(rs)} files {rb} B | {'OK' if ok else 'MISMATCH'} ({time.time()-t0:.0f}s)", flush=True)
print("UPLOAD_SIMBOX_CKPTS_DONE" if ok_all else "UPLOAD_SIMBOX_CKPTS_MISMATCH", flush=True)

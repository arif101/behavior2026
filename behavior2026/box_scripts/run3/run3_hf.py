"""HF upload helpers for the Run-3 driver (token from /root/.hf_token).
  python run3_hf.py ckpt <arm> <step> <params_dir>       -> <REPO>/<arm>/ckpt_<step>/params (keeps newest 1 mid-run)
  python run3_hf.py final <arm> <params_dir> <assets_dir> <log> [extra files...] -> <REPO>/<arm>/{params,assets,provenance}
  python run3_hf.py verify <arm>                           -> prints file count + GB under <arm>/
"""
import os
import pathlib
import sys

os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
from huggingface_hub import HfApi

REPO = os.environ.get("RUN3_HF_REPO", "arif101/b26-run3-params")
tok = open("/root/.hf_token").read().strip()
api = HfApi(token=tok)
api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)
cmd, arm = sys.argv[1], sys.argv[2]


def size_under(prefix):
    info = api.repo_info(REPO, repo_type="model", files_metadata=True)
    sib = [s for s in info.siblings if s.rfilename.startswith(prefix)]
    return len(sib), sum(s.size or 0 for s in sib) / 1e9


if cmd == "ckpt":
    step, pdir = int(sys.argv[3]), sys.argv[4]
    api.upload_folder(folder_path=pdir, path_in_repo=f"{arm}/ckpt_{step}/params", repo_id=REPO, repo_type="model")
    n, gb = size_under(f"{arm}/ckpt_{step}/")
    print(f"RUN3_CKPT_UPLOADED arm={arm} step={step} files={n} {gb:.2f}GB", flush=True)
    older = sorted({f.split("/")[1] for f in api.list_repo_files(REPO, repo_type="model")
                    if f.startswith(f"{arm}/ckpt_") and f.split("/")[1] != f"ckpt_{step}"})
    for o in older:
        api.delete_folder(f"{arm}/{o}", repo_id=REPO, repo_type="model")
        print(f"  deleted older mid-run {arm}/{o}", flush=True)
elif cmd == "final":
    pdir, adir, log = sys.argv[3], sys.argv[4], sys.argv[5]
    api.upload_folder(folder_path=pdir, path_in_repo=f"{arm}/params", repo_id=REPO, repo_type="model")
    api.upload_folder(folder_path=adir, path_in_repo=f"{arm}/assets", repo_id=REPO, repo_type="model")
    for f in [log] + sys.argv[6:]:
        if pathlib.Path(f).exists():
            api.upload_file(path_or_fileobj=f, path_in_repo=f"{arm}/provenance/{pathlib.Path(f).name}", repo_id=REPO, repo_type="model")
    n, gb = size_under(f"{arm}/params/")
    assert gb > 10.0, f"final params upload looks incomplete: {gb:.2f}GB"
    print(f"RUN3_FINAL_UPLOADED arm={arm} params_files={n} {gb:.2f}GB", flush=True)
elif cmd == "verify":
    n, gb = size_under(f"{arm}/")
    print(f"{REPO}/{arm}: {n} files, {gb:.2f}GB")
else:
    raise SystemExit(f"unknown cmd {cmd}")

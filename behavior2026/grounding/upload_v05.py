"""Upload v0.5 checkpoints + results to HF arif101/behavior2026-artifacts under
ckpts/grounding_v05/. One small commit per file (repo 500s on giant commits)."""

import os
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import HfApi

REPO = "arif101/behavior2026-artifacts"
OUT = "/root/grounding_v05"
FILES = [
    "ckpt_v05_dinov2.pt",
    "ckpt_v05_dinov3.pt",
    "ckpt_v05_radio_control.pt",
    "results_v05_dinov2.json",
    "results_v05_dinov3.json",
    "results_v05_radio_control.json",
    "occlusion_exclusion.json",
]
CODE = ["mt_common.py", "mt_build_cache.py", "mt_dataset.py", "model_mt.py",
        "mt_train.py", "mt_eval.py", "mt_validate_labels.py",
        "mt_extract_disp.py", "mt_occlusion_report.py", "mt_gt_overlay.py",
        "upload_v05.py", "run_v05.sh"]


def main():
    tok = open("/root/.hf_token").read().strip()
    api = HfApi(token=tok)
    for f in FILES:
        p = os.path.join(OUT, f)
        if not os.path.exists(p):
            print("skip (missing):", f)
            continue
        for attempt in range(3):
            try:
                api.upload_file(path_or_fileobj=p, repo_id=REPO, repo_type="model",
                                path_in_repo=f"ckpts/grounding_v05/{f}",
                                commit_message=f"grounding v0.5: {f}")
                print("uploaded", f)
                break
            except Exception as e:
                print("retry", f, repr(e))
                time.sleep(10 * (attempt + 1))
    # grids + GT label checks as small folder commits
    for sub in ["grids", "gt_checks"]:
        gd = os.path.join(OUT, sub)
        if os.path.isdir(gd):
            api.upload_folder(folder_path=gd, repo_id=REPO, repo_type="model",
                              path_in_repo=f"ckpts/grounding_v05/{sub}",
                              commit_message=f"grounding v0.5: {sub}")
            print(f"uploaded {sub}/")
    # code snapshot as one small folder commit
    import tempfile, shutil
    td = tempfile.mkdtemp()
    for c in CODE:
        shutil.copy(os.path.join("/root/grounding", c), td)
    api.upload_folder(folder_path=td, repo_id=REPO, repo_type="model",
                      path_in_repo="ckpts/grounding_v05/code",
                      commit_message="grounding v0.5: code snapshot")
    print("uploaded code/")


if __name__ == "__main__":
    main()

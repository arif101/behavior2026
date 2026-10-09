"""Recover the 14-episode b1k_radio_dart_r6 (pushed 2026-09-25 14:57, revision 659bc10b) that the 23:35 loop restart
overwrote with a 9-episode rebuild, and re-upload it under a unique name so nothing overwrites it again."""
import os; os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
from huggingface_hub import HfApi, snapshot_download
tok = open("/root/.hf_token").read().strip(); api = HfApi(token=tok); REPO = "arif101/b26-radio-manufactured"
REV, SRC, DST = "659bc10b", "b1k_radio_dart_r6", "b1k_radio_dart_r6_0925_1457"
local = f"/root/manufactured/{DST}"
snapshot_download(REPO, repo_type="dataset", token=tok, revision=REV, allow_patterns=[f"{SRC}/**"], local_dir="/root/hf_recover_r6")
import shutil, json, pathlib
shutil.rmtree(local, ignore_errors=True); shutil.copytree(f"/root/hf_recover_r6/{SRC}", local)
info = json.load(open(f"{local}/meta/info.json")); print("RECOVERED", DST, info["total_episodes"], "eps", info["total_frames"], "frames", flush=True)
api.upload_folder(folder_path=local, path_in_repo=DST, repo_id=REPO, repo_type="dataset",
                  commit_message=f"{DST}: recovered 14-ep round-6 DART root from revision {REV} (overwritten by a loop restart)")
print("RECOVER_R6_OK", flush=True)

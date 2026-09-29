"""Upload /root/b1k_radio_mix_readout (data + meta + videos, ~4-5 GB) to HF b26-run3-mixes/mix_readout for the sim-box paired-loss readout."""
import os; os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
from huggingface_hub import HfApi
tok = open("/root/.hf_token").read().strip(); api = HfApi(token=tok); REPO = "arif101/b26-run3-mixes"
api.upload_folder(folder_path="/root/b1k_radio_mix_readout", path_in_repo="mix_readout", repo_id=REPO, repo_type="dataset",
                  allow_patterns=["data/**", "meta/**", "videos/**"])
fs = [f for f in api.list_repo_files(REPO, repo_type="dataset") if f.startswith("mix_readout/")]
print(f"READOUT_MIX_UPLOADED files={len(fs)} mp4={sum(f.endswith('.mp4') for f in fs)} parquet={sum(f.endswith('.parquet') for f in fs)}", flush=True)

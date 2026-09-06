"""Run-3 trainer downloads (token from /root/.hf_token, never inlined).
  map      arif101/b26-foveated-backup-20260730 : b1k_radio_map (data+meta, v2) + keys/
  manu     arif101/b26-radio-manufactured       : factory + episodes + poison_windows.json
  ckpt     arif101/b26-run2-params              : params/ (init) + assets/ (Run-2 norm stats)
  approach arif101/b26-radio-manufactured       : b1k_radio_approach (A5, lands later)
"""
import os
import sys

os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
from huggingface_hub import snapshot_download

tok = open("/root/.hf_token").read().strip()
what = sys.argv[1]
if what == "map":
    p = snapshot_download("arif101/b26-foveated-backup-20260730", repo_type="dataset", token=tok,
                          local_dir="/root/backup", allow_patterns=["b1k_radio_map/*", "keys/*"])
elif what == "manu":
    p = snapshot_download("arif101/b26-radio-manufactured", repo_type="dataset", token=tok,
                          local_dir="/root/manufactured", ignore_patterns=["b1k_radio_approach/*"])
elif what == "approach":
    p = snapshot_download("arif101/b26-radio-manufactured", repo_type="dataset", token=tok,
                          local_dir="/root/manufactured", allow_patterns=["b1k_radio_approach/*"])
elif what == "ckpt":
    p = snapshot_download("arif101/b26-run2-params", repo_type="model", token=tok,
                          local_dir="/root/warmstart_run3_raw", allow_patterns=["params/*", "assets/*"])
else:
    raise SystemExit(f"unknown target {what}")
print(f"DL_OK {what} -> {p}", flush=True)

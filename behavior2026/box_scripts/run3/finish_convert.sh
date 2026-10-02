#!/bin/bash
# FINISH clips: raw renders (/root/factory_obs_finish/rac_9<id><code>_200.npz, LIVE-RaC format) -> LeRobot-v3 root
# /root/b1k_radio_finish_r<ROUND> -> depth labels -> sample weights -> HF arif101/b26-radio-manufactured:b1k_radio_finish_r<ROUND>.
# Derived from odart_convert.sh (2026-10-02); finish clips are already rac_<digits>_<digits>.npz, no tag re-encoding.
# BAR=strict (honest_strict) or relaxed. Usage: BAR=strict ROUND=finish1_1002 bash /root/finish_convert.sh
set -eu
BAR=${BAR:-strict}
PY=/root/openpi_fork/.venv/bin/python
SEL=/root/factory_obs_finish_$BAR; rm -rf $SEL; mkdir -p $SEL
$PY - "$BAR" "$SEL" <<'PYX'
import glob, json, os, sys, numpy as np
bar, sel = sys.argv[1], sys.argv[2]; n_all = n_sel = 0
for f in sorted(glob.glob("/root/factory_obs_finish/rac_*_200.npz")):
    n_all += 1
    try:
        m = json.loads(str(np.load(f, allow_pickle=False)["meta"]))
    except Exception as e:
        print(f"{os.path.basename(f)} UNREADABLE ({e!r}) -> removed"); os.remove(f); continue
    ok = bool(m.get("honest_strict")) if bar == "strict" else bool(m.get("honest"))
    print(f"{os.path.basename(f)} tag={m.get('tag')} gap0={m.get('gap0')} strict={m.get('honest_strict')} relaxed={m.get('honest')} n_obs={m.get('n_obs')} -> {'SELECT' if ok else 'skip'}")
    if ok:
        b = os.path.basename(f); os.symlink(f, os.path.join(sel, b)); n_sel += 1
        open(os.path.join(sel, "name_map.txt"), "a").write(f"{b} {m.get('tag')} {m.get('state')}\n")
print(f"SELECTED {n_sel}/{n_all} renders at bar={bar}")
PYX
OUT=/root/b1k_radio_finish_r${ROUND:?}
rm -rf $OUT
$PY /root/convert_clips_to_parquet.py --clips "$SEL/rac_*_200.npz" --out $OUT --ref /root/b1k_radio_map --labels /root/metalink_labels --overwrite
$PY /root/add_depth_aux_labels.py --root $OUT --overwrite-col
$PY /root/add_sample_weights.py --root $OUT --overwrite-col
$PY - <<'PYX'
import json, pathlib, glob, pandas as pd, os
root = pathlib.Path("/root/b1k_radio_finish_r" + os.environ["ROUND"])
info = json.loads((root / "meta" / "info.json").read_text())
df = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))])
print("EPISODES", info.get("total_episodes"), "FRAMES", info.get("total_frames"), "rows", len(df),
      "cols has gt_depth_ds", "gt_depth_ds" in df.columns, "sample_weight", "sample_weight" in df.columns)
PYX
$PY - <<'PYX'
import os
from huggingface_hub import HfApi
tok = open("/root/.hf_token").read().strip(); api = HfApi(token=tok); R = os.environ["ROUND"]
api.upload_folder(folder_path="/root/b1k_radio_finish_r" + R, path_in_repo="b1k_radio_finish_r" + R,
                  repo_id="arif101/b26-radio-manufactured", repo_type="dataset",
                  commit_message="FINISH twins: scripted finish from the 4dall policy stall states")
api.upload_folder(folder_path="/root/factory_clips_finish", path_in_repo="factory_clips_finish",
                  repo_id="arif101/b26-radio-manufactured", repo_type="dataset", allow_patterns=["*_meta.json"],
                  commit_message="finish clip meta")
print("HF_PUSH_OK")
PYX
echo FINISH_CONVERT_DONE

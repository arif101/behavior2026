#!/bin/bash
# Post-sweep: v11 approach renders -> LeRobot-v3 dataset b1k_radio_approach_v2 -> depth labels -> sample
# weights -> HF arif101/b26-radio-manufactured:b1k_radio_approach_v2 (dataset repo).
# BAR=strict (default; honest_strict = 2 cm certificate) or BAR=relaxed (4 cm) selects which renders convert;
# the factory only writes a render when the relaxed bar passed, so relaxed == every render present.
# Usage on the sim box: BAR=strict bash approach_convert_v11.sh > /root/approach_convert.out 2>&1
set -eu
BAR=${BAR:-strict}
PY=/root/openpi_fork/.venv/bin/python
SEL=/root/factory_obs2_$BAR; rm -rf $SEL; mkdir -p $SEL
$PY - "$BAR" "$SEL" <<'PYX'
import glob, json, os, sys, numpy as np
bar, sel = sys.argv[1], sys.argv[2]; n_all = n_sel = 0
for f in sorted(glob.glob("/root/factory_obs2/rac_*_200.npz")):
    n_all += 1
    m = json.loads(str(np.load(f, allow_pickle=False)["meta"]))
    ok = bool(m.get("honest_strict")) if bar == "strict" else bool(m.get("honest"))
    print(f"{os.path.basename(f)} demo={m['demo']} pre_disp={m['radio_disp_precontact']} pre_rot={m['radio_rot_precontact']} "
          f"strict={m.get('honest_strict')} relaxed={m.get('honest')} n_obs={m['n_obs']} -> {'SELECT' if ok else 'skip'}")
    if ok: os.symlink(f, os.path.join(sel, os.path.basename(f))); n_sel += 1
print(f"SELECTED {n_sel}/{n_all} renders at bar={bar}")
PYX
OUT=/root/b1k_radio_approach_v2
rm -rf $OUT
$PY /root/convert_clips_to_parquet.py --clips "$SEL/rac_*_200.npz" --out $OUT --ref /root/b1k_radio_map --labels /root/metalink_labels --overwrite
$PY /root/add_depth_aux_labels.py --root $OUT --overwrite-col
$PY /root/add_sample_weights.py --root $OUT --overwrite-col
$PY - <<'PYX'
import json, pathlib, glob, pandas as pd
root = pathlib.Path("/root/b1k_radio_approach_v2")
info = json.loads((root / "meta" / "info.json").read_text())
df = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))])
print("EPISODES", info.get("total_episodes"), "FRAMES", info.get("total_frames"), "rows", len(df),
      "cols has gt_depth_ds", "gt_depth_ds" in df.columns, "sample_weight", "sample_weight" in df.columns)
PYX
$PY - <<'PYX'
from huggingface_hub import HfApi
tok = open("/root/.hf_token").read().strip(); api = HfApi(token=tok)
api.upload_folder(folder_path="/root/b1k_radio_approach_v2", path_in_repo="b1k_radio_approach_v2",
                  repo_id="arif101/b26-radio-manufactured", repo_type="dataset",
                  commit_message="b1k_radio_approach_v2: v11 ORIENT-FIRST base-drive approach clips (joint 4 locked, 10-DOF)")
api.upload_folder(folder_path="/root/factory_clips_approach", path_in_repo="factory_clips_approach_v11",
                  repo_id="arif101/b26-radio-manufactured", repo_type="dataset", allow_patterns=["*_meta.json", "*_approach.npz"],
                  commit_message="v11 approach clip cmds + honesty meta (all attempted demos)")
print("HF_PUSH_OK")
PYX
echo APPROACH_CONVERT_V11_DONE

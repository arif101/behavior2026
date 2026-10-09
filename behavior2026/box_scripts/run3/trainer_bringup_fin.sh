#!/bin/bash
# FINISH-ARM trainer bring-up (2026-10-09, fresh 1x A100-80GB, >=250 GB disk). Derived from trainer_bringup_all.sh (09-29).
# Prereqs: /root/behavior2026 (repo archive at main >= the 10-09 restore commit: it carries assemble_run3_mix.py, relabel_v2.py,
# precompute_gists.py, add_cam_pose_column.py and the full openpi_fork tree again), /root/.hf_token (chmod 600).
# Stages: tooling -> fork venv (configs incl. pi05_radio_4d_fin) -> downloads in parallel (map backup incl. videos; factory/
# episodes/poison + ALL b1k_radio_[o]dart_r* roots + the 17 b1k_radio_finish_r* roots; A4 + FULL params; mix_all tables, fk/,
# fk_dart_r1-3, fk_b1k_radio_* for all 38 corrective+finish roots, the ODART twin map) -> prep_fin_data.sh (labels for the 17
# finish roots only, assemble /root/b1k_radio_mix_fin, regroup, MIX_FIN_OK, parity vs FULL for 4d_all/4d_fin, 40-step smoke)
# -> TRAINER_BRINGUP_FIN_DONE. Does NOT launch the driver: launch = `ARMS=4dfin RUN3_STEPS=15000 bash /root/run3/run3_driver.sh`
# once prep_fin.out shows PREP_FIN_DONE, PARITY_EXTRA rel_diff <= 1e-5 for both configs, a clean smoke (loader lines, loss=,
# no Traceback) and SMOKE_MEM max anon < 200 GB.
# Usage: setsid nohup bash /root/behavior2026/behavior2026/box_scripts/run3/trainer_bringup_fin.sh > /root/bringup_fin.log 2>&1 &
set -x
export PATH=/root/.local/bin:$PATH
say(){ echo "[bringup_fin $(date -u +%H:%M:%S)] $*"; }
mkdir -p /root/run3 /root/run3_logs /root/frame_cache
R=/root/behavior2026
cp $R/behavior2026/box_scripts/run3/*.py $R/behavior2026/box_scripts/run3/*.sh /root/run3/
cp $R/add_sample_weights.py $R/assemble_run3_mix.py /root/run3/
cp $R/behavior2026/box_scripts/restore_map_videos.py $R/behavior2026/box_scripts/deregister_depth_streams.py /root/run3/ 2>/dev/null
for f in assemble_run3_mix.py relabel_v2.py precompute_gists.py add_cam_pose_column.py precompute_hist_tokens.py unassemble_columns.py canonicalize_columns.py regroup_parquets.py register_feature.py add_sample_weights.py parity_smoke_4d.py prep_fin_data.sh run3_driver.sh; do [ -f /root/run3/$f ] || { say "BRINGUP_FIN_MISSING_HELPER $f"; exit 1; }; done
say STAGE_TOOLING_OK
apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq ffmpeg curl >/dev/null 2>&1 && say STAGE_APT_OK
cat > /root/dl_fin.py <<'PY'
import os, sys
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
from huggingface_hub import snapshot_download
tok = open("/root/.hf_token").read().strip(); what = sys.argv[1]
if what == "map":
    snapshot_download("arif101/b26-foveated-backup-20260730", repo_type="dataset", token=tok, local_dir="/root/backup", allow_patterns=["b1k_radio_map/*", "keys/*"])
elif what == "manu":
    snapshot_download("arif101/b26-radio-manufactured", repo_type="dataset", token=tok, local_dir="/root/manufactured",
                      allow_patterns=["b1k_radio_factory/*", "b1k_radio_episodes/*", "poison_windows.json", "b1k_radio_dart_r*", "b1k_radio_odart_r*", "b1k_radio_finish_r*"])
elif what == "params":
    snapshot_download("arif101/b26-run3-params", repo_type="model", token=tok, local_dir="/root/run3_dl", allow_patterns=["a4/params/*", "a4/assets/*", "full/params/*", "full/assets/*"])
elif what == "mixes":
    snapshot_download("arif101/b26-run3-mixes", repo_type="dataset", token=tok, local_dir="/root/mixes_bk",
                      allow_patterns=["mix_all/*", "fk/*", "fk_dart_r*", "fk_b1k_radio_*", "odart_episode_map.json"])
print(f"DL_OK {what}", flush=True)
PY
DL_PY=/root/openpi_fork/.venv/bin/python
[ -x /root/.local/bin/uv ] || [ -x /usr/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
rm -rf /root/openpi_fork && cp -a $R/openpi_fork /root/openpi_fork && cd /root/openpi_fork
GIT_LFS_SKIP_SMUDGE=1 uv sync 2>&1 | tail -3 && say STAGE_UV_SYNC_OK
.venv/bin/python -c "import openpi, jax; import openpi.training.config as c; [c.get_config(n) for n in ('pi05_radio_full','pi05_radio_geo','pi05_radio_4d','pi05_radio_4d_pd','pi05_radio_4d_all','pi05_radio_4d_allf','pi05_radio_4d_fin')]; print('CONFIGS_OK', c.get_config('pi05_radio_4d_fin').data.base_config.dataset_root)" 2>&1 | tail -2 | grep -q CONFIGS_OK && say STAGE_CONFIGS_OK || { say BRINGUP_FIN_CONFIG_IMPORT_FAILED; exit 1; }
for w in map manu params mixes; do (setsid nohup $DL_PY /root/dl_fin.py $w > /root/run3_logs/dl_$w.log 2>&1 &); done; say STAGE_DL_LAUNCHED
FAIL=0; for w in map manu params mixes; do until grep -qE "DL_OK|Traceback" /root/run3_logs/dl_$w.log; do sleep 20; done; grep -q DL_OK /root/run3_logs/dl_$w.log && say "download $w ok" || { say "download $w FAILED"; tail -3 /root/run3_logs/dl_$w.log; FAIL=1; }; done
[ $FAIL -eq 0 ] || { say BRINGUP_FIN_DL_FAILED; exit 1; }
cp /root/mixes_bk/odart_episode_map.json /root/odart_episode_map.json
echo "corrective roots: $(ls -d /root/manufactured/b1k_radio_dart_r* /root/manufactured/b1k_radio_odart_r* | wc -l) (expect 21)"; echo "finish roots: $(ls -d /root/manufactured/b1k_radio_finish_r* | wc -l) (expect 17)"; echo "fk dirs: $(ls -d /root/mixes_bk/fk* | wc -l) (expect 39)"; echo "mix_all backup: $(ls /root/mixes_bk/mix_all/data/**/*.parquet 2>/dev/null | wc -l) parquet files"
PY=/root/openpi_fork/.venv/bin/python; R3=/root/run3
[ -d /root/backup/b1k_radio_map/videos ] || $PY $R3/restore_map_videos.py
# ---- the whole data prep + parity + smoke -------------------------------------------------------------------------------
bash $R3/prep_fin_data.sh > /root/run3_logs/prep_fin.out 2>&1
tail -12 /root/run3_logs/prep_fin.out
grep -q PREP_FIN_DONE /root/run3_logs/prep_fin.out && say TRAINER_BRINGUP_FIN_DONE || say BRINGUP_FIN_PREP_FAILED

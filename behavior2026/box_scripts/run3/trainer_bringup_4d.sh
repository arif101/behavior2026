#!/bin/bash
# 4D-PERCEPTION trainer bring-up (2026-09-22, fresh 1x A100-80GB). Prereqs: /root/behavior2026 (repo archive at the
# commit with the 4D fork), /root/.hf_token. Stages: tooling + downloads (map/manufactured/A4 params/mix tables/FK
# poses) -> fork venv -> rebuild the mix VIDEO layout by re-running the assembler (deterministic) and overwrite its
# tables+meta with the backed-up ones (gists/relabel columns) -> cam_pose column -> frame cache + history tokens ->
# PARITY SMOKES (pi05_radio_geo, pi05_radio_4d vs pi05_radio_full on one fixed batch).
# Usage: setsid nohup bash /root/trainer_bringup_4d.sh > /root/bringup_4d.log 2>&1 &
set -x
export PATH=/root/.local/bin:$PATH
say(){ echo "[bringup4d $(date -u +%H:%M:%S)] $*"; }
mkdir -p /root/run3 /root/run3_logs /root/fk /root/frame_cache
R=/root/behavior2026
cp $R/behavior2026/box_scripts/run3/*.py $R/behavior2026/box_scripts/run3/*.sh /root/run3/
cp $R/add_sample_weights.py $R/assemble_run3_mix.py /root/run3/
cp $R/behavior2026/box_scripts/restore_map_videos.py $R/behavior2026/box_scripts/deregister_depth_streams.py /root/run3/ 2>/dev/null
say STAGE_TOOLING_OK
apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq ffmpeg curl >/dev/null 2>&1 && say STAGE_APT_OK
python3 -m pip install -q "huggingface_hub[hf_xet]" 2>&1 | tail -1
cat > /root/dl_4d.py <<'PY'
import os, sys
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
from huggingface_hub import snapshot_download
tok = open("/root/.hf_token").read().strip(); what = sys.argv[1]
if what == "map":
    snapshot_download("arif101/b26-foveated-backup-20260730", repo_type="dataset", token=tok, local_dir="/root/backup", allow_patterns=["b1k_radio_map/*", "keys/*"])
elif what == "manu":
    snapshot_download("arif101/b26-radio-manufactured", repo_type="dataset", token=tok, local_dir="/root/manufactured",
                      allow_patterns=["b1k_radio_factory/*", "b1k_radio_episodes/*", "b1k_radio_approach_v2/*", "poison_windows.json"])
elif what == "a4":
    snapshot_download("arif101/b26-run3-params", repo_type="model", token=tok, local_dir="/root/run3_dl", allow_patterns=["a4/params/*", "a4/assets/*"])
elif what == "mixes":
    snapshot_download("arif101/b26-run3-mixes", repo_type="dataset", token=tok, local_dir="/root/mixes_bk", allow_patterns=["mix_full/*", "fk/*"])
print(f"DL_OK {what}", flush=True)
PY
for w in map manu a4 mixes; do (setsid nohup python3 /root/dl_4d.py $w > /root/run3_logs/dl_$w.log 2>&1 &); done
say STAGE_DL_LAUNCHED
[ -x /root/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
rm -rf /root/openpi_fork && cp -a $R/openpi_fork /root/openpi_fork && cd /root/openpi_fork
GIT_LFS_SKIP_SMUDGE=1 uv sync 2>&1 | tail -3 && say STAGE_UV_SYNC_OK
.venv/bin/python -c "import openpi, jax; import openpi.training.config as c; [c.get_config(n) for n in ('pi05_radio_full','pi05_radio_geo','pi05_radio_4d')]; print('configs ok; devices', jax.devices())" && say STAGE_FORK_OK
for w in map manu a4 mixes; do until grep -qE "DL_OK|Traceback" /root/run3_logs/dl_$w.log; do sleep 20; done; grep -q DL_OK /root/run3_logs/dl_$w.log && say "download $w ok" || { say "download $w FAILED"; tail -3 /root/run3_logs/dl_$w.log; }; done
# ---- mix: rebuild the video layout with the assembler, then overwrite tables + meta from the backup -----------------
PY=/root/openpi_fork/.venv/bin/python; R3=/root/run3
MAP=/root/backup/b1k_radio_map; FAC=/root/manufactured/b1k_radio_factory; EPI=/root/manufactured/b1k_radio_episodes; APP=/root/manufactured/b1k_radio_approach_v2
if [ ! -d $MAP/videos ]; then $PY $R3/restore_map_videos.py; fi
$PY $R3/assemble_run3_mix.py --overwrite --sources $MAP $FAC $EPI $APP --out /root/b1k_radio_mix_full 2>&1 | grep -E "ASSEMBLED|Error|Traceback" | tail -2
rm -rf /root/b1k_radio_mix_full/data /root/b1k_radio_mix_full/meta && cp -a /root/mixes_bk/mix_full/data /root/mixes_bk/mix_full/meta /root/b1k_radio_mix_full/ && say STAGE_MIX_RESTORED
$PY -c "
from openpi.training import config as c; from openpi.training.data_loader import create_torch_dataset
cfg=c.get_config('pi05_radio_full'); ds=create_torch_dataset(cfg.data.create(cfg.assets_dirs, cfg.model), cfg.model.action_horizon, cfg.model)
print('MIX_LOAD_OK', len(ds))" 2>&1 | tail -1
# ---- cam_pose column (FK precompute from the sim box, backed up under fk/) ---------------------------------------------
if [ -f /root/mixes_bk/fk/cam_pose.npy ]; then $PY $R3/add_cam_pose_column.py --root /root/b1k_radio_mix_full --npy /root/mixes_bk/fk/cam_pose.npy --index /root/mixes_bk/fk/index.npy && say STAGE_CAM_POSE_OK; else say "CAM_POSE_MISSING (fk/ not on HF yet)"; fi
# ---- frame cache (decode once) + history tokens ------------------------------------------------------------------------
mkdir -p /root/ckpt_full_init && ln -sfn /root/run3_dl/a4/params /root/ckpt_full_init/params
cd /root/openpi_fork && XLA_PYTHON_CLIENT_PREALLOCATE=false $PY $R3/precompute_gists.py --root /root/b1k_radio_mix_full --params /root/run3_dl/a4/params --cache-frames /root/frame_cache/mix.npy 2>&1 | grep -E "PRECOMPUTE_GISTS_OK|Traceback|frame cache" | tail -2
XLA_PYTHON_CLIENT_PREALLOCATE=false $PY $R3/precompute_hist_tokens.py --root /root/b1k_radio_mix_full --params /root/run3_dl/a4/params --from-cache /root/frame_cache/mix.npy 2>&1 | grep -E "PRECOMPUTE_HIST_TOKENS_OK|Traceback" | tail -1
mkdir -p /root/openpi_fork/outputs/assets && for n in pi05_radio_geo pi05_radio_4d; do mkdir -p /root/openpi_fork/outputs/assets/$n && cp -r /root/run3_dl/a4/assets/* /root/openpi_fork/outputs/assets/$n/ 2>/dev/null; done
mkdir -p /root/openpi_fork/outputs/assets/pi05_radio_full && cp -r /root/run3_dl/a4/assets/* /root/openpi_fork/outputs/assets/pi05_radio_full/ 2>/dev/null
say STAGE_PRECOMPUTE_DONE
# ---- parity smokes: same batch, same params -> loss(full) vs loss(geo) vs loss(4d) --------------------------------------
$PY $R3/parity_smoke_4d.py 2>&1 | grep -E "PARITY|Traceback|Error" | tail -5
say TRAINER_BRINGUP_4D_DONE

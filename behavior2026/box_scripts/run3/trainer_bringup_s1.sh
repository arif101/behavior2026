#!/bin/bash
# S1 trainer bring-up (2026-09-15, fresh 1x A100-80GB runpod). Prereqs: /root/behavior2026 (repo archive), /root/.hf_token.
# Stages: apt ffmpeg, uv, fork venv (uv sync), Run-3 tooling at /root/run3, HF downloads (map, manufactured, Run-2 params).
# Usage: setsid nohup bash /root/trainer_bringup_s1.sh > /root/bringup_s1.log 2>&1 &
set -x
export PATH=/root/.local/bin:$PATH
say(){ echo "[bringup $(date -u +%H:%M:%S)] $*"; }
mkdir -p /root/run3 /root/run3_logs
R=/root/behavior2026
# --- tooling
cp $R/behavior2026/box_scripts/run3/*.py $R/behavior2026/box_scripts/run3/*.sh /root/run3/
cp $R/add_sample_weights.py $R/assemble_run3_mix.py /root/run3/
cp $R/behavior2026/box_scripts/restore_map_videos.py $R/behavior2026/box_scripts/deregister_depth_streams.py /root/run3/ 2>/dev/null
say STAGE_TOOLING_OK
# --- downloads first (network-bound, independent of the venv): use system python + huggingface_hub
apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq ffmpeg curl >/dev/null 2>&1 && say STAGE_APT_OK
python3 -m pip install -q "huggingface_hub[hf_xet]" 2>&1 | tail -1
cat > /root/dl_s1.py <<'PY'
import os, sys
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
from huggingface_hub import snapshot_download
tok = open("/root/.hf_token").read().strip(); what = sys.argv[1]
if what == "map":
    snapshot_download("arif101/b26-foveated-backup-20260730", repo_type="dataset", token=tok, local_dir="/root/backup", allow_patterns=["b1k_radio_map/*", "keys/*"])
elif what == "manu":
    snapshot_download("arif101/b26-radio-manufactured", repo_type="dataset", token=tok, local_dir="/root/manufactured",
                      allow_patterns=["b1k_radio_factory/*", "b1k_radio_episodes/*", "b1k_radio_approach_v2/*", "poison_windows.json"])
elif what == "ckpt":
    snapshot_download("arif101/b26-run2-params", repo_type="model", token=tok, local_dir="/root/warmstart_run3_raw", allow_patterns=["params/*", "assets/*"])
print(f"DL_OK {what}", flush=True)
PY
for w in map manu ckpt; do (setsid nohup python3 /root/dl_s1.py $w > /root/run3_logs/dl_$w.log 2>&1 &); done
say STAGE_DL_LAUNCHED
# --- uv + fork venv
[ -x /root/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
uv --version && say STAGE_UV_OK
rm -rf /root/openpi_fork && cp -a $R/openpi_fork /root/openpi_fork && cd /root/openpi_fork
GIT_LFS_SKIP_SMUDGE=1 uv sync 2>&1 | tail -3 && say STAGE_UV_SYNC_OK
.venv/bin/python -c "import openpi, jax; import openpi.training.config as c; [c.get_config(n) for n in ('pi05_radio_run2','pi05_radio_run3_a4','pi05_radio_run3_a5')]; print('fork ok; jax devices', jax.devices())" && say STAGE_FORK_OK
.venv/bin/python -c "import torchcodec, av; print('video deps ok')" 2>&1 | tail -1
# --- wait for downloads
for w in map manu ckpt; do
  until grep -qE "DL_OK|Traceback" /root/run3_logs/dl_$w.log; do sleep 20; done
  grep -q DL_OK /root/run3_logs/dl_$w.log && say "download $w ok" || { say "download $w FAILED"; tail -3 /root/run3_logs/dl_$w.log; }
done
du -sh /root/backup /root/manufactured /root/warmstart_run3_raw; ls /root/manufactured
df -h / | tail -1
say TRAINER_BRINGUP_S1_DONE

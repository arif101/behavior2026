#!/bin/bash
# TRAINER bring-up for the task-62 context-VLA fine-tune (A100-class box, NOT for OmniGibson).
# Mirrors box_scripts/bringup_trainer.sh + serve_run2.sh's fork restore, then adds the two
# task-62 patches. Everything comes from the repo + HF; nothing box-only.
#
#   fork  = base tarball (arif101/behavior2026-artifacts) + fork_snapshot/ (repo, run-1 layer)
#           + run-2 patches (RUN2_CONFIG.md order) + patch_context_cond + patch_t62_config
#   data  = arif101/b26-t62-dataset :: b1k_t62/{data,meta,videos}  -> /root/b1k_t62
#   warm  = arif101/b26-run2-params :: ckpt_49999/params            -> /root/warmstart_t62/params
#   stats = arif101/b26-t62-dataset :: b1k_t62/norm_stats/norm_stats.json -> outputs/assets/pi05_t62_ctx/b1k_t62/
#
# Usage: HF token at /root/.hf_token (chmod 600) then  bash task62/bringup_trainer_t62.sh
set -ex
export PATH=/root/miniconda3/bin:$PATH
export HF_XET_HIGH_PERFORMANCE=1
export DEBIAN_FRONTEND=noninteractive
[ -f /root/.hf_token ] || { echo "MISSING /root/.hf_token"; exit 1; }

# --- 0. system + conda ----------------------------------------------------------------
apt-get update -qq && apt-get install -y -qq git git-lfs ffmpeg unzip >/dev/null && echo STAGE_APT_OK
if [ ! -x /root/miniconda3/bin/conda ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/mc.sh && bash /root/mc.sh -b -p /root/miniconda3 && rm -f /root/mc.sh
fi
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
[ -x /root/miniconda3/envs/openpi/bin/python ] || conda create -y -q -n openpi python=3.11
PY=/root/miniconda3/envs/openpi/bin/python
$PY -m pip install -q --upgrade pip setuptools wheel "huggingface_hub[hf_xet]" && echo STAGE_CONDA_OK

# --- 1. repo + fork restore -----------------------------------------------------------
[ -d /root/behavior2026 ] || git clone -b task-62-context-vla git@github.com:arif101/behavior2026.git /root/behavior2026
cd /root/behavior2026 && git checkout -q task-62-context-vla && git pull -q
if [ ! -d /root/openpi_fork ]; then
  $PY - <<'PYEOF'
import os; os.environ["HF_XET_HIGH_PERFORMANCE"]="1"
from huggingface_hub import hf_hub_download
tok=open("/root/.hf_token").read().strip()
hf_hub_download("arif101/behavior2026-artifacts","code/openpi_fork_adaln_src.tar.gz",token=tok,local_dir="/root/stage_dl")
PYEOF
  mkdir -p /root/fork_extract && tar xzf /root/stage_dl/code/openpi_fork_adaln_src.tar.gz -C /root/fork_extract && mv /root/fork_extract/openpi /root/openpi_fork
fi
F=/root/openpi_fork/src/openpi
for f in pi0.py pi0_config.py model.py; do cp /root/behavior2026/fork_snapshot/$f $F/models/$f; done
cp /root/behavior2026/fork_snapshot/b1k_policy.py $F/policies/
for f in config.py data_loader.py weight_loaders.py; do cp /root/behavior2026/fork_snapshot/$f $F/training/$f; done
cd /root/openpi_fork
$PY -m pip install -q -e . 2>&1 | tail -1
$PY -m pip install -q "jax[cuda12]" 2>&1 | tail -1
$PY -m pip install -q "lerobot[dataset] @ git+https://github.com/wensi-ai/lerobot@release/b1k" 2>&1 | tail -1
$PY -c "import jax; d=jax.devices(); print(d); assert any('cuda' in str(x).lower() for x in d), 'JAX is CPU-only'" && echo STAGE_JAX_GPU_OK
cd /root/behavior2026
for p in patch_stage_head patch_map_adaln patch_depth_aux patch_modality_dropout patch_run2_config patch_stage_oversample patch_context_cond patch_t62_config; do
  /root/miniconda3/bin/python $p.py 2>/dev/null || $PY $p.py; done
$PY -c "import sys; sys.path.insert(0,'/root/openpi_fork/src'); import openpi.training.config as C; c=C._CONFIGS_DICT['pi05_t62_ctx']; assert c.model.context_conditioning and c.model.stage_classes==8; print('pi05_t62_ctx ok')" && echo STAGE_FORK_OK

# --- 2. data + warm start + norm stats ----------------------------------------------------
$PY - <<'PYEOF'
import os; os.environ["HF_XET_HIGH_PERFORMANCE"]="1"
from huggingface_hub import snapshot_download
tok=open("/root/.hf_token").read().strip()
snapshot_download("arif101/b26-t62-dataset", repo_type="dataset", token=tok, local_dir="/root/hf_t62")
snapshot_download("arif101/b26-run2-params", token=tok, local_dir="/root/hf/b26-run2-params", allow_patterns=["ckpt_49999/params/**","assets/**"])
print("DATA_OK")
PYEOF
ln -sfn /root/hf_t62/b1k_t62 /root/b1k_t62
mkdir -p /root/warmstart_t62 && ln -sfn /root/hf/b26-run2-params/ckpt_49999/params /root/warmstart_t62/params
mkdir -p /root/openpi_fork/outputs/assets/pi05_t62_ctx/b1k_t62
cp /root/hf_t62/b1k_t62/norm_stats/norm_stats.json /root/openpi_fork/outputs/assets/pi05_t62_ctx/b1k_t62/
$PY -c "
import sys; sys.path.insert(0,'/root/openpi_fork/src')
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.dataset_reader import DatasetReader
m=LeRobotDatasetMetadata(repo_id='b1k_t62', root='/root/b1k_t62'); r=DatasetReader(meta=m, root=m.root, episodes=None, tolerance_s=5e-4, video_backend=None, delta_timestamps=None, image_transforms=None, return_uint8=False)
assert r.try_load(), 'LeRobot reader cannot load /root/b1k_t62 locally'; print('dataset ok', m.total_episodes, m.total_frames)" && echo STAGE_DATA_OK
df -h / | tail -1
echo BRINGUP_TRAINER_T62_DONE
echo "Launch: cd /root/openpi_fork && setsid nohup env XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 $PY scripts/b1k/train_b1k.py pi05_t62_ctx --exp_name t62_ctx --keep-period 5000 --no-wandb-enabled > /root/t62_train.log 2>&1 &"

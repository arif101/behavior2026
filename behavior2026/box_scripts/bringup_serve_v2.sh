#!/bin/bash
# EVAL-BOX SERVING STACK (runs in PARALLEL with bringup_v5.sh's sim install).
# Waits for v5's miniconda, then: openpi env + wensi fork + fork_snapshot OVERWRITE (canonical —
# fresh forks lack the G3 layer; patch chains only apply to already-G3 forks) + run1b ckpt +
# affordance/serving pieces from backups. set -ex non-negotiable.
set -ex

echo "waiting for miniconda from bringup_v5..."
while [ ! -x /root/miniconda3/bin/conda ]; do sleep 20; done
export PATH=/root/miniconda3/bin:$PATH
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
[ -x /root/miniconda3/envs/openpi/bin/python ] || conda create -y -q -n openpi python=3.11
PY=/root/miniconda3/envs/openpi/bin/python
$PY -m pip install -q --upgrade pip setuptools wheel huggingface_hub
echo STAGE_S1_ENV_OK

[ -d /root/openpi_fork ] || git clone -q -b behavior https://github.com/wensi-ai/openpi.git /root/openpi_fork
# canonical restore: snapshot overwrite (6 model/train files + weight_loaders)
for f in pi0.py pi0_config.py model.py; do
  cp /root/work/behavior2026/fork_snapshot/$f /root/openpi_fork/src/openpi/models/$f; done
cp /root/work/behavior2026/fork_snapshot/b1k_policy.py /root/openpi_fork/src/openpi/policies/
for f in config.py data_loader.py weight_loaders.py; do
  cp /root/work/behavior2026/fork_snapshot/$f /root/openpi_fork/src/openpi/training/$f; done
$PY - <<'EOF'
import py_compile
for f in ("models/pi0.py","models/pi0_config.py","models/model.py",
          "policies/b1k_policy.py","training/config.py","training/data_loader.py",
          "training/weight_loaders.py"):
    py_compile.compile(f"/root/openpi_fork/src/openpi/{f}", doraise=True)
print("snapshot files compile")
EOF
cd /root/openpi_fork
$PY -m pip install -q -e . 2>&1 | tail -1
$PY -m pip install -q "jax[cuda12]" 2>&1 | tail -1
$PY -m pip install -q "lerobot[dataset] @ git+https://github.com/wensi-ai/lerobot@release/b1k" 2>&1 | tail -1
$PY -c "import jax; print('jax devices:', jax.devices())"
echo STAGE_S2_FORK_OK

# downloads: run1b ckpt + serving pieces from the two backup repos
cat > /root/dl_serve.py <<'EOF'
import os
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
from huggingface_hub import snapshot_download
tok = open("/root/.hf_token").read().strip()
snapshot_download("arif101/b26-run1b-ckpt36559", token=tok, local_dir="/root/ckpt_run1b",
                  allow_patterns=["params/*"])
print("CKPT_OK", flush=True)
snapshot_download("arif101/b26-foveated-backup-20260730", repo_type="dataset", token=tok,
                  local_dir="/root/backup",
                  allow_patterns=["keys/*", "aff_out/*", "metalink_labels/*", "map_tokens/*",
                                  "b1k_radio_map/meta/*", "b1k_radio_map/data/*"])
print("PIECES_OK", flush=True)
EOF
$PY /root/dl_serve.py
echo STAGE_S3_DL_OK

# serving layout: ckpt + assets + norm stats + /root modules the wrappers import
mkdir -p /root/ckpt/assets/b1k_radio /root/openpi_fork/outputs/assets/pi05_radio_map/b1k_radio
ln -sfn /root/ckpt_run1b/params /root/ckpt/params
cp /root/backup/keys/norm_stats.json /root/ckpt/assets/b1k_radio/
cp /root/backup/keys/norm_stats.json /root/openpi_fork/outputs/assets/pi05_radio_map/b1k_radio/
ln -sfn /root/backup/b1k_radio_map /root/b1k_radio_map
cp /root/work/behavior2026/mapper/foveated_map.py /root/
cp /root/work/behavior2026/affordance/train_affordance.py /root/
cp /root/backup/keys/camera_intrinsics.json /root/ 2>/dev/null || true
cp /root/backup/keys/episode_map.json /root/ 2>/dev/null || true
mkdir -p /root/aff_out && cp -r /root/backup/aff_out/* /root/aff_out/
# eval wrappers package
mkdir -p /root/behavior2026_eval && touch /root/behavior2026_eval/__init__.py
cp /root/work/behavior2026/eval/*.py /root/behavior2026_eval/
# torch for the affordance head (serving uses behavior env GPU? no — wrapper runs INSIDE the
# eval process (behavior env). DINO+head deps live there; v5's env has torch already.)
echo STAGE_S4_LAYOUT_OK
echo SERVE_BRINGUP_DONE

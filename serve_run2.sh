#!/bin/bash
# RUN-2 SERVING STACK for the contact-skill phase (parallel with resume2.sh's sim install).
# Per HANDOFF.md: fork restore = tarball (BASE AdaLN fork) + fork_snapshot/ overlay (7 files,
# run-1 layer) + 5 run-2 patches (RUN2_CONFIG.md order) + robot_r1 patch. Eval-kit patch chain
# (point_passthrough -> map_passthrough2 + action_logger) runs SEPARATELY after sim install.
# Serving ckpt: radio_run2@49999 (arif101/b26-run2-params). set -ex non-negotiable.
set -ex
export PATH=/root/miniconda3/bin:$PATH
export HF_HUB_DISABLE_XET=1

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
[ -x /root/miniconda3/envs/openpi/bin/python ] || conda create -y -q -n openpi python=3.11
PY=/root/miniconda3/envs/openpi/bin/python
$PY -m pip install -q --upgrade pip setuptools wheel huggingface_hub hf_transfer
echo STAGE_S1_ENV_OK

# --- fork: tarball + snapshot overlay ----------------------------------------
if [ ! -d /root/openpi_fork ]; then
  $PY - <<'PYEOF'
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import hf_hub_download
tok = open("/root/.hf_token").read().strip()
p = hf_hub_download("arif101/behavior2026-artifacts", "code/openpi_fork_adaln_src.tar.gz",
                    repo_type="model", token=tok, local_dir="/root/stage_dl")
print(p)
PYEOF
  mkdir -p /root/fork_extract
  tar xzf /root/stage_dl/code/openpi_fork_adaln_src.tar.gz -C /root/fork_extract
  mv /root/fork_extract/openpi /root/openpi_fork
fi
for f in pi0.py pi0_config.py model.py; do
  cp /root/behavior2026/fork_snapshot/$f /root/openpi_fork/src/openpi/models/$f; done
cp /root/behavior2026/fork_snapshot/b1k_policy.py /root/openpi_fork/src/openpi/policies/
for f in config.py data_loader.py weight_loaders.py; do
  cp /root/behavior2026/fork_snapshot/$f /root/openpi_fork/src/openpi/training/$f; done
$PY - <<'PYEOF'
import py_compile
for f in ("models/pi0.py", "models/pi0_config.py", "models/model.py",
          "policies/b1k_policy.py", "training/config.py", "training/data_loader.py",
          "training/weight_loaders.py"):
    py_compile.compile(f"/root/openpi_fork/src/openpi/{f}", doraise=True)
print("snapshot files compile")
PYEOF
cd /root/openpi_fork
$PY -m pip install -q -e . 2>&1 | tail -1
$PY -m pip install -q "jax[cuda12]" 2>&1 | tail -1
$PY -m pip install -q "lerobot[dataset] @ git+https://github.com/wensi-ai/lerobot@release/b1k" 2>&1 | tail -1
$PY -c "import jax; print('jax devices:', jax.devices())"
echo STAGE_S2_FORK_OK

# --- run-2 patches (RUN2_CONFIG.md order) + robot_r1 -------------------------
cd /root/behavior2026/box_scripts
for p in patch_stage_head patch_map_adaln patch_depth_aux patch_modality_dropout patch_run2_config; do
  $PY $p.py; done
$PY patch_b1k_robot_name.py
echo STAGE_S3_PATCH_OK

# --- downloads: run2 params + serving pieces + prior data + raw demos --------
$PY - <<'PYEOF'
import os
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
from huggingface_hub import snapshot_download
tok = open("/root/.hf_token").read().strip()
snapshot_download("arif101/b26-run2-params", token=tok, local_dir="/root/ckpt_run2",
                  allow_patterns=["params/**", "assets/**"])
print("RUN2_PARAMS_OK", flush=True)
snapshot_download("arif101/b26-foveated-backup-20260730", repo_type="dataset", token=tok,
                  local_dir="/root/backup",
                  allow_patterns=["keys/*", "aff_out/*", "metalink_labels/*", "map_tokens/*",
                                  "b1k_radio_map/**"])
print("PIECES_OK", flush=True)
snapshot_download("arif101/b26-corrective-backup-20260806", repo_type="dataset", token=tok,
                  local_dir="/root/corrective", allow_patterns=["b1k_radio_corrective/**"])
print("CORRECTIVE_OK", flush=True)
pats = [f"task-0000/episode_{i:08d}.hdf5" for i in range(40)]
snapshot_download("behavior-1k/2026-challenge-rawdata", repo_type="dataset",
                  local_dir="/root/rawdemos", allow_patterns=pats)
print("RAWDEMOS_OK", flush=True)
PYEOF
echo STAGE_S4_DL_OK

# --- serving layout ----------------------------------------------------------
mkdir -p /root/ckpt/assets
ln -sfn /root/ckpt_run2/params /root/ckpt/params
cp -r /root/ckpt_run2/assets/* /root/ckpt/assets/
for cfg in pi05_radio_map pi05_radio_run2; do
  mkdir -p /root/openpi_fork/outputs/assets/$cfg
  cp -r /root/ckpt_run2/assets/* /root/openpi_fork/outputs/assets/$cfg/
done
ln -sfn /root/backup/b1k_radio_map /root/b1k_radio_map
ln -sfn /root/backup/metalink_labels /root/metalink_labels
ln -sfn /root/corrective/b1k_radio_corrective /root/b1k_radio_corrective
cp /root/behavior2026/mapper/foveated_map.py /root/
cp /root/behavior2026/affordance/train_affordance.py /root/
cp /root/backup/keys/*.json /root/ 2>/dev/null || true
mkdir -p /root/aff_out && cp -r /root/backup/aff_out/* /root/aff_out/
mkdir -p /root/behavior2026_eval && touch /root/behavior2026_eval/__init__.py
cp /root/behavior2026/eval/*.py /root/behavior2026_eval/
cp /root/behavior2026/box_scripts/*.py /root/ 2>/dev/null || true
echo STAGE_S5_LAYOUT_OK
echo SERVE_RUN2_BRINGUP_DONE

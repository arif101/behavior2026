#!/bin/bash
# TRAINER BRING-UP (2xA100-80GB) — stages 1-4: env + fork + data + ckpt. Gates run separately.
# set -ex is NOT optional (the install_openpi_simbox.sh lesson: silent DONE over total failure).
set -ex

echo "=== STAGE 1: miniconda + env ==="
if [ ! -d /root/miniconda3 ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/mc.sh
  bash /root/mc.sh -b -p /root/miniconda3
fi
export PATH=/root/miniconda3/bin:$PATH
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
[ -x /root/miniconda3/envs/openpi/bin/python ] || conda create -y -q -n openpi python=3.11
PY=/root/miniconda3/envs/openpi/bin/python
$PY -m pip install -q --upgrade pip setuptools wheel
echo STAGE1_OK

echo "=== STAGE 2: fork + deps (background downloads run in parallel below) ==="
[ -d /root/openpi_fork ] || git clone -q -b behavior https://github.com/wensi-ai/openpi.git /root/openpi_fork

echo "=== STAGE 3 (parallel): dataset + labels + ckpt downloads ==="
cat > /root/dl.py <<'EOF'
import os, sys
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
from huggingface_hub import snapshot_download
tok = open("/root/.hf_token").read().strip()
what = sys.argv[1]
if what == "data":
    snapshot_download("arif101/b26-foveated-backup-20260730", repo_type="dataset", token=tok,
                      local_dir="/root/backup",
                      allow_patterns=["b1k_radio_map/*", "map_tokens/*", "metalink_labels/*"])
    print("DATA_DL_OK")
else:
    snapshot_download("arif101/behavior2026-radio-gate-49999", token=tok,
                      local_dir="/root/warmstart_49999_raw")
    print("CKPT_DL_OK")
EOF
$PY -m pip install -q huggingface_hub
nohup $PY /root/dl.py data > /root/dl_data.log 2>&1 &
nohup $PY /root/dl.py ckpt > /root/dl_ckpt.log 2>&1 &

echo "=== STAGE 2b: fork deps (the slow pip) ==="
cd /root/openpi_fork
$PY -m pip install -q -e . 2>&1 | tail -2
$PY -m pip install -q "jax[cuda12]" 2>&1 | tail -1
# pip -e ignores [tool.uv.sources]; the lerobot pin must be explicit (approved 2026-07-29)
$PY -m pip install -q "lerobot[dataset] @ git+https://github.com/wensi-ai/lerobot@release/b1k" 2>&1 | tail -2
$PY -c "import jax; print('jax devices:', jax.devices())"
echo STAGE2_OK

echo "=== STAGE 4: patch chain + snapshot diff ==="
for p in patch_map_fork.py patch_aux_losses.py patch_aux_v2.py patch_stage_oversample.py patch_antishortcut.py; do
  cp /root/work/behavior2026/box_scripts/$p /root/
  $PY /root/$p
done
for f in models/pi0.py models/pi0_config.py models/model.py policies/b1k_policy.py training/config.py training/data_loader.py; do
  if ! diff -q /root/openpi_fork/src/openpi/$f /root/work/behavior2026/fork_snapshot/$(basename $f) >/dev/null 2>&1; then
    echo "SNAPSHOT DIFF (review): $f"
  fi
done
echo STAGE4_OK

echo "=== waiting for downloads ==="
while ! grep -q DATA_DL_OK /root/dl_data.log 2>/dev/null; do sleep 15; done
while ! grep -q CKPT_DL_OK /root/dl_ckpt.log 2>/dev/null; do sleep 15; done
ln -sfn /root/backup/b1k_radio_map /root/b1k_radio_map
ln -sfn /root/backup/map_tokens /root/map_tokens
ln -sfn /root/backup/metalink_labels /root/metalink_labels
mkdir -p /root/smoke_cwd/outputs/assets/pi05_radio_map/b1k_radio
cp /root/backup/keys/norm_stats.json /root/smoke_cwd/outputs/assets/pi05_radio_map/b1k_radio/
echo "ckpt layout:"; find /root/warmstart_49999_raw -maxdepth 3 -type d | head -10
echo BRINGUP_STAGES_1_4_OK

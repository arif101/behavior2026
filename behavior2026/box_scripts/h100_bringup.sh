#!/bin/bash
# H100 BRING-UP → RUN-1 LAUNCH (Aug 2-3). Executes the checklist from
# project_simbox_spindown_2026_07_30.md end to end. Run sections in order; each gate must
# print its PASS line before proceeding. Token is FILE-BASED: put it at /root/.hf_token first.
set -euo pipefail

BACKUP=arif101/b26-foveated-backup-20260730
CKPT_BACKUP=arif101/openpi-b1k-ckpt49999   # trainer-era checkpoint backup repo (adjust if named differently)
PY=python3   # adjust to the box's env python once the env exists

echo "=== [0] sanity: token present, GPU visible ==="
test -s /root/.hf_token
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== [1] fetch backup keys + code ==="
export HF_XET_HIGH_PERFORMANCE=1
$PY - <<'EOF'
import os
os.environ["HF_XET_HIGH_PERFORMANCE"]="1"
from huggingface_hub import snapshot_download
tok=open("/root/.hf_token").read().strip()
snapshot_download("arif101/b26-foveated-backup-20260730", repo_type="dataset", token=tok,
                  local_dir="/root/backup", allow_patterns=["keys/*","code/*"])
print("keys+code fetched")
EOF

echo "=== [2] restore code from bundles ==="
mkdir -p /root/work && cd /root/work
git clone /root/backup/code/behavior2026.bundle behavior2026
# fork: clone the b1k openpi fork (same origin as before), then apply the patch chain
# git clone <b1k-openpi-fork-url> openpi_fork   # <- fill at run time
cd /root/work/openpi_fork
for p in patch_map_fork.py patch_aux_losses.py patch_aux_v2.py patch_stage_oversample.py patch_antishortcut.py; do
  cp /root/work/behavior2026/box_scripts/$p /root/
  $PY /root/$p
done
echo "PATCH CHAIN APPLIED — diff against fork_snapshot/ for belt-and-braces:"
for f in models/pi0.py models/pi0_config.py models/model.py policies/b1k_policy.py training/config.py training/data_loader.py; do
  diff -q src/openpi/$f /root/work/behavior2026/fork_snapshot/$(basename $f) || echo "  REVIEW DIFF: $f"
done

echo "=== [3] dataset + norm stats ==="
$PY - <<'EOF'
import os
os.environ["HF_XET_HIGH_PERFORMANCE"]="1"
from huggingface_hub import snapshot_download
tok=open("/root/.hf_token").read().strip()
snapshot_download("arif101/b26-foveated-backup-20260730", repo_type="dataset", token=tok,
                  local_dir="/root/backup", allow_patterns=["b1k_radio_map/*"])
print("dataset fetched")
EOF
ln -sfn /root/backup/b1k_radio_map /root/b1k_radio_map
mkdir -p /root/smoke_cwd/outputs/assets/pi05_radio_map/b1k_radio
cp /root/backup/keys/norm_stats.json /root/smoke_cwd/outputs/assets/pi05_radio_map/b1k_radio/

echo "=== [4] warm-start checkpoint ==="
# restore ckpt 49999 from its backup repo -> /root/warmstart_49999/params
# (path is what pi05_radio_map's weight_loader expects)
# $PY restore_ckpt.py  # <- trainer-era restore script from the bundle
test -d /root/warmstart_49999/params && echo "WARMSTART IN PLACE" || echo "!! RESTORE CKPT FIRST"

echo "=== [5] GATE: loader smoke (must print SMOKE PASS) ==="
cp /root/work/behavior2026/box_scripts/smoke_map_loader.py /root/
cd /root/smoke_cwd && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 $PY /root/smoke_map_loader.py | tail -3

echo "=== [6] GATE: oversampler fires (must print [stage-oversample]) ==="
cd /root/smoke_cwd && B1K_STAGE_OVERSAMPLE=8 JAX_PLATFORMS=cpu $PY - <<'EOF'
import dataclasses
from openpi.training import config as _config
from openpi.training import data_loader as _dl
cfg = dataclasses.replace(_config.get_config("pi05_radio_map"), batch_size=4, num_workers=0)
next(iter(_dl.create_b1k_data_loader(cfg, shuffle=True, num_batches=1)))
print("OVERSAMPLE GATE PASS")
EOF

echo "=== [7] GATE: G0 preflight with RESTORED weights (all 4 checks) ==="
cp /root/work/behavior2026/box_scripts/preflight_map_influence.py /root/
$PY /root/preflight_map_influence.py --ckpt /root/warmstart_49999 --config pi05_radio_map

echo "=== [8] LAUNCH RUN 1 ==="
cat <<'CMD'
cd /root/work/openpi_fork && setsid nohup env B1K_STAGE_OVERSAMPLE=8 \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 \
  python scripts/train.py pi05_radio_map --exp-name radio_map_run1 \
  > /root/run1_train.log 2>&1 &
CMD
echo "review the printed command, then execute it. Monitor: aux-loss curves per term,"
echo "grad norms, and the FIRST checkpoint's quick-eval before committing the full 11h."

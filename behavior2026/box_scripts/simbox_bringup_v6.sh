#!/bin/bash
# SIM BOX BRING-UP v6 (2026-09-12, RTX PRO 4500 Blackwell / Ubuntu 24.04 / runpod slice).
# Differences from bringup.sh: everything under /root/bw (this box's /workspace is a MooseFS
# network volume); assets via AUTHENTICATED hf_hub_download + xet (not unauthenticated aria2);
# our openpi fork = the TRACKED tree in the repo (already run-2 patched + robot_r1 + trainer
# layer) installed with uv; eval-kit patch chain applied; no G2/G3 HF restore stage.
# Prereqs: /root/.hf_token, /root/behavior2026 (repo archive), miniconda at /root/miniconda3,
# apt deps (ffmpeg kmod git-lfs libxt6 libglu1-mesa libxrandr2 libxinerama1 libxcursor1 libxi6
# libegl1 libgl1 libglvnd0 libglx0 libopengl0 gcc), uv at /root/.local/bin.
# Run: setsid nohup bash /root/behavior2026/behavior2026/box_scripts/simbox_bringup_v6.sh > /root/bringup_v6.log 2>&1 &
set -x
export PATH=/root/miniconda3/bin:/root/.local/bin:$PATH
BW=/root/bw; mkdir -p $BW/wheels $BW/dl
REPO=/root/behavior2026
PYB=/root/miniconda3/envs/behavior/bin/python
say(){ echo "[bringup $(date -u +%H:%M:%S)] $*"; }

# ---- 1. asset zip (31.5 GB) via authenticated HF, in the background --------------------------
cat > /root/dl_assets.py <<'PY'
import os; os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
from huggingface_hub import hf_hub_download
tok = open("/root/.hf_token").read().strip()
p = hf_hub_download("behavior-1k/zipped-datasets", "behavior-1k-assets-3.9.0.zip", repo_type="dataset", token=tok, local_dir="/root/bw/dl")
print("ASSETS_DL_OK", p, flush=True)
PY
python3 -m pip install -q --break-system-packages "huggingface_hub[hf_xet]" 2>&1 | tail -1
nohup python3 /root/dl_assets.py > /root/dl_assets.log 2>&1 &

# ---- 2. BEHAVIOR-1K v3.9.0 + torch cu128 wheels (Blackwell sm_120) -----------------------------
[ -d $BW/BEHAVIOR-1K ] || git clone -b v3.9.0 --depth 1 https://github.com/StanfordVL/BEHAVIOR-1K.git $BW/BEHAVIOR-1K && say STAGE_CLONE_OK
cd $BW/wheels
for f in torch-2.7.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl \
         torchvision-0.22.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl \
         torchaudio-2.7.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl \
         torchcodec-0.5%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl; do
  [ -f "$(echo $f | sed 's/%2B/+/')" ] || curl -sO "https://download.pytorch.org/whl/cu128/$f" &
done; wait
ls -la $BW/wheels
sed -i "s|--index-url https://download.pytorch.org/whl/cu\${CUDA_VER_SHORT}|--find-links $BW/wheels|" $BW/BEHAVIOR-1K/setup.sh
say STAGE_WHEELS_OK

# ---- 3. wait for assets, extract ---------------------------------------------------------------
until grep -qE "ASSETS_DL_OK|Traceback" /root/dl_assets.log; do sleep 20; done
grep -q ASSETS_DL_OK /root/dl_assets.log || { say "ASSETS_DL_FAILED"; tail -5 /root/dl_assets.log; exit 1; }
mkdir -p $BW/BEHAVIOR-1K/datasets/behavior-1k-assets
python3 -c 'import zipfile; zipfile.ZipFile("/root/bw/dl/behavior-1k-assets-3.9.0.zip").extractall("/root/bw/BEHAVIOR-1K/datasets/behavior-1k-assets")' && say STAGE_ASSETS_OK
du -sh $BW/BEHAVIOR-1K/datasets/behavior-1k-assets

# ---- 4. B1K setup (conda env `behavior`, Isaac Sim, OmniGibson, eval kit, task instances) -----
cd $BW/BEHAVIOR-1K
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
./setup.sh --new-env --omnigibson --bddl --joylo --eval --dataset \
  --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos && say STAGE_B1K_SETUP_OK
mkdir -p /root/og_appdata
mv $BW/BEHAVIOR-1K/OmniGibson/appdata $BW/BEHAVIOR-1K/OmniGibson/appdata.orig 2>/dev/null
ln -sfn /root/og_appdata $BW/BEHAVIOR-1K/OmniGibson/appdata && say STAGE_APPDATA_OK
$PYB -c "import omnigibson, torch; print('OG', omnigibson.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_capability())" && say STAGE_OG_IMPORT_OK
ls $BW/BEHAVIOR-1K/datasets/ ; ls $BW/BEHAVIOR-1K/datasets/2026-challenge-task-instances 2>/dev/null | head -3

# ---- 5. our openpi fork (tracked tree) via uv --------------------------------------------------
rm -rf /root/openpi_fork && cp -a $REPO/openpi_fork /root/openpi_fork && cd /root/openpi_fork
GIT_LFS_SKIP_SMUDGE=1 uv sync 2>&1 | tail -3 && say STAGE_OPENPI_SYNC_OK
.venv/bin/python -c "import openpi, jax; import openpi.training.config as c; c.get_config('pi05_radio_run2'); c.get_config('pi05_radio_press'); print('fork ok; jax devices', jax.devices())" && say STAGE_OPENPI_OK
grep -n 'name="robot_r1"' src/openpi/configs/robots/b1k.py | head -1

# ---- 6. eval-kit patch chain + behavior2026_eval package + serving pieces ----------------------
BS=$REPO/behavior2026/box_scripts
sed -i "s|/root/bw/BEHAVIOR-1K|$BW/BEHAVIOR-1K|g" $BS/patch_point_passthrough.py $BS/patch_map_passthrough2.py $BS/patch_action_logger.py 2>/dev/null
for p in patch_point_passthrough patch_map_passthrough2 patch_action_logger; do python3 $BS/$p.py; done
mkdir -p /root/behavior2026_eval && touch /root/behavior2026_eval/__init__.py
cp $REPO/behavior2026/eval/*.py /root/behavior2026_eval/
python3 $BS/patch_wrapper_arms.py
cp $REPO/behavior2026/mapper/foveated_map.py /root/ ; cp $REPO/behavior2026/affordance/train_affordance.py /root/ 2>/dev/null
cp $REPO/behavior2026/task_targets.json /root/task_targets.json
cp $REPO/*.py /root/ 2>/dev/null; cp $BS/*.py /root/ 2>/dev/null
say STAGE_EVALKIT_OK

# ---- 7. HF pieces: run2 + a4 params, backup keys/aff_out/labels/map_tokens/map data, raw demos --
cat > /root/dl_pieces.py <<'PY'
import os; os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
from huggingface_hub import snapshot_download
tok = open("/root/.hf_token").read().strip()
snapshot_download("arif101/b26-run2-params", token=tok, local_dir="/root/ckpt_run2ref", allow_patterns=["params/**", "assets/**"]); print("RUN2_PARAMS_OK", flush=True)
snapshot_download("arif101/b26-run3-params", token=tok, local_dir="/root/run3_dl", allow_patterns=["a4/params/**", "a4/assets/**"]); print("A4_PARAMS_OK", flush=True)
snapshot_download("arif101/b26-foveated-backup-20260730", repo_type="dataset", token=tok, local_dir="/root/backup",
                  allow_patterns=["keys/*", "aff_out/*", "metalink_labels/*", "map_tokens/*", "b1k_radio_map/**"]); print("PIECES_OK", flush=True)
snapshot_download("arif101/b26-radio-manufactured", repo_type="dataset", token=tok, local_dir="/root/manufactured",
                  ignore_patterns=["raw_segments/*"]); print("MANUFACTURED_OK", flush=True)
pats = [f"task-0000/episode_{i:08d}.hdf5" for i in range(40)]
snapshot_download("behavior-1k/2026-challenge-rawdata", repo_type="dataset", token=tok, local_dir="/root/rawdemos", allow_patterns=pats); print("RAWDEMOS_OK", flush=True)
PY
/root/openpi_fork/.venv/bin/python /root/dl_pieces.py 2>&1 | grep -E "_OK|Traceback|Error" && say STAGE_DL_OK
mkdir -p /root/ckpt_a4 && ln -sfn /root/run3_dl/a4/params /root/ckpt_a4/params && cp -r /root/run3_dl/a4/assets /root/ckpt_a4/ 2>/dev/null
mkdir -p /root/ckpt && ln -sfn /root/ckpt_run2ref/params /root/ckpt/params && cp -r /root/ckpt_run2ref/assets /root/ckpt/ 2>/dev/null
ln -sfn /root/backup/b1k_radio_map /root/b1k_radio_map; ln -sfn /root/backup/metalink_labels /root/metalink_labels; ln -sfn /root/backup/map_tokens /root/map_tokens
mkdir -p /root/aff_out && cp -r /root/backup/aff_out/* /root/aff_out/; cp /root/backup/keys/*.json /root/ 2>/dev/null
say STAGE_LAYOUT_OK

# ---- 8. serve smoke (pi05_radio_run2 config + a4 params) --------------------------------------
cd /root/openpi_fork && sed "s|/root/ckpt\b|/root/ckpt_a4|; s|RUN2_SERVE_SMOKE_OK|A4_SERVE_SMOKE_OK|" $REPO/smoke_run2_serve.py > /root/smoke_a4_serve.py
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python /root/smoke_a4_serve.py 2>&1 | grep -E "SMOKE_OK|Error|Traceback" | tail -3 && say STAGE_SERVE_SMOKE_DONE
say SIMBOX_BRINGUP_V6_DONE

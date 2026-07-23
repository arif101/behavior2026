#!/bin/bash
# Stage-head v4 box bring-up: BEHAVIOR-1K v3.9 (OmniGibson+bddl+eval) for the
# predicate-eval replay pass, plus the light stage-pipeline deps. Trimmed from
# the canonical bringup.sh (v3, 2026-07-07): no openpi, no policy-ckpt/HF
# restore. Carries the box-1-3 fixes: IPv4 preference, CloudFront wheels,
# aria2 assets, XET off, appdata on local disk.
# Run detached:  setsid nohup bash bringup_stage_v4.sh > /workspace/bringup.log 2>&1 < /dev/null &
set -x
export DEBIAN_FRONTEND=noninteractive
export HF_HUB_DISABLE_XET=1

# --- 0. network hardening + base deps -------------------------------------
grep -q "precedence ::ffff:0:0/96 100" /etc/gai.conf 2>/dev/null || echo "precedence ::ffff:0:0/96 100" >> /etc/gai.conf
apt-get update -qq
apt-get install -y ffmpeg kmod git-lfs aria2 libxt6 libglu1-mesa libxrandr2 libxinerama1 libxcursor1 libxi6 && echo STAGE_APT_OK

# --- 1. asset zip via aria2 (parallel, resumable) — runs while installs proceed
mkdir -p /workspace
nohup aria2c -x16 -s16 -k4M --file-allocation=none -c -d /workspace -o b1k_assets.zip \
  "https://huggingface.co/datasets/behavior-1k/zipped-datasets/resolve/main/behavior-1k-assets-3.9.0.zip" \
  > /workspace/aria2.log 2>&1 &
ARIA_PID=$!

# --- 2. light stage-pipeline deps (system python; v3 pipeline ran on these)
python3 -c "import torch" 2>/dev/null || pip3 install -q torch --index-url https://download.pytorch.org/whl/cu124
pip3 install -q pandas pyarrow scipy pillow huggingface_hub && echo STAGE_PYDEPS_OK

# --- 3. conda + BEHAVIOR-1K clone ------------------------------------------
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh \
  && bash /tmp/mc.sh -b -p /root/miniconda3 && echo STAGE_CONDA_OK
export PATH=/root/miniconda3/bin:/root/.local/bin:$PATH
git clone -b v3.9.0 --depth 1 https://github.com/StanfordVL/BEHAVIOR-1K.git /workspace/BEHAVIOR-1K && echo STAGE_CLONE_OK
ln -sfn /workspace/BEHAVIOR-1K /root/BEHAVIOR-1K   # repo tooling uses /root paths

# --- 4. torch wheels from CloudFront (r2/Cloudflare unreliable from some DCs)
mkdir -p /workspace/wheels && cd /workspace/wheels
for f in torch-2.7.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl \
         torchvision-0.22.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl \
         torchaudio-2.7.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl \
         torchcodec-0.5%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl; do
  curl -sO "https://download.pytorch.org/whl/cu128/$f" &
done; wait
for f in *%2B*; do mv "$f" "$(echo $f | sed s/%2B/+/)"; done
sed -i "s|--index-url https://download.pytorch.org/whl/cu\${CUDA_VER_SHORT}|--find-links /workspace/wheels|" /workspace/BEHAVIOR-1K/setup.sh
echo STAGE_WHEELS_OK

# --- 5. extract assets when aria2 finishes (pre-empts setup.sh's flaky downloader)
# CRC-verify BEFORE extracting, and never rm a zip that failed to extract
# (2026-07-23: a corrupt download extracted partially and the unconditional
# rm forced a full re-fetch)
wait $ARIA_PID
for try in 1 2 3; do
  python3 -c "
import zipfile, sys
try:
    bad = zipfile.ZipFile('/workspace/b1k_assets.zip').testzip()
    sys.exit(1 if bad else 0)
except Exception as e:
    print('ZIPCHECK', e); sys.exit(1)" && { echo ZIP_CRC_OK; break; }
  echo "ZIP_CORRUPT_TRY_$try"
  rm -f /workspace/b1k_assets.zip
  aria2c -x16 -s16 -k4M --file-allocation=none -c -d /workspace -o b1k_assets.zip \
    "https://huggingface.co/datasets/behavior-1k/zipped-datasets/resolve/main/behavior-1k-assets-3.9.0.zip" \
    >> /workspace/aria2.log 2>&1
done
mkdir -p /workspace/BEHAVIOR-1K/datasets/behavior-1k-assets
python3 -c 'import zipfile; zipfile.ZipFile("/workspace/b1k_assets.zip").extractall("/workspace/BEHAVIOR-1K/datasets/behavior-1k-assets")' \
  && { echo STAGE_ASSETS_OK; rm -f /workspace/b1k_assets.zip; } \
  || echo ASSETS_EXTRACT_FAILED

# --- 6. BEHAVIOR-1K install ------------------------------------------------
# v3.9 setup.sh: --eval requires --joylo (keep the canonical flag set)
cd /workspace/BEHAVIOR-1K
export PIP_FIND_LINKS=/workspace/wheels
./setup.sh --new-env --omnigibson --bddl --joylo --eval --dataset \
  --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos && echo STAGE_B1K_SETUP_OK
# Isaac texture cache must live on local disk
mkdir -p /root/og_appdata
mv /workspace/BEHAVIOR-1K/OmniGibson/appdata /workspace/BEHAVIOR-1K/OmniGibson/appdata.orig 2>/dev/null
ln -sfn /root/og_appdata /workspace/BEHAVIOR-1K/OmniGibson/appdata && echo STAGE_APPDATA_OK

echo BRINGUP_STAGE_V4_DONE

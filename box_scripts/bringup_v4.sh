#!/bin/bash
# BEHAVIOR-2026 sim-box bring-up v4 (2026-07-27).
#
# Supersedes bringup.sh, which had three bugs and one obsolete strategy. Every fix below cost a
# debug cycle on box 213.173.104.69; none of them are optional.
#
# PREREQUISITE — run this FIRST and reject the box if it fails. Two RunPod boxes were rejected
# on it, and the previously-banked "no /dev/nvidia0" theory was WRONG (a rejected box had the
# nodes). Without `graphics`, the NVIDIA Vulkan userspace is not mounted, vulkaninfo silently
# falls back to llvmpipe, and Isaac refuses to start:
#     tr '\0' '\n' < /proc/1/environ | grep NVIDIA_DRIVER_CAPABILITIES   # must contain "graphics"
#     vulkaninfo --summary | grep deviceName                             # must NOT say llvmpipe
#
# FIXES vs bringup.sh:
#   1. apt-get update BEFORE install — without it EVERY package fails "Unable to locate package"
#      and the script continues silently, so setup.sh falls back to its own slow downloader.
#   2. unzip added (needed for the asset zips, absent from the old list).
#   3. aria2 stage DELETED. Authenticated hf_transfer measured ~700 MB/s vs aria2 unauthenticated
#      at 138 KiB/s on the same link — HF throttles anonymous traffic. 32 GB in ~1 min vs weeks.
#   4. Robot assets must live in datasets/omnigibson-robot-assets/ (VERSION is read from there).
#   5. omnigibson.key must be downloaded — assets are encrypted and Isaac SEGFAULTS without it.
#   6. Task instances are needed in TWO places: merged into behavior-1k-assets/ (scene files) AND
#      standalone at datasets/2026-challenge-task-instances/ (metadata/B100_task_misc.csv).
#
# Install to LOCAL /root — RunPod /workspace is MooseFS (slow, hidden ~125-150 GB quota).
# Expects /root/.hf_token (0600).
set -x
export DEBIAN_FRONTEND=noninteractive
export HF_HUB_DISABLE_XET=1
W=/root/bw

# --- 0. system deps (FIX 1 + FIX 2) ---------------------------------------
apt-get update -qq && apt-get install -y -qq ffmpeg kmod git-lfs unzip libxt6 libglu1-mesa \
  libxrandr2 libxinerama1 libxcursor1 libxi6 libvulkan1 vulkan-tools && echo STAGE_APT_OK
grep -q "precedence ::ffff:0:0/96 100" /etc/gai.conf 2>/dev/null || \
  echo "precedence ::ffff:0:0/96 100" >> /etc/gai.conf

# --- 1. conda + BEHAVIOR-1K ------------------------------------------------
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh \
  && bash /tmp/mc.sh -b -p /root/miniconda3 && echo STAGE_CONDA_OK
export PATH=/root/miniconda3/bin:/root/.local/bin:$PATH
git clone -b v3.9.0 --depth 1 https://github.com/StanfordVL/BEHAVIOR-1K.git $W/BEHAVIOR-1K \
  && echo STAGE_CLONE_OK

# --- 2. torch wheels from CloudFront (r2 unreliable from some DCs) ---------
mkdir -p $W/wheels && cd $W/wheels
for f in torch-2.7.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl \
         torchvision-0.22.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl \
         torchaudio-2.7.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl \
         torchcodec-0.5%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl; do
  curl -sO "https://download.pytorch.org/whl/cu128/$f" &
done; wait
for f in *%2B*; do mv "$f" "$(echo $f | sed s/%2B/+/)"; done
sed -i "s|--index-url https://download.pytorch.org/whl/cu\${CUDA_VER_SHORT}|--find-links $W/wheels|" \
  $W/BEHAVIOR-1K/setup.sh
echo STAGE_WHEELS_OK

# --- 3. install (NO --dataset: we fetch assets ourselves, far faster) ------
cd $W/BEHAVIOR-1K
./setup.sh --new-env --omnigibson --bddl --joylo --eval \
  --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos && echo STAGE_SETUP_OK
PY=/root/miniconda3/envs/behavior/bin/python
$PY -m pip install -q hf_transfer && echo STAGE_HFTRANSFER_OK

# --- 4. assets via authenticated hf_transfer (FIX 3) ----------------------
mkdir -p $W/zips
cat > /root/_fetch.py <<'PY'
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import hf_hub_download
tok = open("/root/.hf_token").read().strip()
for f in ["omnigibson-robot-assets-3.8.2.zip", "behavior-1k-assets-3.9.0.zip",
          "2026-challenge-task-instances.zip"]:
    p = hf_hub_download("behavior-1k/zipped-datasets", f, repo_type="dataset",
                        local_dir="/root/bw/zips", token=tok)
    print("DONE", p, flush=True)
PY
$PY /root/_fetch.py && echo STAGE_FETCH_OK

# --- 5. extract with the EXACT layout the version check wants (FIX 4 + 6) --
D=$W/BEHAVIOR-1K/datasets
mkdir -p $D/behavior-1k-assets $D/omnigibson-robot-assets $D/2026-challenge-task-instances
unzip -q -o $W/zips/behavior-1k-assets-3.9.0.zip       -d $D/behavior-1k-assets
unzip -q -o $W/zips/omnigibson-robot-assets-3.8.2.zip  -d $D/omnigibson-robot-assets
unzip -q -o $W/zips/2026-challenge-task-instances.zip  -d $D/behavior-1k-assets          # scene files
unzip -q -o $W/zips/2026-challenge-task-instances.zip  -d $D/2026-challenge-task-instances  # metadata
echo STAGE_EXTRACT_OK

# --- 6. decryption key — Isaac SEGFAULTS without it (FIX 5) ----------------
$PY -c "from omnigibson.utils.asset_utils import download_key; download_key()" && echo STAGE_KEY_OK
ls -la $D/omnigibson.key $D/omnigibson-robot-assets/VERSION

mkdir -p $W/BEHAVIOR-1K/OmniGibson/appdata /tmp/xdg
echo BRINGUP_V4_DONE

#!/bin/bash
# BEHAVIOR-2026 sim-box bring-up v5 (2026-07-28) — for boxes with Isaac Sim PREINSTALLED.
# Box: 213.173.104.72:45551, RTX PRO 4000 Blackwell 24GB (sm_120), Isaac 5.1.0-rc.19, 200 GB root.
#
# THE GATE — run first, reject the box if it fails. Three boxes have now been tested against it.
#     tr '\0' '\n' < /proc/1/environ | grep NVIDIA_DRIVER_CAPABILITIES   # must contain graphics/all
# THIS BOX PASSES and, unlike the previous two, was verified EMPIRICALLY rather than by env var:
#     /isaac-sim/isaac-sim.compatibility_check.sh  ->  "Graphics API: Vulkan",
#     "NVIDIA RTX PRO 4000 Blackwell | Yes: 0 | 24467 MB", "Driver version [supported]",
#     and critically NO "llvmpipe" anywhere in the log.
#
# Carries FIXES 1-6 from bringup_v4 (all still required):
#   1. apt-get update BEFORE install, else every package fails silently.
#   2. unzip is needed for the asset zips.
#   3. NO aria2 — authenticated hf_transfer ~700 MB/s vs anonymous aria2 138 KiB/s.
#   4. Robot assets must live in datasets/omnigibson-robot-assets/ (VERSION is read from there).
#   5. omnigibson.key must be downloaded — assets are encrypted and Isaac SEGFAULTS without it.
#   6. Task instances go in TWO places: merged into behavior-1k-assets/ AND standalone.
#
# NEW IN v5:
#   7. OMNI_KIT_ALLOW_ROOT=1 — Omniverse Kit refuses to run as root and SEGFAULTS with a
#      misleading core dump. The real message ("cannot be run as the root user without the
#      --allow-root flag") only appears in the log, not on the segfault. Cost one cycle today.
#   8. DISK. Assets need ~66 GB (31.5 zipped + ~35 extracted) on top of ~15 GB of conda+env.
#      The FIRST pod for this box had a 20 GB root and would have needed a /workspace split;
#      it was rebuilt with a 200 GB root, so EVERYTHING now lives on local /root. Keep it that
#      way: /workspace is MooseFS and a conda env is ~100k small files, where MooseFS is
#      punishing on import. If you ever land on a small-root pod again, move only $W (not
#      miniconda3) to /workspace.
#      NOTE the 17 GB /isaac-sim does NOT consume the overlay — it is in the read-only image layer.
#   9. Zips are deleted after extraction — they are 31.5 GB of pure duplicate once unpacked.
#  10. Blackwell is sm_120 and needs CUDA >= 12.8. The cu128 wheels below are already correct;
#      do NOT downgrade them. Verified: Isaac's bundled torch is 2.7.0+cu128 with sm_120 and
#      compute_120 in its arch list.
set -x
export DEBIAN_FRONTEND=noninteractive
export HF_HUB_DISABLE_XET=1
export OMNI_KIT_ALLOW_ROOT=1                       # FIX 7
W=/root/bw                                          # FIX 8: 200 GB local root, no MooseFS
mkdir -p $W

# --- 0. system deps (FIX 1 + FIX 2) ---------------------------------------
apt-get update -qq && apt-get install -y -qq ffmpeg kmod git-lfs unzip libxt6 libglu1-mesa \
  libxrandr2 libxinerama1 libxcursor1 libxi6 libvulkan1 vulkan-tools && echo STAGE_APT_OK
grep -q "precedence ::ffff:0:0/96 100" /etc/gai.conf 2>/dev/null || \
  echo "precedence ::ffff:0:0/96 100" >> /etc/gai.conf

# --- 1. conda + BEHAVIOR-1K (both on local /root) --------------------------
if [ ! -x /root/miniconda3/bin/conda ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh \
    && bash /tmp/mc.sh -b -p /root/miniconda3 && echo STAGE_CONDA_OK
fi
export PATH=/root/miniconda3/bin:/root/.local/bin:$PATH
[ -d $W/BEHAVIOR-1K ] || git clone -b v3.9.0 --depth 1 \
  https://github.com/StanfordVL/BEHAVIOR-1K.git $W/BEHAVIOR-1K
echo STAGE_CLONE_OK

# --- 2. torch wheels from CloudFront (r2 unreliable from some DCs) ---------
mkdir -p $W/wheels && cd $W/wheels
for f in torch-2.7.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl \
         torchvision-0.22.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl \
         torchaudio-2.7.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl \
         torchcodec-0.5%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl; do
  [ -f "$(echo $f | sed s/%2B/+/)" ] || curl -sO "https://download.pytorch.org/whl/cu128/$f" &
done; wait
for f in *%2B*; do [ -e "$f" ] && mv "$f" "$(echo $f | sed s/%2B/+/)"; done
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
unzip -q -o $W/zips/2026-challenge-task-instances.zip  -d $D/behavior-1k-assets             # scene files
unzip -q -o $W/zips/2026-challenge-task-instances.zip  -d $D/2026-challenge-task-instances  # metadata
echo STAGE_EXTRACT_OK

# --- 6. decryption key — Isaac SEGFAULTS without it (FIX 5) ----------------
$PY -c "from omnigibson.utils.asset_utils import download_key; download_key()" && echo STAGE_KEY_OK
ls -la $D/omnigibson.key $D/omnigibson-robot-assets/VERSION

# --- 7. reclaim the 31.5 GB of zips (FIX 9) --------------------------------
rm -rf $W/zips && echo STAGE_ZIPS_PURGED
df -h / /workspace | grep -v Filesystem

mkdir -p $W/BEHAVIOR-1K/OmniGibson/appdata /tmp/xdg
echo BRINGUP_V5_DONE

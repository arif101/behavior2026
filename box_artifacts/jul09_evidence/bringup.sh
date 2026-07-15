#!/bin/bash
# BEHAVIOR-2026 canonical box bring-up (v3, 2026-07-07).
# Prereq: acceptance gate passed (GPU + vulkaninfo + bandwidth + dual-stack CDN).
# Carries every fix learned on boxes 1-3: IPv4 preference, CloudFront wheel routing,
# aria2 parallel asset fetch, MooseFS mitigations (XET off, appdata local, ckpts local).
# Expects: /root/.hf_token (write-scoped) for artifact restore. Logs: /workspace/bringup.log
set -x
export DEBIAN_FRONTEND=noninteractive
export HF_HUB_DISABLE_XET=1

# --- 0. network hardening + base deps -------------------------------------
grep -q "precedence ::ffff:0:0/96 100" /etc/gai.conf 2>/dev/null || echo "precedence ::ffff:0:0/96 100" >> /etc/gai.conf
apt-get install -y ffmpeg kmod git-lfs aria2 libxt6 libglu1-mesa libxrandr2 libxinerama1 libxcursor1 libxi6 && echo STAGE_APT_OK

# --- 1. asset zip via aria2 (parallel, resumable) — runs while installs proceed
mkdir -p /workspace
nohup aria2c -x16 -s16 -k4M --file-allocation=none -c -d /workspace -o b1k_assets.zip \
  "https://huggingface.co/datasets/behavior-1k/zipped-datasets/resolve/main/behavior-1k-assets-3.9.0.zip" \
  > /workspace/aria2.log 2>&1 &
ARIA_PID=$!

# --- 2. conda + BEHAVIOR-1K clone ------------------------------------------
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh \
  && bash /tmp/mc.sh -b -p /root/miniconda3 && echo STAGE_CONDA_OK
export PATH=/root/miniconda3/bin:/root/.local/bin:$PATH
git clone -b v3.9.0 --depth 1 https://github.com/StanfordVL/BEHAVIOR-1K.git /workspace/BEHAVIOR-1K && echo STAGE_CLONE_OK

# --- 3. torch wheels from CloudFront (r2/Cloudflare unreliable from some DCs)
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

# --- 4. extract assets when aria2 finishes (pre-empts setup.sh's flaky downloader)
wait $ARIA_PID
mkdir -p /workspace/BEHAVIOR-1K/datasets/behavior-1k-assets
python3 -c 'import zipfile; zipfile.ZipFile("/workspace/b1k_assets.zip").extractall("/workspace/BEHAVIOR-1K/datasets/behavior-1k-assets")' && echo STAGE_ASSETS_OK

# --- 5. BEHAVIOR-1K install ------------------------------------------------
cd /workspace/BEHAVIOR-1K
export PIP_FIND_LINKS=/workspace/wheels
./setup.sh --new-env --omnigibson --bddl --joylo --eval --dataset \
  --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos && echo STAGE_B1K_SETUP_OK
# MooseFS mitigation: Isaac texture cache must live on local disk
mkdir -p /root/og_appdata
mv /workspace/BEHAVIOR-1K/OmniGibson/appdata /workspace/BEHAVIOR-1K/OmniGibson/appdata.orig 2>/dev/null
ln -sfn /root/og_appdata /workspace/BEHAVIOR-1K/OmniGibson/appdata && echo STAGE_APPDATA_OK

# --- 6. openpi fork --------------------------------------------------------
curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=/root/.local/bin:$PATH
git clone -b behavior https://github.com/wensi-ai/openpi.git /workspace/openpi
cd /workspace/openpi && git submodule update --init --recursive
sed -i "s|download-r2\.pytorch\.org|download.pytorch.org|g" uv.lock
GIT_LFS_SKIP_SMUDGE=1 uv sync && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e . && echo STAGE_OPENPI_OK

# --- 7. restore artifacts from HF (norm stats save a 2h recompute; ckpt for probes)
/root/miniconda3/envs/behavior/bin/python - << 'PY'
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import snapshot_download
tok = open("/root/.hf_token").read().strip() if os.path.exists("/root/.hf_token") else None
snapshot_download("arif101/behavior2026-artifacts", token=tok, local_dir="/workspace/hf_restore")
print("STAGE_HF_RESTORE_OK")
PY
mkdir -p /workspace/openpi/outputs/assets
cp -r /workspace/hf_restore/assets/pi05_b1k_ours /workspace/openpi/outputs/assets/ 2>/dev/null
ln -sfn pi05_b1k_ours /workspace/openpi/outputs/assets/pi05_b1k_ours_lora
mkdir -p /root/g2_ckpts/pi05_b1k_ours_lora/g2_radio_lora
cp -r /workspace/hf_restore/ckpts/g2_radio_lora_5000 /root/g2_ckpts/pi05_b1k_ours_lora/g2_radio_lora/5000 2>/dev/null

# --- 8. dataset subset (radio) + rawdata (decoder work) ---------------------
/root/miniconda3/envs/behavior/bin/python - << 'PY'
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import snapshot_download
snapshot_download("behavior-1k/2026-challenge-demos", repo_type="dataset",
                  local_dir="/workspace/2026-challenge-demos", allow_patterns=["meta/**"])
import pandas as pd, glob
files = sorted(glob.glob("/workspace/2026-challenge-demos/meta/episodes/**/*.parquet", recursive=True))
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
sel = df[df["tasks"].apply(lambda t: "turning_on_radio" in list(t))]
pats = set()
cams = ["observation.rgb.zed_link_camera_0","observation.rgb.left_realsense_link_camera_0","observation.rgb.right_realsense_link_camera_0",
        "observation.depth_linear.zed_link_camera_0","observation.depth_linear.left_realsense_link_camera_0","observation.depth_linear.right_realsense_link_camera_0"]
for _, r in sel.iterrows():
    pats.add("data/chunk-%03d/file-%03d.parquet" % (r["data/chunk_index"], r["data/file_index"]))
    for cam in cams:
        pats.add("videos/%s/chunk-%03d/file-%03d.mp4" % (cam, r[f"videos/{cam}/chunk_index"], r[f"videos/{cam}/file_index"]))
snapshot_download("behavior-1k/2026-challenge-demos", repo_type="dataset",
                  local_dir="/workspace/2026-challenge-demos", allow_patterns=sorted(pats))
snapshot_download("behavior-1k/2026-challenge-rawdata", repo_type="dataset",
                  local_dir="/workspace/rawdata", allow_patterns=["task-0000/**"])
print("STAGE_DATA_OK")
PY

echo BRINGUP_DONE

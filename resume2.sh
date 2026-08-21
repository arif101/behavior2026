#!/bin/bash
# Resume bringup.sh after the driver shell died at stage-3's bare `wait` (aria2 survived).
# Completed + idempotent: STAGE_APT/CONDA/CLONE. This finishes stages 3-5 verbatim.
set -x
export DEBIAN_FRONTEND=noninteractive
export PATH=/root/miniconda3/bin:/root/.local/bin:$PATH

# --- stage 1 tail: wait for aria2 to finish the asset zip ---------------------
while pgrep -x aria2c >/dev/null; do sleep 30; done
tail -3 /workspace/aria2.log
grep -q "download completed" /workspace/aria2.log || {
  # aria2 died incomplete -> resume it (resumable: -c)
  aria2c -x16 -s16 -k4M --file-allocation=none -c -d /workspace -o b1k_assets.zip \
    "https://huggingface.co/datasets/behavior-1k/zipped-datasets/resolve/main/behavior-1k-assets-3.9.0.zip" \
    >> /workspace/aria2.log 2>&1
}
echo STAGE_ARIA_OK

# --- stage 3 tail: wheel integrity + rename + setup.sh sed --------------------
cd /workspace/wheels
for f in torch-2.7.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl \
         torchvision-0.22.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl \
         torchaudio-2.7.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl \
         torchcodec-0.5%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl; do
  [ -f "$f" ] || [ -f "$(echo $f | sed s/%2B/+/)" ] || curl -sO "https://download.pytorch.org/whl/cu128/$f"
done
for f in *%2B*; do [ -f "$f" ] && mv "$f" "$(echo $f | sed s/%2B/+/)"; done
for w in *.whl; do
  python3 -c "import zipfile,sys; sys.exit(0 if zipfile.ZipFile('$w').testzip() is None else 1)" || {
    echo "corrupt wheel $w -- refetching"
    u=$(echo $w | sed 's/+/%2B/')
    rm -f "$w"; curl -sO "https://download.pytorch.org/whl/cu128/$u"; mv "$u" "$w"
  }
done
sed -i "s|--index-url https://download.pytorch.org/whl/cu\${CUDA_VER_SHORT}|--find-links /workspace/wheels|" /workspace/BEHAVIOR-1K/setup.sh
echo STAGE_WHEELS_OK

# --- stage 4: extract assets --------------------------------------------------
mkdir -p /workspace/BEHAVIOR-1K/datasets/behavior-1k-assets
python3 -c 'import zipfile; zipfile.ZipFile("/workspace/b1k_assets.zip").extractall("/workspace/BEHAVIOR-1K/datasets/behavior-1k-assets")' && echo STAGE_ASSETS_OK
rm -f /root/bw/b1k_assets.zip /workspace/b1k_assets.zip

# --- stage 5: BEHAVIOR-1K install --------------------------------------------
cd /workspace/BEHAVIOR-1K
export PIP_FIND_LINKS=/workspace/wheels
./setup.sh --new-env --omnigibson --bddl --joylo --eval --dataset \
  --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos && echo STAGE_B1K_SETUP_OK
mkdir -p /root/og_appdata
mv /workspace/BEHAVIOR-1K/OmniGibson/appdata /workspace/BEHAVIOR-1K/OmniGibson/appdata.orig 2>/dev/null
ln -sfn /root/og_appdata /workspace/BEHAVIOR-1K/OmniGibson/appdata && echo STAGE_APPDATA_OK
/root/miniconda3/envs/behavior/bin/python -c "import omnigibson; print('OG_IMPORT_OK', omnigibson.__version__)"
echo RESUME2_DONE

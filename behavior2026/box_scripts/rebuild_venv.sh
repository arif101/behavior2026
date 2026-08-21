#!/bin/bash
# Clean venv rebuild with a PINNED, mutually-consistent torch stack.
#
# The problem this solves: torchcodec ships prebuilt .so files compiled against a specific torch
# ABI. Installing a mismatched pair fails at dlopen with e.g.
#     libtorchcodec_core4.so: undefined symbol: torch_dtype_float4_e2m1fn_x2
# `float4_e2m1fn_x2` is a torch 2.8 symbol; torch 2.7.1 does not have it (verified:
# hasattr(torch, "float4_e2m1fn_x2") is False). So torchcodec 0.5 CANNOT work on torch 2.7.
#
# Worse, incremental `pip install --force-reinstall` did not fix it: the .so files kept their
# original timestamps through three reinstall attempts, because pip considered the requirement
# satisfied and never rewrote them. Layering reinstalls onto a broken env wastes more time than
# rebuilding it.
#
# Compatibility (torchcodec -> torch):  0.2->2.5, 0.3->2.6, 0.4->2.7, 0.5->2.8
# openpi targets torch 2.7, so torchcodec 0.4.0 is the correct pin.
#
# ffmpeg system libs are also required — torchcodec probes libtorchcodec_core{8..4}.so and each
# needs the matching libavutil. Ubuntu 22.04 ships ffmpeg 4 (libavutil.so.56), which core4 uses.
set -euo pipefail

TORCH=2.7.1
TVISION=0.22.1
TAUDIO=2.7.1
TCODEC=0.4.0
IDX=https://download.pytorch.org/whl/cu128

export DEBIAN_FRONTEND=noninteractive
apt-get install -y -qq ffmpeg libavutil-dev libavcodec-dev libavformat-dev libswscale-dev
ldconfig
echo "STAGE_FFMPEG_OK $(ls /usr/lib/x86_64-linux-gnu/libavutil.so.* 2>/dev/null | head -1)"

cd /root/openpi
export PATH=/root/.local/bin:$PATH
rm -rf .venv
uv venv --python 3.11 .venv
. .venv/bin/activate

# Torch stack FIRST and pinned, so nothing downstream can pull a mismatched version.
uv pip install --index-url "$IDX" \
  "torch==$TORCH" "torchvision==$TVISION" "torchaudio==$TAUDIO" "torchcodec==$TCODEC"
echo STAGE_TORCH_OK

uv pip install "jax[cuda12]"
uv pip install -e .
uv pip install hf_xet
echo STAGE_DEPS_OK

python - <<'PY'
import torch, torchcodec, jax
print("torch     ", torch.__version__)
print("torchcodec", torchcodec.__version__)
print("jax devices", jax.devices())
# The real test: force the decoder to load its shared libs.
from torchcodec.decoders import VideoDecoder  # noqa: F401
print("VideoDecoder import OK")
PY
echo VENV_REBUILD_DONE

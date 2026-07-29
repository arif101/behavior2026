#!/bin/bash
# Install openpi + JAX on the SIM box so we can serve ckpt 49999 to the BEHAVIOR eval harness.
#
# Context: the `behavior` conda env holds OmniGibson + torch(cu128) and must NOT be disturbed --
# the eval client runs there. openpi needs JAX, so it gets its OWN venv. Two processes, one box:
#   policy server (this env, JAX)  <--websocket-->  eval client (behavior env, OmniGibson)
#
# BLACKWELL: this GPU is sm_120 (compute cap 12.0) and needs CUDA >= 12.8. Isaac's bundled torch
# is 2.7.0+cu128 with sm_120/compute_120 in its arch list, so the driver side is fine; JAX must
# also be a build new enough to emit sm_120. Install the current jax[cuda12] and VERIFY the device
# is visible before declaring success -- a silent CPU fallback would make the eval meaninglessly
# slow rather than fail loudly.
#
# 24GB card shared with Isaac: params are 12.4GB in bf16, so cap JAX's preallocation. Without this
# JAX grabs 75% of the card at import and Isaac gets nothing.
# set -e is NOT optional here. The first version of this script used `python3 -m venv`, which fails
# on this image because ensurepip is absent (needs apt python3.12-venv) -- and then EVERY later step
# failed with "No module named pip" while the script cheerfully printed INSTALL_OPENPI_DONE at the
# end. A success marker on a failed install is how you end up "running" an eval that is not running.
set -ex
export HF_XET_HIGH_PERFORMANCE=1          # see bringup_v5.sh FIX 11: ~48x vs the deprecated path
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.55

W=/root/openpi
[ -d $W ] || git clone --depth 1 https://github.com/Physical-Intelligence/openpi.git $W
cd $W && echo STAGE_CLONE_OK

# conda, not venv: it ships pip, and 3.11 matches what openpi and the behavior env both use.
export PATH=/root/miniconda3/bin:$PATH
conda create -y -q -n openpi python=3.11 >/dev/null
PY=/root/miniconda3/envs/openpi/bin/python
$PY -m pip install -q --upgrade pip setuptools wheel && echo STAGE_PIP_OK

# openpi itself (editable, so our patches apply) + JAX with CUDA.
$PY -m pip install -q -e . 2>&1 | tail -5
$PY -m pip install -q "jax[cuda12]" && echo STAGE_JAX_OK

# HARD GATE: refuse to continue on a CPU-only JAX. A silent fallback here would look like a
# working eval that runs 100x slow and produces numbers we would wrongly trust.
$PY - <<'PYEOF'
import sys
import jax
d = jax.devices()
print("jax", jax.__version__, "devices:", d)
if not any(dev.platform == "gpu" for dev in d):
    print("FATAL: JAX sees no GPU -- refusing to proceed (CPU fallback would silently ruin the eval)")
    sys.exit(1)
print("STAGE_JAX_GPU_OK")
PYEOF

echo INSTALL_OPENPI_DONE

#!/bin/bash
# Phase-A eval prep — run AFTER bringup.sh stages 0-5 on a SIM box (RT cores + vulkaninfo
# showing the GPU; see .claude/skills/setup-sim-box). Mirrors g3_evalprep.sh but restores the
# Phase-A A/B checkpoints (contact weights ON vs OFF) instead of the G3 arms.
# Prereq: /root/.hf_token. Logs everything; grep for STAGE_.*_OK.
#
# The two arms share a config (pi05_phaseA_point) and therefore ONE set of norm stats — this
# is what makes the comparison clean, so do not regenerate them per arm.
#
# POINT SOURCE: the primary A/B runs on ORACLE points (g3_deep_labels), deliberately. The
# question under test is whether contact oversampling raises the closed-loop floor; feeding
# predicted points would confound that with grounding error. Swap to the learned grounder only
# as a follow-up, after the BC question is answered.
#
# NOTE (grounding): only v0.5 is on HF. The v0.6 gate fine-tune that fixed thawing
# (124.6px -> 4.56px) lived on the spun-down G3 eval box and was never backed up. It is ~13 min
# to re-derive (grounding/mt_train.py, gate split) and is NOT needed for the oracle-point A/B.
set -x
export HF_HUB_DISABLE_XET=1

ARMS="phaseA_weighted phaseA_uniform"

# --- 6a. our openpi fork (AdaLN point-conditioning; NOT stock wensi-ai) -----
curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=/root/.local/bin:$PATH
/root/miniconda3/envs/behavior/bin/python - << 'PY'
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import hf_hub_download
tok = open("/root/.hf_token").read().strip()
p = hf_hub_download("arif101/behavior2026-artifacts", "code/openpi_fork_adaln_src.tar.gz",
                    repo_type="model", token=tok, local_dir="/root/stage")
print(p)
PY
tar xzf /root/stage/code/openpi_fork_adaln_src.tar.gz -C /root
mv /root/openpi /root/openpi_adaln
cd /root/openpi_adaln
sed -i "s|download-r2\.pytorch\.org|download.pytorch.org|g" uv.lock 2>/dev/null
GIT_LFS_SKIP_SMUDGE=1 uv sync && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e . && echo STAGE_FORK_OK

# --- 7a. Phase-A checkpoints + norm stats + grounding + eval labels ---------
/root/miniconda3/envs/behavior/bin/python - << 'PY'
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download
tok = open("/root/.hf_token").read().strip()
snapshot_download("arif101/behavior2026-artifacts", token=tok, local_dir="/root/hf_restore",
                  allow_patterns=[
                      "ckpts/phaseA/*/*/params/**",       # both arms, final step
                      "assets_phaseA/pi05_phaseA_point/**",
                      "ckpts/grounding_v05/**",           # see NOTE below re: v0.6
                      "g3_pipeline/*.json",               # task_targets, maps
                      "g3_deep_labels/**",                # oracle points (primary eval source)
                  ])
print("STAGE_PHASEA_RESTORE_OK")
PY
mkdir -p /root/openpi_adaln/outputs/assets
cp -r /root/hf_restore/assets_phaseA/* /root/openpi_adaln/outputs/assets/
mkdir -p /root/phaseA_ckpts && cp -r /root/hf_restore/ckpts/phaseA/* /root/phaseA_ckpts/
cp -r /root/hf_restore/ckpts/grounding_v05 /root/grounding_v05 2>/dev/null
cp -r /root/hf_restore/g3_pipeline /root/g3_pipeline
cp -r /root/hf_restore/g3_deep_labels /root/g3_deep_labels
# create_trained_policy reads <ckpt>/assets first, so mirror norm stats into each arm/step
for arm in $ARMS; do
  for step in /root/phaseA_ckpts/$arm/*/; do
    [ -d "$step" ] || continue
    mkdir -p "$step/assets"
    cp -r /root/openpi_adaln/outputs/assets/pi05_phaseA_point/* "$step/assets/" 2>/dev/null
  done
done
rm -rf /root/hf_restore /root/stage
echo STAGE_PHASEA_ARTIFACTS_OK

# --- 8a. serve smoke: load EACH arm, one dummy infer ------------------------
# Both arms must load. A silent norm-stats mismatch shows up here, not 200 episodes later.
cd /root/openpi_adaln
for arm in $ARMS; do
  CK=$(ls -d /root/phaseA_ckpts/$arm/*/ 2>/dev/null | sort -t/ -k5 -n | tail -1)
  [ -n "$CK" ] || { echo "STAGE_SERVE_SMOKE_FAIL missing $arm"; continue; }
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python - "$CK" "$arm" << 'PY'
import sys
import numpy as np
import openpi.training.config as _config
from openpi.policies import policy_config as _policy_config
ck, arm = sys.argv[1], sys.argv[2]
cfg = _config.get_config("pi05_phaseA_point")
policy = _policy_config.create_trained_policy(cfg, ck, default_prompt="smoke")
obs = {f"observation/image_{k}": np.zeros((224, 224, 3), np.uint8) for k in range(3)}
obs.update({"observation/state": np.zeros(61, np.float32), "prompt": "smoke",
            "target_points": np.zeros((2, 3), np.float32),
            "target_points_mask": np.zeros(2, bool)})
a = policy.infer(obs)["actions"]
print("STAGE_SERVE_SMOKE_OK", arm, np.asarray(a).shape)
PY
done
echo PHASEA_EVALPREP_DONE

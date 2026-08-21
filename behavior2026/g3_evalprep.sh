#!/bin/bash
# G3 eval prep — run AFTER bringup.sh stages 0-5 (base deps, conda, B1K+OmniGibson+eval).
# Replaces bringup stages 6-8 for the G3 gate-eval box: OUR AdaLN fork + G3 checkpoints
# + grounding v0.5 + gate labels/maps. Targeted HF pulls only (artifacts repo is huge now).
# Prereq: /root/.hf_token. Logs everything; grep for STAGE_.*_OK.
set -x
export HF_HUB_DISABLE_XET=1

# --- 6g3. our openpi fork (AdaLN point-conditioning; NOT stock wensi-ai) ----
curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=/root/.local/bin:$PATH
/root/miniconda3/envs/behavior/bin/python - << 'PY'
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import hf_hub_download
tok = open("/root/.hf_token").read().strip()
p = hf_hub_download("arif101/behavior2026-artifacts", "code/openpi_fork_adaln_src.tar.gz",
                    repo_type="model", token=tok, local_dir="/workspace/stage")
print(p)
PY
mkdir -p /workspace && tar xzf /workspace/stage/code/openpi_fork_adaln_src.tar.gz -C /workspace
mv /workspace/openpi /workspace/openpi_adaln
cd /workspace/openpi_adaln
sed -i "s|download-r2\.pytorch\.org|download.pytorch.org|g" uv.lock 2>/dev/null
GIT_LFS_SKIP_SMUDGE=1 uv sync && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e . && echo STAGE_FORK_OK

# --- 7g3. G3 checkpoints + norm stats + grounding + gate labels -------------
/root/miniconda3/envs/behavior/bin/python - << 'PY'
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download
tok = open("/root/.hf_token").read().strip()
snapshot_download("arif101/behavior2026-artifacts", token=tok, local_dir="/workspace/hf_restore",
                  allow_patterns=[
                      "ckpts/g3/*/29999/params/**",        # final params, all trained arms
                      "assets_g3/pi05_g3_*/**",            # norm stats per config
                      "ckpts/grounding_v05/ckpt_v05_dinov3.pt",
                      "ckpts/grounding_v05/code/**",       # archived matching code
                      "g3_pipeline/*.json",                # maps, task_targets, episodes
                      "g3_deep_labels/**",                 # oracle points at eval (arm 3)
                      "g3_pipeline/gate_grounding/gate_grounding_results.json",
                  ])
print("STAGE_G3_RESTORE_OK")
PY
mkdir -p /workspace/openpi_adaln/outputs/assets
cp -r /workspace/hf_restore/assets_g3/* /workspace/openpi_adaln/outputs/assets/
mkdir -p /root/g3_ckpts && cp -r /workspace/hf_restore/ckpts/g3/* /root/g3_ckpts/
cp -r /workspace/hf_restore/ckpts/grounding_v05 /root/grounding_v05
cp -r /workspace/hf_restore/g3_pipeline /root/g3_pipeline
cp -r /workspace/hf_restore/g3_deep_labels /root/g3_deep_labels
# norm stats also mirrored into each ckpt dir (create_trained_policy reads ckpt/assets first)
for arm in lang_point lang taskid; do
  if [ -d /root/g3_ckpts/$arm/29999 ]; then
    mkdir -p /root/g3_ckpts/$arm/29999/assets
    cp -r /workspace/openpi_adaln/outputs/assets/pi05_g3_$arm/* /root/g3_ckpts/$arm/29999/assets/ 2>/dev/null
  fi
done
rm -rf /workspace/hf_restore /workspace/stage
echo STAGE_G3_ARTIFACTS_OK

# --- 8g3. serve smoke: load lang_point params on GPU1, one dummy infer ------
cd /workspace/openpi_adaln
CUDA_VISIBLE_DEVICES=1 .venv/bin/python - << 'PY'
import numpy as np
import openpi.training.config as _config
from openpi.policies import policy_config as _policy_config
cfg = _config.get_config("pi05_g3_lang_point")
policy = _policy_config.create_trained_policy(
    cfg, "/root/g3_ckpts/lang_point/29999", default_prompt="smoke")
obs = {f"observation/image_{k}": np.zeros((224, 224, 3), np.uint8) for k in range(3)}
obs.update({"observation/state": np.zeros(61, np.float32), "prompt": "smoke",
            "target_points": np.zeros((2, 3), np.float32),
            "target_points_mask": np.zeros(2, bool)})
a = policy.infer(obs)["actions"]
print("STAGE_SERVE_SMOKE_OK", np.asarray(a).shape)
PY
echo G3_EVALPREP_DONE

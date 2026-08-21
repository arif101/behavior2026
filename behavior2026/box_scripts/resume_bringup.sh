#!/bin/bash
# Resume bring-up after the asset extraction was redirected to LOCAL disk.
# Covers bringup.sh stage 5 only (BEHAVIOR-1K install + appdata symlink); stages 6-8 are
# obsolete and are replaced by phaseA_evalprep.sh, which installs OUR AdaLN fork instead of
# stock wensi-ai/openpi.
set -x
# wait for the local extraction to finish (no new files for 60s => done)
prev=0
while true; do
  n=$(find /root/b1k_datasets -type f 2>/dev/null | wc -l)
  if [ "$n" = "$prev" ] && [ "$n" -gt 1000 ]; then break; fi
  prev=$n; sleep 60
done
echo "STAGE_ASSETS_OK extracted $(find /root/b1k_datasets -type f | wc -l) files, $(du -sm /root/b1k_datasets | cut -f1) MB"
rm -f /workspace/b1k_assets.zip   # MooseFS has a hidden per-pod quota (~125-150GB)

cd /workspace/BEHAVIOR-1K
export PIP_FIND_LINKS=/workspace/wheels
export PATH=/root/miniconda3/bin:/root/.local/bin:$PATH
./setup.sh --new-env --omnigibson --bddl --joylo --eval --dataset \
  --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos && echo STAGE_B1K_SETUP_OK
# Isaac texture cache must live on local disk (MooseFS mitigation, same reason as the datasets)
mkdir -p /root/og_appdata
mv /workspace/BEHAVIOR-1K/OmniGibson/appdata /workspace/BEHAVIOR-1K/OmniGibson/appdata.orig 2>/dev/null
ln -sfn /root/og_appdata /workspace/BEHAVIOR-1K/OmniGibson/appdata && echo STAGE_APPDATA_OK
/root/miniconda3/envs/behavior/bin/python -c "import omnigibson; print('OG_IMPORT_OK', omnigibson.__version__)"
echo RESUME_BRINGUP_DONE

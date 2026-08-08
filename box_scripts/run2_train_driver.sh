#!/bin/bash
# RUN-2 DISK-SAFE TRAINING DRIVER (4xA100 trainer) — skill step-5 pattern.
# - launches pi05_radio_run2 via scripts/b1k/train_b1k.py (the create_b1k_data_loader
#   entrypoint — the ONLY one the stage-oversample patch lives in) with
#   B1K_STAGE_OVERSAMPLE=8
# - sidecar pruner every 10 min: strip train_state/ from every COMMITTED checkpoint
#   except the newest (keeps ~1 full + N params-only; a full ckpt is ~43GB)
# - auto-resume up to 2 times if the trainer dies before the final step
# - finalize: upload final params + assets to HF (arif101/b26-run2-params), print
#   RUN2_TRAIN_DONE
# Usage: bash run2_train_driver.sh   (idempotent-ish; safe to re-run after a box reboot)
set -u
FORK=/root/openpi_fork
EXP=radio_run2
CFG=pi05_radio_run2
CKDIR=$FORK/outputs/checkpoints/$CFG/$EXP
LOG=/root/run2_train.log
FINAL_STEP=49999
RESUMES=0

launch() {
  cd $FORK
  setsid nohup env B1K_STAGE_OVERSAMPLE=8 XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 \
    .venv/bin/python scripts/b1k/train_b1k.py $CFG \
    --exp_name $EXP --keep-period 10000 --no-wandb-enabled $1 \
    >> $LOG 2>&1 &
  TRAIN_PID=$!
  echo $TRAIN_PID > /root/run2_train.pid
  echo "[driver] launched trainer pid=$TRAIN_PID args='$1'" | tee -a /root/run2_driver.log
}

committed_steps() {
  ls -d $CKDIR/*/ 2>/dev/null | grep -v "orbax-checkpoint-tmp" \
    | sed "s|$CKDIR/||; s|/||" | grep -E "^[0-9]+$" | sort -n
}

prune() {
  local steps newest s
  steps=$(committed_steps)
  newest=$(echo "$steps" | tail -1)
  for s in $steps; do
    if [ "$s" != "$newest" ] && [ -d "$CKDIR/$s/train_state" ]; then
      rm -rf "$CKDIR/$s/train_state"
      echo "[driver] pruned train_state of step $s (kept params)" | tee -a /root/run2_driver.log
    fi
  done
  df -h / | tail -1 | awk '{print "[driver] disk free: "$4}' >> /root/run2_driver.log
}

finalize() {
  echo "[driver] finalizing: uploading params + assets to HF" | tee -a /root/run2_driver.log
  python3 - <<'PYEOF'
import os
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
from huggingface_hub import HfApi
tok = open("/root/.hf_token").read().strip()
api = HfApi(token=tok)
REPO = "arif101/b26-run2-params"
api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)
ck = "/root/openpi_fork/outputs/checkpoints/pi05_radio_run2/radio_run2"
steps = sorted(int(d) for d in os.listdir(ck) if d.isdigit())
final = steps[-1]
api.upload_folder(folder_path=f"{ck}/{final}/params", path_in_repo="params",
                  repo_id=REPO, repo_type="model")
api.upload_folder(folder_path="/root/openpi_fork/outputs/assets/pi05_radio_run2",
                  path_in_repo="assets", repo_id=REPO, repo_type="model")
info = api.repo_info(REPO, repo_type="model", files_metadata=True)
print(f"RUN2_PARAMS_UPLOADED step={final} files={len(info.siblings)} "
      f"total={sum(s.size or 0 for s in info.siblings)/1e9:.2f}GB", flush=True)
PYEOF
}

launch ""
while true; do
  sleep 600
  if ! kill -0 "$(cat /root/run2_train.pid)" 2>/dev/null; then
    last=$(committed_steps | tail -1)
    last=${last:-0}
    if [ "$last" -ge "$FINAL_STEP" ]; then
      prune
      finalize
      echo "RUN2_TRAIN_DONE final_step=$last" | tee -a /root/run2_driver.log
      exit 0
    fi
    if [ "$RESUMES" -ge 2 ]; then
      echo "RUN2_TRAIN_FAILED last_step=$last resumes_exhausted" | tee -a /root/run2_driver.log
      exit 1
    fi
    RESUMES=$((RESUMES + 1))
    echo "[driver] trainer died at step $last — resume $RESUMES/2" | tee -a /root/run2_driver.log
    launch "--resume"
  fi
  prune
done

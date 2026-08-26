#!/bin/bash
# Training driver for pi05_t62_ctx on the trainer box. Clone of box_scripts/run2_train_driver.sh
# with one change: every committed checkpoint's params are uploaded to HF AS THEY LAND
# ("checkpoints leave the box ALWAYS" — the 08-08 trainer termination lost a run to
# finalize-only-at-end). Resume <= 2 times; prune train_state of older steps to save disk.
# Usage: setsid nohup bash task62/train_driver_t62.sh > /root/t62_driver.log 2>&1 &
set -u
FORK=/root/openpi_fork
PY=/root/miniconda3/envs/openpi/bin/python
EXP=t62_ctx
CFG=pi05_t62_ctx
CKDIR=$FORK/outputs/checkpoints/$CFG/$EXP
LOG=/root/t62_train.log
DLOG=/root/t62_driver.log
FINAL_STEP=29999
REPO=arif101/b26-t62-params
RESUMES=0
export HF_XET_HIGH_PERFORMANCE=1

launch() {
  cd $FORK
  setsid nohup env XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 \
    $PY scripts/b1k/train_b1k.py $CFG --exp_name $EXP --keep-period 5000 --no-wandb-enabled $1 \
    >> $LOG 2>&1 &
  echo $! > /root/t62_train.pid
  echo "[driver] $(date -u +%FT%TZ) launched trainer pid=$(cat /root/t62_train.pid) args='$1'" | tee -a $DLOG
}
committed_steps() {
  ls -d $CKDIR/*/ 2>/dev/null | grep -v "orbax-checkpoint-tmp" | sed "s|$CKDIR/||; s|/||" | grep -E "^[0-9]+$" | sort -n
}
upload_step() {   # $1 = step; idempotent via marker file
  local s=$1
  [ -f "$CKDIR/$s/.uploaded" ] && return 0
  [ -d "$CKDIR/$s/params" ] || return 0
  $PY - "$s" <<'PYEOF' && touch "$CKDIR/$s/.uploaded"
import os, sys
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
from huggingface_hub import HfApi
s = sys.argv[1]; tok = open("/root/.hf_token").read().strip(); api = HfApi(token=tok)
REPO = "arif101/b26-t62-params"; api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)
ck = f"/root/openpi_fork/outputs/checkpoints/pi05_t62_ctx/t62_ctx/{s}/params"
api.upload_folder(folder_path=ck, path_in_repo=f"ckpt_{s}/params", repo_id=REPO, repo_type="model")
print(f"[driver] uploaded step {s}", flush=True)
PYEOF
  echo "[driver] $(date -u +%FT%TZ) step $s params -> HF ($REPO/ckpt_$s)" | tee -a $DLOG
}
prune() {
  local steps newest s
  steps=$(committed_steps); newest=$(echo "$steps" | tail -1)
  for s in $steps; do
    upload_step "$s"
    if [ "$s" != "$newest" ] && [ -d "$CKDIR/$s/train_state" ]; then
      rm -rf "$CKDIR/$s/train_state"; echo "[driver] pruned train_state of step $s (kept params)" | tee -a $DLOG
    fi
  done
  df -h / | tail -1 | awk '{print "[driver] disk free: "$4}' >> $DLOG
}
finalize() {
  local final; final=$(committed_steps | tail -1)
  upload_step "$final"
  $PY - <<'PYEOF'
import os
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
from huggingface_hub import HfApi
tok = open("/root/.hf_token").read().strip(); api = HfApi(token=tok)
REPO = "arif101/b26-t62-params"
api.upload_folder(folder_path="/root/openpi_fork/outputs/assets/pi05_t62_ctx", path_in_repo="assets", repo_id=REPO, repo_type="model")
api.upload_file(path_or_fileobj="/root/t62_train.log", path_in_repo="provenance/t62_train.log", repo_id=REPO, repo_type="model")
info = api.repo_info(REPO, repo_type="model", files_metadata=True)
print(f"T62_PARAMS_UPLOADED files={len(info.siblings)} total={sum(s.size or 0 for s in info.siblings)/1e9:.2f}GB", flush=True)
PYEOF
}

launch ""
while true; do
  sleep 600
  if ! kill -0 "$(cat /root/t62_train.pid)" 2>/dev/null; then
    last=$(committed_steps | tail -1); last=${last:-0}
    if [ "$last" -ge "$FINAL_STEP" ]; then
      prune; finalize; echo "T62_TRAIN_DONE final_step=$last" | tee -a $DLOG; exit 0
    fi
    if [ "$RESUMES" -ge 2 ]; then
      prune; echo "T62_TRAIN_FAILED last_step=$last resumes_exhausted" | tee -a $DLOG; exit 1
    fi
    RESUMES=$((RESUMES + 1)); echo "[driver] trainer died at step $last — resume $RESUMES/2" | tee -a $DLOG
    launch "--resume"
  fi
  prune
done

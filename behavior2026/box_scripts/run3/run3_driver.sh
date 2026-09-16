#!/bin/bash
# RUN-3 DISK-SAFE ARM DRIVER (1xA100-80GB, 300GB disk) — skill step-5 pattern, per arm:
#   launch  scripts/b1k/train_b1k.py (the ONLY entrypoint routing create_b1k_data_loader) with
#           B1K_STAGE_OVERSAMPLE=8 and, for a1..a5, B1K_SAMPLE_WEIGHT_COL=sample_weight
#   verify  the loader's own log lines ([stage-oversample] and, for a1+, [sample-weight]) appear —
#           their absence is a FAILED launch (an arm must never silently train uniform)
#   prune   every 5 min strip train_state/ from all committed ckpts but the newest
#   offbox  every 5 min push the newest committed ckpt's params to HF (keeps 1 mid-run copy)
#   resume  up to 2 auto-resumes if the trainer dies before the final step
#   final   upload params + assets + logs to HF, verify, then slim local ckpts to final params only
# Any arm failure STOPS the driver (RUN3_DRIVER_FAILED) — the operator decides.
# Usage: ARMS="a0 a2 a1 a3 a4 a5" RUN3_STEPS=15000 nohup bash run3_driver.sh > /root/run3_logs/driver.out 2>&1 &
#        full stack: ARMS=full RUN3_STEPS=10000 FORK_SRC=/root/openpi_fork_v2/src bash run3_driver.sh  (FORK_SRC = patched src via PYTHONPATH)
set -u
FORK=/root/openpi_fork
R3=/root/run3
ARMS=${ARMS:-"a0 a2 a1 a3 a4 a5"}
STEPS=${RUN3_STEPS:-15000}; FINAL_STEP=$((STEPS - 1))
KEEP=${RUN3_KEEP_PERIOD:-5000}
LOGDIR=/root/run3_logs; mkdir -p $LOGDIR
DRV=$LOGDIR/driver.log
PY=$FORK/.venv/bin/python
say(){ echo "[driver $(date -u +%m-%dT%H:%M:%S)] $*" | tee -a $DRV; }

committed_steps(){ ls -d $CKDIR/*/ 2>/dev/null | grep -v "orbax-checkpoint-tmp" | sed "s|$CKDIR/||; s|/||" | grep -E "^[0-9]+$" | sort -n; }
launch(){
  cd $FORK
  setsid nohup env $ENVS ${FORK_SRC:+PYTHONPATH=$FORK_SRC} PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 \
    $PY scripts/b1k/train_b1k.py $CFG --exp_name $EXP --keep-period $KEEP --no-wandb-enabled $1 >> $LOG 2>&1 &
  echo $! > $LOGDIR/$ARM.pid
  say "$ARM: launched pid=$(cat $LOGDIR/$ARM.pid) args='$1' env='$ENVS'"
}
alive(){ kill -0 "$(cat $LOGDIR/$ARM.pid 2>/dev/null)" 2>/dev/null; }
prune(){
  local steps newest s; steps=$(committed_steps); newest=$(echo "$steps" | tail -1)
  for s in $steps; do
    if [ "$s" != "$newest" ] && [ -d "$CKDIR/$s/train_state" ]; then rm -rf "$CKDIR/$s/train_state"; say "$ARM: pruned train_state of step $s"; fi
  done
}
offbox(){
  local newest; newest=$(committed_steps | tail -1); [ -z "$newest" ] && return 0
  [ "$newest" = "$(cat $LOGDIR/$ARM.offbox 2>/dev/null)" ] && return 0
  [ -d "$CKDIR/$newest/params" ] || return 0
  if $PY $R3/run3_hf.py ckpt $ARM $newest $CKDIR/$newest/params >> $LOGDIR/offbox_$ARM.log 2>&1; then
    echo $newest > $LOGDIR/$ARM.offbox; say "$ARM: off-box ckpt $newest pushed"
  else say "$ARM: WARNING off-box push of $newest failed (see offbox_$ARM.log)"; fi
}
check_loader_lines(){  # returns 0 once the loader printed its weighting lines, 1 if still waiting
  grep -q "\[stage-oversample\]" $LOG || return 1
  if [ "$ARM" != "a0" ]; then grep -q "\[sample-weight\] column sample_weight" $LOG || return 1; fi
  return 0
}
failure_sig(){ grep -nE "Traceback|RESOURCE_EXHAUSTED|No space left|Killed|CUDA_ERROR|OOM" $LOG | tail -3; }

for ARM in $ARMS; do
  # arm name -> config: a0..a5 = Run-3 data arms; press / full = the 2026-09 stacked configs (PRESS_FIX_SPEC, TEMPORAL_FORCING_BUILD)
  case $ARM in press) CFG=pi05_radio_press; EXP=radio_press;; full) CFG=pi05_radio_full; EXP=radio_full;; *) CFG=pi05_radio_run3_$ARM; EXP=radio_run3_$ARM;; esac
  CKDIR=$FORK/outputs/checkpoints/$CFG/$EXP; LOG=$LOGDIR/train_$ARM.log
  if [ -f $LOGDIR/$ARM.DONE ]; then say "$ARM: already DONE, skipping"; continue; fi
  ROOT=$(env ${FORK_SRC:+PYTHONPATH=$FORK_SRC} $PY -c "import openpi.training.config as c; print(c.get_config('$CFG').data.base_config.dataset_root)" 2>/dev/null)
  if [ ! -d "$ROOT" ]; then say "$ARM: dataset $ROOT missing — skipping (A5 lands later)"; continue; fi
  ENVS="B1K_STAGE_OVERSAMPLE=8"; [ "$ARM" != "a0" ] && ENVS="$ENVS B1K_SAMPLE_WEIGHT_COL=sample_weight"
  RESUMES=0
  say "=== ARM $ARM  cfg=$CFG root=$ROOT steps=$STEPS ==="
  if [ -n "$(committed_steps)" ]; then launch "--resume"; else launch "--overwrite"; fi
  # launch verification: loader lines within 30 min
  t0=$(date +%s); LAUNCH_OK=0
  while [ $(( $(date +%s) - t0 )) -lt 1800 ]; do
    if check_loader_lines; then LAUNCH_OK=1; break; fi
    if ! alive; then break; fi
    sleep 30
  done
  if [ $LAUNCH_OK -eq 1 ]; then say "$ARM: loader weighting lines verified: $(grep -E '\[stage-oversample\]|\[sample-weight\]' $LOG | tr '\n' ' ')"
  else say "$ARM: FAILED LAUNCH — loader weighting lines absent ($(failure_sig))"; kill "$(cat $LOGDIR/$ARM.pid)" 2>/dev/null; echo "RUN3_DRIVER_FAILED arm=$ARM launch" | tee -a $DRV; exit 1; fi
  while true; do
    sleep 300
    if ! alive; then
      last=$(committed_steps | tail -1); last=${last:-0}
      if [ "$last" -ge "$FINAL_STEP" ]; then
        prune
        say "$ARM: training complete at step $last; finalizing"
        if $PY $R3/run3_hf.py final $ARM $CKDIR/$last/params $FORK/outputs/assets/$CFG $LOG $DRV $LOGDIR/preflight_$ARM.json >> $LOGDIR/finalize_$ARM.log 2>&1; then
          tail -1 $LOGDIR/finalize_$ARM.log | tee -a $DRV
          for s in $(committed_steps); do [ "$s" != "$last" ] && rm -rf "$CKDIR/$s"; done
          rm -rf "$CKDIR/$last/train_state"
          touch $LOGDIR/$ARM.DONE; say "$ARM: DONE (local: params-only at $CKDIR/$last)"; df -h / | tail -1 >> $DRV
        else say "$ARM: FINALIZE FAILED (see finalize_$ARM.log)"; echo "RUN3_DRIVER_FAILED arm=$ARM finalize" | tee -a $DRV; exit 1; fi
        break
      fi
      if [ "$RESUMES" -ge 2 ]; then echo "RUN3_DRIVER_FAILED arm=$ARM last_step=$last resumes_exhausted ($(failure_sig))" | tee -a $DRV; exit 1; fi
      RESUMES=$((RESUMES + 1)); say "$ARM: trainer died at step $last ($(failure_sig)) — resume $RESUMES/2"; launch "--resume"
    fi
    prune; offbox
    df -h / | tail -1 | awk -v a=$ARM '{print "[driver] "a" disk free: "$4}' >> $DRV
  done
done
echo "RUN3_ALL_DONE arms='$ARMS'" | tee -a $DRV

#!/bin/bash
# FULL-STACK launch, STAGE 3+ only (decision wait -> optional S1 re-tower -> driver). Re-armed after the false trigger of 09-16 21:51. Waits for S1 (arm a5) to finish, then:
#   1. GPU smoke of pi05_radio_full (20 steps on mix_full, checkpoint write) from the A4 warm start
#   2. frame cache: decode+resize every source once (CPU, ~2 h) so a tower re-pass is minutes
#   3. wait for the decision file /root/run3_logs/FULL_WARMSTART containing "a4" or "s1" (written after the S1 eval)
#      s1 -> ckpt_full_init -> S1 params; gists re-towered from cache with S1's SigLIP; mix_full re-assembled
#   4. launch the driver: ARMS=full RUN3_STEPS=10000 FORK_SRC=/root/openpi_fork_v2/src
# Usage: setsid nohup bash /root/run3/after_s1.sh > /root/run3_logs/after_s1.out 2>&1 &
set -eo pipefail
P=/root/openpi_fork/.venv/bin/python; R3=/root/run3; L=/root/run3_logs; SRC=/root/openpi_fork_v2/src
say(){ echo "[after_s1 $(date -u +%m-%dT%H:%M:%S)] $*" | tee -a $L/after_s1.log; }
mkdir -p /root/ckpt_full_init; [ -e /root/ckpt_full_init/params ] || ln -sfn /root/ckpt_a4/params /root/ckpt_full_init/params
say "3. waiting for the decision file $L/FULL_WARMSTART (a4|s1)"
until [ -s $L/FULL_WARMSTART ]; do sleep 120; done
CHOICE=$(tr -d '[:space:]' < $L/FULL_WARMSTART); say "decision: $CHOICE"
if [ "$CHOICE" = "s1" ]; then
  S1P=$(ls -d /root/openpi_fork/outputs/checkpoints/pi05_radio_run3_a5/radio_run3_a5/[0-9]*/params | sort -t/ -k8 -n | tail -1)
  test -d "$S1P" || { say "no S1 params dir"; exit 1; }
  ln -sfn $S1P /root/ckpt_full_init/params; say "warm start -> $S1P"
  for r in /root/manufactured/b1k_radio_factory /root/manufactured/b1k_radio_episodes /root/manufactured/b1k_radio_approach_v2 /root/b1k_radio_map; do
    n=$(basename $r)
    XLA_PYTHON_CLIENT_PREALLOCATE=false $P $R3/precompute_gists.py --root $r --params $S1P --from-cache /root/frame_cache/$n.npy 2>&1 | grep -E "tower pass|PRECOMPUTE" | tee -a $L/after_s1.log
  done
  $P $R3/assemble_run3_mix.py --overwrite --sources /root/b1k_radio_map /root/manufactured/b1k_radio_factory /root/manufactured/b1k_radio_episodes /root/manufactured/b1k_radio_approach_v2 --out /root/b1k_radio_mix_full 2>&1 | grep ASSEMBLED | tee -a $L/after_s1.log
  $P $R3/deregister_depth_streams.py --root /root/b1k_radio_mix_full | tail -1
  for n in sample_weight:1 gt_depth_ds:768 stage_v2:1 progress:1 toggled:1 target_points_v2:6 gist_head:2048 hist_geo:9; do $P $R3/register_feature.py --root /root/b1k_radio_mix_full --name ${n%%:*} --shape ${n##*:} >/dev/null; done
fi
say "4. launching the full-stack arm (warm start: $(readlink -f /root/ckpt_full_init/params))"
ARMS=full RUN3_STEPS=10000 FORK_SRC=$SRC setsid nohup bash $R3/run3_driver.sh > $L/driver_full.out 2>&1 < /dev/null &
echo $! > $L/driver_full.pid; say "driver pid $(cat $L/driver_full.pid)"
echo AFTER_S1_OK | tee -a $L/after_s1.log

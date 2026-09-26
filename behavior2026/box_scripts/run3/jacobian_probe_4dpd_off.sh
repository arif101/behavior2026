#!/bin/bash
# CORRECTIVE-FIELD MEASUREMENT, pointer-dropout arm ckpt 5000 served POINTER-OFF (2026-09-26): AFF_TAU=2 -> the affordance
# wrapper never injects a point (target_points_mask False -> the anchor-follow rule degenerates the third anchor to EE_R, as
# in the 70% dropped training samples); the online map therefore carries no target either; stage from the System-2 head
# (SERVE_STAGE_SOURCE=head) so the stage tracker's pointer dependence is out of the loop. Same 4 harvested near-grasp states
# and 15 conditions as the 09-21 (full) and 09-25 (4D) probes -> /root/jacobian_probe_4dpd_off/. Waits for the params.
# Usage: setsid nohup bash /root/jacobian_probe_4dpd_off.sh > /root/jacobian_probe_4dpd_off.out 2>&1 &
set -u
say(){ echo "[jac4dpd $(date -u +%m-%dT%H:%M:%S)] $*"; }
PYB=/root/miniconda3/envs/behavior/bin/python; S=/root/near_states; O=/root/jacobian_probe_4dpd_off; mkdir -p $O
until grep -q DL_4DPD_5000_OK /root/dl_4dpd_5000.log 2>/dev/null; do grep -q Traceback /root/dl_4dpd_5000.log 2>/dev/null && { say DL_FAILED; exit 1; }; sleep 20; done
mkdir -p /root/ckpt_4dpd && ln -sfn /root/run3_dl/4dpd/ckpt_5000/params /root/ckpt_4dpd/params && cp -r /root/ckpt_a4/assets /root/ckpt_4dpd/ 2>/dev/null
[ -d /root/openpi_fork/outputs/assets/pi05_radio_4d_pd ] || { mkdir -p /root/openpi_fork/outputs/assets/pi05_radio_4d_pd && cp -r /root/ckpt_a4/assets/* /root/openpi_fork/outputs/assets/pi05_radio_4d_pd/; }
POLICY_CONFIG=pi05_radio_4d_pd SERVE_FORWARD_MAP_TOKENS=1 HISTORY_MODE=normal SERVE_STAGE_SOURCE=head bash /root/serve_arm.sh 4dpd || { say SERVER_FAILED; exit 1; }
[ $(ls $S/near_tr*.npz 2>/dev/null | wc -l) -ge 4 ] || { say "NO_HARVESTED_STATES"; exit 1; }
for f in $S/near_tr0.npz $S/near_tr1.npz $S/near_tr2.npz $S/near_tr3.npz; do
  tag=$(basename $f .npz); k=${tag#near_tr}
  say "PROBE $tag pointer-off (45 rollouts)"
  cd /root/bw/BEHAVIOR-1K
  env PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 XDG_RUNTIME_DIR=/tmp/xdg OMNIGIBSON_HEADLESS=1 MAP_ARM=B AFF_TAU=2 SERVE_STAGE_SOURCE=head B1K_TASK_TARGETS=/root/task_targets.json B1K_TASK_NAME=turning_on_radio \
    JAC_STATE=$f JAC_TAG=$tag JAC_OUT=$O JAC_STEPS=16 JAC_SAMPLES=3 \
    $PYB -m omnigibson.eval.eval --task-name turning_on_radio --host 127.0.0.1 --port 8000 \
    --env-wrapper behavior2026_eval.jacobian_probe.JacobianProbeWrapper --instance-indices $k --num-rollouts 45 --mode train \
    --output-dir $O/eval_$tag --max-steps 6000 --headless > $O/eval_$tag.log 2>&1
  say "  -> $(grep -c JAC_RESULT $O/eval_$tag.log) results; tracebacks: $(grep -c Traceback $O/eval_$tag.log)"
done
say "JAC_DONE"

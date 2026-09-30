#!/bin/bash
# CORRECTIVE-FIELD MEASUREMENT, all-corrective arm (4dall) ckpt 5000 served POINTER-OFF (2026-09-30): identical protocol to
# jacobian_probe_4dpd_off.sh (AFF_TAU=2 -> no injected point, third anchor degenerates to EE_R, online map target-blind, stage
# from the System-2 head; same 4 harvested near-grasp states x 15 conditions x 3 samples) so the numbers compare directly
# with the three null probes (full +0.011 m / corr 0.013; 4D@5000 +0.010 / 0.086; 4dpd@5000 pointer-off +0.012 / 0.131).
# Params: pulled by dl_readout_mix.py --step 5000. Out: /root/jacobian_probe_4dall_off/.
# Usage: ( setsid nohup bash /root/jacobian_probe_4dall_off.sh > /root/jacobian_probe_4dall_off.out 2>&1 < /dev/null & )
set -u
say(){ echo "[jac4dall $(date -u +%m-%dT%H:%M:%S)] $*"; }
PYB=/root/miniconda3/envs/behavior/bin/python; S=/root/near_states; O=/root/jacobian_probe_4dall_off; mkdir -p $O
P=/root/run3_dl/4dall_5000/4dall/ckpt_5000/params; [ -d $P ] || { say "NO_PARAMS $P"; exit 1; }
mkdir -p /root/ckpt_4dall && ln -sfn $P /root/ckpt_4dall/params && cp -r /root/ckpt_a4/assets /root/ckpt_4dall/ 2>/dev/null
[ -d /root/openpi_fork/outputs/assets/pi05_radio_4d_all ] || { mkdir -p /root/openpi_fork/outputs/assets/pi05_radio_4d_all && cp -r /root/ckpt_a4/assets/* /root/openpi_fork/outputs/assets/pi05_radio_4d_all/; }
POLICY_CONFIG=pi05_radio_4d_all SERVE_FORWARD_MAP_TOKENS=1 HISTORY_MODE=normal SERVE_STAGE_SOURCE=head bash /root/serve_arm.sh 4dall || { say SERVER_FAILED; exit 1; }
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

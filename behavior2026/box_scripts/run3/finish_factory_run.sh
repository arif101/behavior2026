#!/bin/bash
# FINISH-TWIN FACTORY DRIVER (2026-10-02). For every stall state /root/stall_states/h4d_tr<id>.npz (ascending id) one Isaac
# boot runs K perturbation specs (unperturbed 00 + K-1 random: radio dlat/ddepth +-[0.02,0.08] m, yaw +-[0,20] deg, hand
# dy/dz +-[0,0.04] m, wrist yaw +-[0,15] deg; seeded per state) through FinishFactoryWrapper (--policy local). The 10-DOF
# servo reaches gaps <= FF_ARM_REACH (0.60 m); farther states are skipped (SKIP_GAP_TOO_LARGE) -> base-drive band later.
# Every BATCH states (or at the end) the strict renders are converted + pushed (finish_convert.sh, unique ROUND per batch)
# and the raw renders deleted (0.4-0.7 GB each). Skips states already done. STOP: touch /root/finish_factory.STOP
# Out: /root/factory_obs_finish (renders, transient), /root/factory_clips_finish/<tag>_meta.json, /root/finish_runs/tr<id>/,
#      HF b26-radio-manufactured/b1k_radio_finish_r<batch>_<ts>. Log: /root/finish_factory.out
# Usage: ( setsid nohup bash /root/finish_factory_run.sh > /root/finish_factory.out 2>&1 < /dev/null & )
set -u
say(){ echo "[finish_factory $(date -u +%m-%dT%H:%M:%S)] $*"; }
K=${K:-6}; BATCH=${BATCH:-6}; W=behavior2026_eval.finish_factory.FinishFactoryWrapper
export FF_ARM_REACH=${FF_ARM_REACH:-0.60} FF_MAX_GAP=${FF_MAX_GAP:-0.60}
mkdir -p /root/finish_runs /root/factory_obs_finish /root/factory_clips_finish
[ "$(pgrep -cf '^/root/miniconda3/envs/behavior/bin/python -m omnigibson.eval.eval')" = "0" ] || { say "HARNESS_BUSY"; exit 1; }
say "finish factory over $(ls /root/stall_states/h4d_tr*.npz | wc -l) stall states, K=$K specs each, batch=$BATCH, arm reach $FF_ARM_REACH"
specs_for(){ /root/openpi_fork/.venv/bin/python - "$1" "$K" <<'PY'
import sys, numpy as np
sid, K = int(sys.argv[1]), int(sys.argv[2]); rng = np.random.default_rng(1000 + sid)
out = ["0,0,0,0,0,0,00"]
for c in range(1, K):
    dl = rng.choice([-1, 1]) * rng.uniform(0.02, 0.08) if rng.random() < 0.7 else 0.0
    dd = rng.choice([-1, 1]) * rng.uniform(0.02, 0.08) if rng.random() < 0.7 else 0.0
    yw = rng.choice([-1, 1]) * rng.uniform(0.0, 20.0) if rng.random() < 0.5 else 0.0
    hy = rng.choice([-1, 1]) * rng.uniform(0.0, 0.04) if rng.random() < 0.5 else 0.0
    hz = rng.choice([-1, 1]) * rng.uniform(0.0, 0.04) if rng.random() < 0.5 else 0.0
    hw = rng.choice([-1, 1]) * rng.uniform(0.0, 15.0) if rng.random() < 0.4 else 0.0
    out.append(f"{dl:.3f},{dd:.3f},{yw:.1f},{hy:.3f},{hz:.3f},{hw:.1f},{c:02d}")
print(";".join(out))
PY
}
convert_batch(){ local nb=$1
  ls /root/factory_obs_finish/rac_*_200.npz >/dev/null 2>&1 || { say "batch $nb: no renders to convert"; return 0; }
  local R="finish${nb}_$(date -u +%m%d%H%M)"
  say "batch $nb: converting $(ls /root/factory_obs_finish/rac_*_200.npz | wc -l) renders -> b1k_radio_finish_r$R"
  BAR=strict ROUND=$R bash /root/finish_convert.sh > /root/finish_convert_r$R.out 2>&1
  grep -E "SELECTED|EPISODES|HF_PUSH_OK|FINISH_CONVERT_DONE|Traceback|Error" /root/finish_convert_r$R.out | tail -5
  if grep -q HF_PUSH_OK /root/finish_convert_r$R.out; then rm -f /root/factory_obs_finish/rac_*_200.npz; say "batch $nb pushed as b1k_radio_finish_r$R; renders deleted"
  else say "CONVERT/PUSH FAILED batch $nb (renders kept; factory continues, disk permitting)"; fi; }
n=0; nb=0; since=0
for f in $(ls /root/stall_states/h4d_tr*.npz | sort -V); do
  id=$(basename $f .npz | sed -E "s/h4d_tr//"); tag=tr$(printf %03d $id)
  [ -f /root/factory_clips_finish/${tag}_00_meta.json ] && continue
  free_gb=$(df -BG / | awk "NR==2{gsub(\"G\",\"\",\$4); print \$4}"); [ "$free_gb" -lt 8 ] && { nb=$((nb+1)); convert_batch $nb; since=0; }
  SP=$(specs_for $id); n=$((n+1))
  say "=== $tag ($n): $SP ==="
  FF_STATE=$f FF_ID=$id FF_PERTURBS="$SP" AFF_TAU=2 timeout ${STATE_TIMEOUT:-6000} bash /root/freeze_rollout.sh $W train $id /root/finish_runs/$tag 50 --policy local > /root/finish_runs/${tag}.out 2>&1
  grep -hE "FINISH_RESULT|FINISH_DONE|DIAG_TRACEBACK" /root/finish_runs/$tag/eval.log 2>/dev/null | cut -c1-200
  rm -rf /root/finish_runs/$tag/videos
  find /tmp -mindepth 1 -maxdepth 1 -type d -user root -name "tmp????????" -mmin +120 -exec rm -rf {} + 2>/dev/null
  since=$((since+1)); [ $since -ge $BATCH ] && { nb=$((nb+1)); convert_batch $nb; since=0; }
  [ -f /root/finish_factory.STOP ] && { say "STOP file seen"; break; }
done
nb=$((nb+1)); convert_batch $nb
say "FINISH_FACTORY_DONE states=$n clips_strict=$(grep -l "\"honest_strict\": true" /root/factory_clips_finish/*_meta.json 2>/dev/null | wc -l) ok=$(grep -l "\"ok\": true" /root/factory_clips_finish/*_meta.json 2>/dev/null | wc -l) meta=$(ls /root/factory_clips_finish/*_meta.json 2>/dev/null | wc -l) skipped_far=$(grep -l SKIP_GAP_TOO_LARGE /root/factory_clips_finish/*_meta.json 2>/dev/null | wc -l)"

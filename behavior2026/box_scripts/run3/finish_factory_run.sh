#!/bin/bash
# FINISH-TWIN FACTORY DRIVER (2026-10-01). For every stall state in /root/stall_states/h4d_tr<id>.npz: one Isaac boot runs
# K perturbation specs (unperturbed 00 + K-1 random: radio dlat/ddepth in +-[0.02,0.08] m, yaw +-[0,20] deg, hand dy/dz
# +-[0,0.04] m, wrist yaw +-[0,15] deg; seeded per state) through FinishFactoryWrapper (--policy local, no server).
# Waits for the stall harvest to finish (it owns the GPU with the policy server). Skips states already done.
# Out: /root/factory_obs_finish/rac_9<id><code>_200.npz (+ name_map.txt), /root/factory_clips_finish/<tag>_meta.json,
#      per-state harness dirs /root/finish_runs/tr<id>/. Log: /root/finish_factory.out. STOP: touch /root/finish_factory.STOP
# Usage: ( setsid nohup bash /root/finish_factory_run.sh > /root/finish_factory.out 2>&1 < /dev/null & )
set -u
say(){ echo "[finish_factory $(date -u +%m-%dT%H:%M:%S)] $*"; }
K=${K:-6}; W=behavior2026_eval.finish_factory.FinishFactoryWrapper
mkdir -p /root/finish_runs /root/factory_obs_finish /root/factory_clips_finish
until grep -q STALL_HARVEST_DONE /root/stall_harvest.out 2>/dev/null || [ "$(pgrep -cf '^bash /root/stall_harvest_4dall.sh')" = "0" ]; do sleep 120; done
for P in $(pgrep -f "^/root/openpi_fork/.venv/bin/python scripts/b1k/serve_b1k.py"); do kill $P; done; sleep 3
say "harvest done -> finish factory over $(ls /root/stall_states/h4d_tr*.npz | wc -l) stall states, K=$K specs each"
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
n=0
for f in $(ls /root/stall_states/h4d_tr*.npz | sort -V); do
  id=$(basename $f .npz | sed -E "s/h4d_tr//"); tag=tr$(printf %03d $id)
  [ -f /root/factory_clips_finish/${tag}_00_meta.json ] && continue
  SP=$(specs_for $id); n=$((n+1))
  say "=== $tag ($n): $SP ==="
  FF_STATE=$f FF_ID=$id FF_PERTURBS="$SP" AFF_TAU=2 timeout 3000 bash /root/freeze_rollout.sh $W train $id /root/finish_runs/$tag 50 --policy local > /root/finish_runs/${tag}.out 2>&1
  grep -hE "FINISH_RESULT|FINISH_DONE|OBS_SAVED|DIAG_TRACEBACK" /root/finish_runs/$tag/eval.log 2>/dev/null | cut -c1-200
  find /tmp -mindepth 1 -maxdepth 1 -type d -user root -name "tmp????????" -mmin +120 -exec rm -rf {} + 2>/dev/null
  [ -f /root/finish_factory.STOP ] && { say "STOP file seen"; break; }
done
say "FINISH_FACTORY_DONE states=$n clips=$(ls /root/factory_obs_finish/rac_*_200.npz 2>/dev/null | wc -l) strict=$(grep -l "\"honest_strict\": true" /root/factory_clips_finish/*_meta.json 2>/dev/null | wc -l) meta=$(ls /root/factory_clips_finish/*_meta.json 2>/dev/null | wc -l)"

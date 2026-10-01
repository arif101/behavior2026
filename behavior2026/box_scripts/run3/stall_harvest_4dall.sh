#!/bin/bash
# STALL-STATE HARVEST for the finish factory (2026-10-01). Runs the 4dall FINAL policy POINTER-OFF (the eval's serving stack:
# pi05_radio_4d_all, map tokens forwarded, HISTORY_MODE=normal, stage from the System-2 head, AFF_TAU=2) on TRAIN layouts
# through FreezeHarvestWrapper, which dumps the full sim state when the robot has been stationary for FREEZE_WIN steps after
# FREEZE_MIN (or at FREEZE_CAP). These are the policy's OWN hover-and-freeze states -> start states for the finish twins
# (DAgger ingredient). NEVER instance 301 / public_test. Train instance ids are direct (0..299); SPREAD picks every 3rd.
# Out: /root/stall_states/h4d_tr<id>.{npz,json} + per-rollout dirs under /root/stall_harvest/. Log: /root/stall_harvest.out
# Usage: ( setsid nohup bash /root/stall_harvest_4dall.sh > /root/stall_harvest.out 2>&1 < /dev/null & )
set -u
say(){ echo "[stall_harvest $(date -u +%m-%dT%H:%M:%S)] $*"; }
S=/root/stall_states; D=/root/stall_harvest; mkdir -p $S $D
HARV=behavior2026_eval.freeze_harvest.FreezeHarvestWrapper
IDS=${IDS:-$(seq 0 3 297)}
[ -d /root/ckpt_4dall_final/params ] || { say NO_CKPT; exit 1; }
[ "$(pgrep -cf '^/root/miniconda3/envs/behavior/bin/python -m omnigibson.eval.eval')" = "0" ] || { say "HARNESS_BUSY"; exit 1; }
POLICY_CONFIG=pi05_radio_4d_all SERVE_FORWARD_MAP_TOKENS=1 HISTORY_MODE=normal SERVE_STAGE_SOURCE=head bash /root/serve_arm.sh 4dall_final || { say SERVER_FAILED; exit 1; }
n=0; ok=0
for k in $IDS; do
  tag=h4d_tr$k; [ -f $S/$tag.npz ] && continue
  n=$((n+1)); say "=== harvest $tag ($n) ==="
  rm -f /root/stage_head_log.jsonl
  FREEZE_OUT=$S FREEZE_TAG=$tag FREEZE_MIN=450 FREEZE_WIN=150 FREEZE_CAP=1500 AFF_TAU=2 SERVE_STAGE_SOURCE=head \
    bash /root/freeze_rollout.sh $HARV train $k $D/$tag 1700 | grep -E "FREEZE_HARVESTED|FREEZE_DUMP_FAILED|DIAG_RESULT|DIAG_TRACEBACK" | cut -c1-220
  cp /root/stage_head_log.jsonl $D/$tag/ 2>/dev/null
  if [ -f $S/$tag.npz ]; then ok=$((ok+1)); else echo "NO_FREEZE $tag"; fi
  [ -f /root/stall_harvest.STOP ] && { say "STOP file seen"; break; }
done
for P in $(pgrep -f "^/root/openpi_fork/.venv/bin/python scripts/b1k/serve_b1k.py"); do kill $P; done
/root/openpi_fork/.venv/bin/python - <<'PY'
import glob, json, numpy as np
rows = []
for f in sorted(glob.glob("/root/stall_states/h4d_tr*.json")):
    m = json.load(open(f)); rows.append((m.get("tag"), m.get("step"), m.get("stationary"), m.get("base_to_radio_xy"), m.get("dist_R_last"), m.get("dist_L_last")))
print(f"STALL_BANK n={len(rows)} stationary={sum(1 for r in rows if r[2])} base_to_radio<1.2m={sum(1 for r in rows if r[3] is not None and r[3] < 1.2)}")
b = [r[3] for r in rows if r[3] is not None]; print("base_to_radio_xy median %.2f, IQR %s" % (np.median(b), np.percentile(b, [25, 75]).round(2).tolist()) if b else "no base_to_radio")
for r in rows: print("  ", r)
PY
say "STALL_HARVEST_DONE attempted=$n harvested=$ok"

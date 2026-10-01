#!/bin/bash
# HELD-OUT ODART TWINS (2026-10-01): object-perturbation counterfactual clips on factory demos the 4dall arm NEVER saw
# (its 99 twins came from 15 of the 21 factory_obs2 demos; these 17 demos are the rest of the 38-demo factory set).
# Purpose: paired-loss readout on unseen demos = memorization-vs-perception test; the clips double as next-arm data.
# Same factory, same 8 perturbations, same strict bar, same converter; HF root b1k_radio_odart_rheldout<tier>_<ts>.
# Waits for the n=25 eval chain to release the GPU. Tier 1 = 6 demos spread over the range, converted first so the
# readout can start; tier 2 = the other 11. Skip a demo after 2 failed attempts (as odart_round.sh).
# Usage: ( setsid nohup bash /root/odart_heldout.sh > /root/odart_heldout.out 2>&1 < /dev/null & )
set -u
say(){ echo "[odart_heldout $(date -u +%m-%dT%H:%M:%S)] $*"; }
PY=/root/miniconda3/envs/behavior/bin/python; L=/root/odart_heldout_logs; mkdir -p $L /root/factory_obs_odart /root/factory_clips_odart
PERTS="0.05,0,0,ol5 -0.05,0,0,olm5 0,0.05,0,od5 0,-0.05,0,odm5 0,0,15,oy15 0,0,-15,oym15 0.04,0.03,10,omix1 -0.04,-0.03,-10,omix2"
TIER1="100 120 230 280 350 400"; TIER2="110 130 140 220 240 290 300 360 380 410 420"
until grep -q EVAL_4DALL_CHAIN_DONE /root/eval_4dall_chain.out 2>/dev/null && [ "$(pgrep -cf '^/root/openpi_fork/.venv/bin/python scripts/b1k/serve_b1k.py')" = "0" ] && [ "$(pgrep -cf '^/root/miniconda3/envs/behavior/bin/python -m omnigibson.eval.eval')" = "0" ]; do sleep 120; done
say "eval chain done, GPU free -> held-out factory"
[ -z "$(ls /root/factory_obs_odart/*.npz 2>/dev/null)" ] || { say "STALE_RENDERS in /root/factory_obs_odart (would mix into the held-out root)"; exit 1; }
run_tier(){ local tier=$1; shift; local n=0
  for d in "$@"; do for pert in $PERTS; do tag=${pert##*,}
    [ "$(grep -cx "$d" $L/fail.txt 2>/dev/null)" -ge 2 ] && continue
    [ -s $L/d${d}_${tag}.log ] && grep -qE "RESULT d|OBS_SAVED|OBJPERTURB dlat" $L/d${d}_${tag}.log && continue
    find /tmp -mindepth 1 -maxdepth 1 -type d -user root -name "tmp????????" -mmin +120 -exec rm -rf {} + 2>/dev/null
    OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 OMNIGIBSON_HEADLESS=1 XDG_RUNTIME_DIR=/tmp/xdg CUDA_VISIBLE_DEVICES=0 \
      timeout 2400 $PY -u /root/factory_approach_cap_v13_odart.py --demo $d --perturb-object="$pert" > $L/d${d}_${tag}.log 2>&1
    rm -f /root/fap_tmp_$d.hdf5; n=$((n+1))
    echo "d$d $tag $(date -u +%H:%M) $(grep -oE "RESULT d[0-9]+ [A-Z_]+|OBS_SAVED rac_[0-9a-z_]+_200.npz \([0-9]+ steps\)" $L/d${d}_${tag}.log | tail -1)"
    grep -qE "OBS_SAVED" $L/d${d}_${tag}.log || echo $d >> $L/fail.txt
  done; done
  say "tier $tier: $n attempts, renders=$(ls /root/factory_obs_odart/*.npz 2>/dev/null | wc -l)"
  if ls /root/factory_obs_odart/rac_*_200.npz >/dev/null 2>&1; then
    local R="heldout${tier}_$(date -u +%m%d%H%M)"
    BAR=strict ROUND=$R bash /root/odart_convert.sh > /root/odart_convert_r$R.out 2>&1
    grep -E "SELECTED|EPISODES|HF_PUSH_OK|DART_CONVERT_DONE|Traceback|Error" /root/odart_convert_r$R.out | tail -5
    if grep -q HF_PUSH_OK /root/odart_convert_r$R.out; then rm -f /root/factory_obs_odart/rac_*_200.npz; say "tier $tier converted+pushed as b1k_radio_odart_r$R (root kept at /root/b1k_radio_odart_r$R), renders deleted"
    else say "CONVERT/PUSH FAILED tier $tier (renders kept)"; fi
  fi; }
run_tier 1 $TIER1
run_tier 2 $TIER2
say "ODART_HELDOUT_DONE strict_total=$(grep -l "\"honest_strict\": true" /root/factory_clips_odart/*_meta.json | wc -l) fail_list=$(sort -n $L/fail.txt 2>/dev/null | uniq -c | tr "\n" " ")"

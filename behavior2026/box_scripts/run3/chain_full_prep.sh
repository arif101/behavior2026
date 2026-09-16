#!/bin/bash
# After the gist precompute: assemble the FULL mix (v2 labels + gist_head/hist_geo) and run the CPU preflight for
# pi05_radio_full. Waits for PRECOMPUTE_ALL_DONE in /root/run3_logs/precompute_all.log.
# Usage: setsid nohup bash /root/run3/chain_full_prep.sh > /root/run3_logs/chain_full_prep.out 2>&1 &
set -eo pipefail
P=/root/openpi_fork/.venv/bin/python; R3=/root/run3; L=/root/run3_logs
say(){ echo "[full_prep $(date -u +%m-%dT%H:%M:%S)] $*"; }
until grep -q PRECOMPUTE_ALL_DONE $L/precompute_all.log 2>/dev/null; do sleep 60; done
grep -c PRECOMPUTE_GISTS_OK $L/precompute_all.log | xargs -I{} echo "sources with gists (this run): {} (+ approach_v2 earlier)"
MAP=/root/b1k_radio_map; FAC=/root/manufactured/b1k_radio_factory; EPI=/root/manufactured/b1k_radio_episodes; APP=/root/manufactured/b1k_radio_approach_v2
for r in $MAP $FAC $EPI $APP; do $P -c "import json,sys; f=json.load(open('$r/meta/info.json'))['features']; assert all(k in f for k in ('gist_head','hist_geo','stage_v2','progress','target_points_v2')), '$r missing columns'"; done
say "1. assemble mix_full"
$P $R3/assemble_run3_mix.py --overwrite --sources $MAP $FAC $EPI $APP --out /root/b1k_radio_mix_full 2>&1 | grep ASSEMBLED
$P $R3/deregister_depth_streams.py --root /root/b1k_radio_mix_full | tail -1
for n in sample_weight:1 gt_depth_ds:768 stage_v2:1 progress:1 toggled:1 target_points_v2:6 gist_head:2048 hist_geo:9; do $P $R3/register_feature.py --root /root/b1k_radio_mix_full --name ${n%%:*} --shape ${n##*:} >/dev/null; done
# the full config's dataset_root is the A4 mix path (cloned from the press config): point the preflight at the full mix
say "2. preflight_full (CPU)"
cd /root/openpi_fork
PYTHONPATH=/root/openpi_fork_v2/src JAX_PLATFORMS=cpu B1K_STAGE_OVERSAMPLE=8 B1K_SAMPLE_WEIGHT_COL=sample_weight taskset -c 64-127 $P $R3/preflight_full.py /root/b1k_radio_mix_full 6 2>&1 | grep -vE "Warning|warn" | tail -8
mkdir -p outputs/assets/pi05_radio_full/b1k_radio && cp outputs/assets/pi05_radio_run3_a5/b1k_radio/norm_stats.json outputs/assets/pi05_radio_full/b1k_radio/
df -h / | tail -1
echo CHAIN_FULL_PREP_DONE

#!/bin/bash
# RUN-3 DATA PREP (trainer box). Prereqs: /root/.hf_token; fork venv; dl_run3.py map/manu/ckpt done.
# Builds per-arm mixes (hardlinked videos, no duplication):
#   mix_a1 = map            (a0 and a1 share it; a0 just launches WITHOUT B1K_SAMPLE_WEIGHT_COL)
#   mix_a2 = map+factory    mix_a3 = map+episodes    mix_a4 = map+factory+episodes   mix_a5 = a4+approach
# Source weights (sample_weight column): map 1.0 with poison windows -> 0.1; factory 4.78; episodes 2.0
# (the assembler's measured advisory: factory mass == map's stage-1->2 transition-neighbourhood mass).
set -eo pipefail
PY=/root/openpi_fork/.venv/bin/python
B=/root/behavior2026/behavior2026/box_scripts
R3=/root/run3
MAP=/root/b1k_radio_map; FAC=/root/manufactured/b1k_radio_factory; EPI=/root/manufactured/b1k_radio_episodes; APP=/root/manufactured/b1k_radio_approach
W_FACTORY=${W_FACTORY:-4.78}; W_EPISODES=${W_EPISODES:-2.0}; W_APPROACH=${W_APPROACH:-1.0}
log(){ echo "[prep $(date -u +%H:%M:%S)] $*"; }

log "1. map videos (organizer repo) + meta rewrite -> $MAP"
if [ ! -f $MAP/meta/episodes/chunk-000/file-000.parquet ] || [ ! -d $MAP/videos ]; then
  $PY $B/restore_map_videos.py
fi
$PY $R3/register_feature.py --root $MAP --name gt_depth_ds --shape 768

log "2. sample_weight columns (deterministic: always --overwrite-col)"
$PY $R3/add_sample_weights.py --root $MAP --poison /root/manufactured/poison_windows.json --overwrite-col | tail -3
$PY $R3/add_sample_weights.py --root $FAC --default-weight $W_FACTORY --overwrite-col | tail -2
$PY $R3/add_sample_weights.py --root $EPI --default-weight $W_EPISODES --overwrite-col | tail -2
if [ -d $APP ]; then $PY $R3/add_sample_weights.py --root $APP --default-weight $W_APPROACH --overwrite-col | tail -2; fi

log "3. dry-run (expect 296 eps / 464,242 frames)"
$PY $R3/assemble_run3_mix.py --dry-run --sources $MAP $FAC $EPI

log "4. assemble per-arm mixes"
$PY $R3/assemble_run3_mix.py --overwrite --sources $MAP           --out /root/b1k_radio_mix_a1 | tail -2
$PY $R3/assemble_run3_mix.py --overwrite --sources $MAP $FAC      --out /root/b1k_radio_mix_a2 | tail -2
$PY $R3/assemble_run3_mix.py --overwrite --sources $MAP $EPI      --out /root/b1k_radio_mix_a3 | tail -2
$PY $R3/assemble_run3_mix.py --overwrite --sources $MAP $FAC $EPI --out /root/b1k_radio_mix_a4 | tail -2
if [ -d $APP ]; then $PY $R3/assemble_run3_mix.py --overwrite --sources $MAP $FAC $EPI $APP --out /root/b1k_radio_mix_a5 | tail -2; fi

log "5. de-register depth streams (converter pts jitter; training never decodes depth) + register columns"
for m in a1 a2 a3 a4 a5; do
  [ -d /root/b1k_radio_mix_$m ] || continue
  $PY $B/deregister_depth_streams.py --root /root/b1k_radio_mix_$m | tail -1
  $PY $R3/register_feature.py --root /root/b1k_radio_mix_$m --name sample_weight --shape 1
  $PY $R3/register_feature.py --root /root/b1k_radio_mix_$m --name gt_depth_ds --shape 768
done

log "6. warm start (Run-2 final params) + Run-2 norm stats copied per arm"
ln -sfn /root/warmstart_run3_raw /root/warmstart_run3
ls /root/warmstart_run3/params >/dev/null
NS=$(ls /root/warmstart_run3_raw/assets/*/norm_stats.json | head -1)
for a in a0 a1 a2 a3 a4 a5; do
  mkdir -p /root/openpi_fork/outputs/assets/pi05_radio_run3_$a/b1k_radio
  cp $NS /root/openpi_fork/outputs/assets/pi05_radio_run3_$a/b1k_radio/norm_stats.json
done
df -h / | tail -1
echo PREP_RUN3_DATA_OK

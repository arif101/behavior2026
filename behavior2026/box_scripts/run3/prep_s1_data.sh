#!/bin/bash
# S1 DATA PREP (2026-09-15): mix_a5 = map + factory + episodes + b1k_radio_approach_v2 (the FIXED-restore approach clips).
# Weights: map 1.0 (poison windows 0.1), factory 4.78, episodes 2.0, approach: stage-1 frames 2.0 / transport tail 0.5.
# Prereqs: /root/.hf_token, fork venv, dl_s1.py map/manu/ckpt done, /root/run3 tooling.
set -eo pipefail
PY=/root/openpi_fork/.venv/bin/python; R3=/root/run3
MAP=/root/b1k_radio_map; FAC=/root/manufactured/b1k_radio_factory; EPI=/root/manufactured/b1k_radio_episodes; APP=/root/manufactured/b1k_radio_approach_v2
log(){ echo "[prep_s1 $(date -u +%H:%M:%S)] $*"; }
log "1. map videos (organizer repo) + meta rewrite -> $MAP"
if [ ! -f $MAP/meta/episodes/chunk-000/file-000.parquet ] || [ ! -d $MAP/videos ]; then $PY $R3/restore_map_videos.py; fi
$PY $R3/register_feature.py --root $MAP --name gt_depth_ds --shape 768
log "2. sample_weight columns"
$PY $R3/add_sample_weights.py --root $MAP --poison /root/manufactured/poison_windows.json --overwrite-col | tail -2
$PY $R3/add_sample_weights.py --root $FAC --default-weight 4.78 --overwrite-col | tail -1
$PY $R3/add_sample_weights.py --root $EPI --default-weight 2.0 --overwrite-col | tail -1
$PY $R3/add_sample_weights.py --root $APP --default-weight 2.0 --overwrite-col | tail -1
$PY $R3/set_stage_weights.py --root $APP --w-approach 2.0 --w-tail 0.5
log "3. dry-run"
$PY $R3/assemble_run3_mix.py --dry-run --sources $MAP $FAC $EPI $APP
log "4. assemble mix_a5"
$PY $R3/assemble_run3_mix.py --overwrite --sources $MAP $FAC $EPI $APP --out /root/b1k_radio_mix_a5 | tail -3
log "5. de-register depth streams + register columns"
$PY $R3/deregister_depth_streams.py --root /root/b1k_radio_mix_a5 | tail -1
$PY $R3/register_feature.py --root /root/b1k_radio_mix_a5 --name sample_weight --shape 1
$PY $R3/register_feature.py --root /root/b1k_radio_mix_a5 --name gt_depth_ds --shape 768
log "6. warm start (Run-2 final params) + Run-2 norm stats"
ln -sfn /root/warmstart_run3_raw /root/warmstart_run3; ls /root/warmstart_run3/params >/dev/null
NS=$(ls /root/warmstart_run3_raw/assets/*/norm_stats.json | head -1)
mkdir -p /root/openpi_fork/outputs/assets/pi05_radio_run3_a5/b1k_radio && cp $NS /root/openpi_fork/outputs/assets/pi05_radio_run3_a5/b1k_radio/norm_stats.json
md5sum /root/openpi_fork/outputs/assets/pi05_radio_run3_a5/b1k_radio/norm_stats.json
df -h / | tail -1
echo PREP_S1_DATA_OK

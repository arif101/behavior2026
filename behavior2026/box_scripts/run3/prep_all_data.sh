#!/bin/bash
# ALL-CORRECTIVE DATA PREP (2026-09-28): build /root/b1k_radio_mix_all = map + factory + episodes + ALL DART roots (r1-r3 as
# in mix_4d, plus r4, r5, r6, r6_0925_1457, r7, r8 (9-ep HF version), r8_09261035, r9_09261201) + ALL ODART roots (r1..r10).
# Reuses the mix_4d backup for the 322 already-processed episodes (their v2 labels, gists, cam_pose AND hist_tok columns are
# copied back into the source roots by unassemble_columns.py) and computes labels only for the ~176 new episodes.
# Every 09-25 lesson is built in: arrow-native hist writes, 16k-row row groups, canonical column order, atomic replace,
# guarded scripts, FULL warm start for parity, norm-stats assets per config name, MIX_ALL_* gates on explicit markers.
# Prereqs (trainer_bringup_4d.sh-style): fork venv at /root/openpi_fork, /root/run3/*.py helpers, /root/.hf_token,
# downloads under /root/backup (map), /root/manufactured (factory, episodes, poison_windows.json, all dart/odart roots),
# /root/mixes_bk/{mix_4d,fk,fk_dart_r1..3,fk_<rootname>...}, /root/run3_dl/{a4,full}. Log: /root/run3_logs/prep_all.out
set -u
PY=/root/openpi_fork/.venv/bin/python; R3=/root/run3; L=/root/run3_logs; mkdir -p $L /root/frame_cache
say(){ echo "[prep_all $(date -u +%m-%dT%H:%M:%S)] $*"; }
MAP=/root/backup/b1k_radio_map; FAC=/root/manufactured/b1k_radio_factory; EPI=/root/manufactured/b1k_radio_episodes; M=/root/manufactured
OLD="$M/b1k_radio_dart_r1 $M/b1k_radio_dart_r2 $M/b1k_radio_dart_r3"
NEW="$M/b1k_radio_dart_r4 $M/b1k_radio_dart_r5 $M/b1k_radio_dart_r6 $M/b1k_radio_dart_r6_0925_1457 $M/b1k_radio_dart_r7 $M/b1k_radio_dart_r8 $M/b1k_radio_dart_r8_09261035 $M/b1k_radio_dart_r9_09261201 $(ls -d $M/b1k_radio_odart_r* | sort -V | tr '\n' ' ')"
for D in $OLD $NEW; do [ -d $D/data ] || { say "MISSING_ROOT $D"; exit 1; }; done
fkdir(){ n=$(basename $1); case $n in b1k_radio_dart_r1) echo /root/mixes_bk/fk_dart_r1;; b1k_radio_dart_r2) echo /root/mixes_bk/fk_dart_r2;; b1k_radio_dart_r3) echo /root/mixes_bk/fk_dart_r3;; *) echo /root/mixes_bk/fk_$n;; esac; }
for D in $NEW; do [ -f $(fkdir $D)/cam_pose.npy ] || { say "MISSING_FK $(fkdir $D)"; exit 1; }; done
# ---- 1. sample weights ------------------------------------------------------------------------------------------------
$PY $R3/register_feature.py --root $MAP --name gt_depth_ds --shape 768 2>&1 | tail -1
$PY $R3/add_sample_weights.py --root $MAP --poison $M/poison_windows.json --overwrite-col | tail -1
$PY $R3/add_sample_weights.py --root $FAC --default-weight 4.78 --overwrite-col | tail -1
$PY $R3/add_sample_weights.py --root $EPI --default-weight 2.0 --overwrite-col | tail -1
for D in $OLD $NEW; do $PY $R3/add_sample_weights.py --root $D --default-weight 4.78 --overwrite-col | tail -1; done
# ---- 2. the 322 processed episodes: copy every derived column back from the mix_4d backup (v2 labels, gists, cam_pose,
#         hist tokens). mix_4d's episode order = map 0-199, factory 200-237, episodes 238-295, dart r1 296-313, r2 314, r3 315-321,
#         i.e. exactly $MAP $FAC $EPI $OLD; the mix already carries cam_pose, so no --fk (the fk/ index is mix_full's layout). -----
$PY $R3/unassemble_columns.py --mix /root/mixes_bk/mix_4d --sources $MAP $FAC $EPI $OLD \
  --cols stage_v2 progress toggled target_points_v2 gist_head hist_geo cam_pose hist_tok hist_cellxyz hist_cellvalid odom_xyyaw 2>&1 | tail -7
grep -q UNASSEMBLE_OK $L/prep_all.out || { say "UNASSEMBLE_FAILED"; exit 1; }
# ---- 3. new roots: v2 labels, gists (A4 tower), cam_pose, frame cache + hist tokens (FULL tower) ------------------------
cd /root/openpi_fork
for D in $NEW; do
  n=$(basename $D); say "labels: $n"
  $PY $R3/relabel_v2.py --root $D --press-anchor none 2>&1 | tail -1
  XLA_PYTHON_CLIENT_PREALLOCATE=false $PY $R3/precompute_gists.py --root $D --params /root/run3_dl/a4/params --cache-frames /root/frame_cache/$n.npy 2>&1 | grep -E "PRECOMPUTE_GISTS_OK|Traceback" | tail -1
  $PY $R3/add_cam_pose_column.py --root $D --npy $(fkdir $D)/cam_pose.npy --index $(fkdir $D)/index.npy | tail -1
  XLA_PYTHON_CLIENT_PREALLOCATE=false $PY $R3/precompute_hist_tokens.py --root $D --params /root/run3_dl/full/params --from-cache /root/frame_cache/$n.npy 2>&1 | grep -E "PRECOMPUTE_HIST_TOKENS_OK|Traceback" | tail -1
  rm -f /root/frame_cache/$n.npy
done
for D in $NEW; do n=$(basename $D); grep -q "PRECOMPUTE_HIST_TOKENS_OK $n" $L/prep_all.out || { say "NEW_ROOT_INCOMPLETE $n (hist tokens)"; exit 1; }; grep -q "PRECOMPUTE_GISTS_OK $n" $L/prep_all.out || { say "NEW_ROOT_INCOMPLETE $n (gists)"; exit 1; }; done
[ "$(grep -c "CAM_POSE_OK rows=[0-9]* missing=0" $L/prep_all.out)" -ge "$(echo $NEW | wc -w)" ] || { say "CAM_POSE_INCOMPLETE (a root has missing rows)"; exit 1; }
# ---- 4. canonical column order (the assembler asserts schema equality) and assemble ----------------------------------------
ORDER="$($PY - <<PYX
import glob, pyarrow.parquet as pq
names = pq.ParquetFile(sorted(glob.glob("$MAP/data/**/*.parquet", recursive=True))[0]).schema_arrow.names
add = ["sample_weight","gt_depth_ds","stage_v2","progress","toggled","target_points_v2","gist_head","hist_geo","cam_pose","hist_tok","hist_cellxyz","hist_cellvalid","odom_xyyaw"]
print(" ".join([c for c in names if c not in add] + add))
PYX
)"
for D in $MAP $FAC $EPI $OLD $NEW; do $PY $R3/canonicalize_columns.py --root $D --order $ORDER | tail -1; done
$PY $R3/assemble_run3_mix.py --overwrite --sources $MAP $FAC $EPI $OLD $NEW --out /root/b1k_radio_mix_all 2>&1 | grep -E "ASSEMBLED|Traceback|Error|assert" | tail -3
grep -q "ASSEMBLED /root/b1k_radio_mix_all" $L/prep_all.out || { say "ASSEMBLE_FAILED"; exit 1; }
$PY $R3/deregister_depth_streams.py --root /root/b1k_radio_mix_all | tail -1
$PY $R3/regroup_parquets.py --root /root/b1k_radio_mix_all --rows 16384 2>&1 | tail -2
$PY - <<'PYX'
import glob, json, pyarrow.parquet as pq
info = json.load(open("/root/b1k_radio_mix_all/meta/info.json")); f = info["features"]
fs = sorted(glob.glob("/root/b1k_radio_mix_all/data/**/*.parquet", recursive=True))
need = ["sample_weight","gt_depth_ds","stage_v2","progress","toggled","target_points_v2","gist_head","hist_geo","cam_pose","hist_tok","hist_cellxyz","hist_cellvalid","odom_xyyaw"]
missing_f = [c for c in need if c not in f]; missing_p = [c for c in need if c not in pq.ParquetFile(fs[0]).schema_arrow.names]
maxrg = max(pq.ParquetFile(x).metadata.row_group(0).num_rows for x in fs)
print(f"MIX_ALL_FEATURES eps={info['total_episodes']} frames={info['total_frames']} files={len(fs)} missing_in_features={missing_f} missing_in_parquet={missing_p} max_rowgroup={maxrg} depth_streams={[k for k in f if 'depth_linear' in k]}")
print("MIX_ALL_OK" if not missing_f and not missing_p and maxrg <= 16384 else "MIX_ALL_FAILED", flush=True)
PYX
grep -q MIX_ALL_OK $L/prep_all.out || { say "MIX_ALL_FAILED"; exit 1; }
# ---- 5. warm start, norm stats, twin map, parity, smoke --------------------------------------------------------------------
mkdir -p /root/ckpt_4d_init && ln -sfn /root/run3_dl/full/params /root/ckpt_4d_init/params
for n in pi05_radio_full pi05_radio_geo pi05_radio_4d pi05_radio_4d_pd pi05_radio_4d_all pi05_radio_4d_allf; do mkdir -p outputs/assets/$n && cp -r /root/run3_dl/a4/assets/* outputs/assets/$n/; done
PARITY_EXTRA=pi05_radio_4d_all,pi05_radio_4d_allf $PY $R3/parity_smoke_4d.py > $L/parity_all.log 2>&1; grep -E "PARITY|Traceback|Error" $L/parity_all.log | tail -8
( while true; do a=$(grep -E "^anon " /sys/fs/cgroup/memory.stat | awk '{print $2}'); c=$(cat /sys/fs/cgroup/memory.current); echo "$a $c"; sleep 5; done ) > $L/smoke_all_mem.samples 2>/dev/null &
SAMPLER=$!
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 B1K_STAGE_OVERSAMPLE=8 B1K_SAMPLE_WEIGHT_COL=sample_weight .venv/bin/python scripts/b1k/train_b1k.py pi05_radio_4d_all --exp_name=smoke_all --overwrite --num_train_steps=40 --save_interval=1000 --no-wandb-enabled > $L/smoke_all.log 2>&1
kill $SAMPLER 2>/dev/null
grep -oE "\[stage-oversample\][^\n]{0,60}|\[sample-weight\][^\n]{0,60}" $L/smoke_all.log | head -2
grep -oE "loss=[0-9.]+|grad_norm=[0-9.]+|Traceback|RESOURCE_EXHAUSTED|Killed" $L/smoke_all.log | tail -6 | tr "\n" " "; echo
awk 'BEGIN{ma=0;mc=0} {if($1>ma)ma=$1; if($2>mc)mc=$2} END{printf "SMOKE_MEM max anon %.0f GB, max cgroup current %.0f GB, samples %d\n", ma/2^30, mc/2^30, NR}' $L/smoke_all_mem.samples
rm -rf outputs/checkpoints/pi05_radio_4d_all/smoke_all
say PREP_ALL_DONE

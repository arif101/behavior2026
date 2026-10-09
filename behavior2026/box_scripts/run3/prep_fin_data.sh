#!/bin/bash
# FINISH-ARM DATA PREP (2026-10-09): build /root/b1k_radio_mix_fin = mix_all (map + factory + episodes + all 21 DART/ODART
# roots, 498 eps / 557,541 frames) + the 17 on-policy FINISH roots (b1k_radio_finish_r*, 273 strict clips / 76,094 frames from
# the stall-state factory, 10-02..04). Derived from prep_all_data.sh: every one of the 498 already-processed episodes gets its
# derived columns (v2 labels, gists, cam_pose, hist tokens) copied back from the mix_all BACKUP by unassemble_columns.py; only
# the 17 finish roots are labeled/precomputed here. Weights: map poison 0.1, factory 4.78, episodes 2.0, corrective 4.78,
# FINISH W_FINISH (default 4.78 = the corrective weight -> the finish clips are ~27% of the weighted mass; B1K_STAGE_OVERSAMPLE=8
# multiplies on their own stage transitions at launch). Every 09-25 lesson kept: arrow-native hist writes, 16k-row row groups,
# canonical column order, atomic replace, FULL warm start for parity, norm-stats assets per config name, explicit markers.
# Prereqs (trainer_bringup_fin.sh): fork venv at /root/openpi_fork (configs incl. pi05_radio_4d_fin), /root/run3/*.py helpers,
# /root/.hf_token, downloads under /root/backup (map), /root/manufactured (factory, episodes, poison_windows.json, all dart/odart
# AND finish roots), /root/mixes_bk/{mix_all,fk,fk_dart_r1..3,fk_<rootname>...}, /root/run3_dl/{a4,full}. Log: /root/run3_logs/prep_fin.out
set -u
PY=/root/openpi_fork/.venv/bin/python; R3=/root/run3; L=/root/run3_logs; OUT=$L/prep_fin.out; mkdir -p $L /root/frame_cache
say(){ echo "[prep_fin $(date -u +%m-%dT%H:%M:%S)] $*"; }
W_FINISH=${W_FINISH:-4.78}
MAP=/root/backup/b1k_radio_map; FAC=/root/manufactured/b1k_radio_factory; EPI=/root/manufactured/b1k_radio_episodes; M=/root/manufactured
OLD="$M/b1k_radio_dart_r1 $M/b1k_radio_dart_r2 $M/b1k_radio_dart_r3"
NEW="$M/b1k_radio_dart_r4 $M/b1k_radio_dart_r5 $M/b1k_radio_dart_r6 $M/b1k_radio_dart_r6_0925_1457 $M/b1k_radio_dart_r7 $M/b1k_radio_dart_r8 $M/b1k_radio_dart_r8_09261035 $M/b1k_radio_dart_r9_09261201 $(ls -d $M/b1k_radio_odart_r* | sort -V | tr '\n' ' ')"
FIN="$(ls -d $M/b1k_radio_finish_r* 2>/dev/null | sort -V | tr '\n' ' ')"
say "finish roots: $(echo $FIN | wc -w) (expect 17); W_FINISH=$W_FINISH"
[ "$(echo $FIN | wc -w)" -eq 17 ] || { say "FINISH_ROOTS_MISSING"; exit 1; }
for D in $OLD $NEW $FIN; do [ -d $D/data ] || { say "MISSING_ROOT $D"; exit 1; }; done
fkdir(){ n=$(basename $1); case $n in b1k_radio_dart_r1) echo /root/mixes_bk/fk_dart_r1;; b1k_radio_dart_r2) echo /root/mixes_bk/fk_dart_r2;; b1k_radio_dart_r3) echo /root/mixes_bk/fk_dart_r3;; *) echo /root/mixes_bk/fk_$n;; esac; }
for D in $FIN; do [ -f $(fkdir $D)/cam_pose.npy ] || { say "MISSING_FK $(fkdir $D)"; exit 1; }; done
[ -f /root/mixes_bk/mix_all/meta/info.json ] || { say "MISSING_MIX_ALL_BACKUP"; exit 1; }
# ---- 1. sample weights ------------------------------------------------------------------------------------------------
$PY $R3/register_feature.py --root $MAP --name gt_depth_ds --shape 768 2>&1 | tail -1
$PY $R3/add_sample_weights.py --root $MAP --poison $M/poison_windows.json --overwrite-col | tail -1
$PY $R3/add_sample_weights.py --root $FAC --default-weight 4.78 --overwrite-col | tail -1
$PY $R3/add_sample_weights.py --root $EPI --default-weight 2.0 --overwrite-col | tail -1
for D in $OLD $NEW; do $PY $R3/add_sample_weights.py --root $D --default-weight 4.78 --overwrite-col | tail -1; done
for D in $FIN; do $PY $R3/add_sample_weights.py --root $D --default-weight $W_FINISH --overwrite-col | tail -1; done
# ---- 2. the 498 processed episodes: copy every derived column back from the mix_all backup (v2 labels, gists, cam_pose,
#         hist tokens). mix_all's episode order = exactly $MAP $FAC $EPI $OLD $NEW (prep_all_data.sh step 4). ---------------
$PY $R3/unassemble_columns.py --mix /root/mixes_bk/mix_all --sources $MAP $FAC $EPI $OLD $NEW \
  --cols stage_v2 progress toggled target_points_v2 gist_head hist_geo cam_pose hist_tok hist_cellxyz hist_cellvalid odom_xyyaw 2>&1 | tail -25
grep -q UNASSEMBLE_OK $OUT || { say "UNASSEMBLE_FAILED"; exit 1; }
# ---- 3. finish roots: v2 labels, gists (A4 tower), cam_pose, frame cache + hist tokens (FULL tower) --------------------
cd /root/openpi_fork
for D in $FIN; do
  n=$(basename $D); say "labels: $n"
  $PY $R3/relabel_v2.py --root $D --press-anchor none 2>&1 | tail -1
  XLA_PYTHON_CLIENT_PREALLOCATE=false $PY $R3/precompute_gists.py --root $D --params /root/run3_dl/a4/params --cache-frames /root/frame_cache/$n.npy 2>&1 | grep -E "PRECOMPUTE_GISTS_OK|Traceback" | tail -1
  $PY $R3/add_cam_pose_column.py --root $D --npy $(fkdir $D)/cam_pose.npy --index $(fkdir $D)/index.npy | tail -1
  XLA_PYTHON_CLIENT_PREALLOCATE=false $PY $R3/precompute_hist_tokens.py --root $D --params /root/run3_dl/full/params --from-cache /root/frame_cache/$n.npy 2>&1 | grep -E "PRECOMPUTE_HIST_TOKENS_OK|Traceback" | tail -1
  rm -f /root/frame_cache/$n.npy
done
for D in $FIN; do n=$(basename $D); grep -q "PRECOMPUTE_HIST_TOKENS_OK $n" $OUT || { say "FIN_ROOT_INCOMPLETE $n (hist tokens)"; exit 1; }; grep -q "PRECOMPUTE_GISTS_OK $n" $OUT || { say "FIN_ROOT_INCOMPLETE $n (gists)"; exit 1; }; done
[ "$(grep -c "CAM_POSE_OK rows=[0-9]* missing=0" $OUT)" -ge "$(echo $FIN | wc -w)" ] || { say "CAM_POSE_INCOMPLETE (a finish root has missing rows)"; exit 1; }
# ---- 4. canonical column order (the assembler asserts schema equality) and assemble ----------------------------------------
ORDER="$($PY - <<PYX
import glob, pyarrow.parquet as pq
names = pq.ParquetFile(sorted(glob.glob("$MAP/data/**/*.parquet", recursive=True))[0]).schema_arrow.names
add = ["sample_weight","gt_depth_ds","stage_v2","progress","toggled","target_points_v2","gist_head","hist_geo","cam_pose","hist_tok","hist_cellxyz","hist_cellvalid","odom_xyyaw"]
print(" ".join([c for c in names if c not in add] + add))
PYX
)"
for D in $MAP $FAC $EPI $OLD $NEW $FIN; do $PY $R3/canonicalize_columns.py --root $D --order $ORDER | tail -1; done
$PY $R3/assemble_run3_mix.py --overwrite --sources $MAP $FAC $EPI $OLD $NEW $FIN --out /root/b1k_radio_mix_fin 2>&1 | grep -E "ASSEMBLED|Traceback|Error|assert" | tail -3
grep -q "ASSEMBLED /root/b1k_radio_mix_fin" $OUT || { say "ASSEMBLE_FAILED"; exit 1; }
$PY $R3/deregister_depth_streams.py --root /root/b1k_radio_mix_fin | tail -1
$PY $R3/regroup_parquets.py --root /root/b1k_radio_mix_fin --rows 16384 2>&1 | tail -2
$PY - <<'PYX'
import glob, json, pyarrow.parquet as pq
info = json.load(open("/root/b1k_radio_mix_fin/meta/info.json")); f = info["features"]
fs = sorted(glob.glob("/root/b1k_radio_mix_fin/data/**/*.parquet", recursive=True))
need = ["sample_weight","gt_depth_ds","stage_v2","progress","toggled","target_points_v2","gist_head","hist_geo","cam_pose","hist_tok","hist_cellxyz","hist_cellvalid","odom_xyyaw"]
missing_f = [c for c in need if c not in f]; missing_p = [c for c in need if c not in pq.ParquetFile(fs[0]).schema_arrow.names]
maxrg = max(pq.ParquetFile(x).metadata.row_group(0).num_rows for x in fs)
print(f"MIX_FIN_FEATURES eps={info['total_episodes']} frames={info['total_frames']} files={len(fs)} missing_in_features={missing_f} missing_in_parquet={missing_p} max_rowgroup={maxrg} depth_streams={[k for k in f if 'depth_linear' in k]} (expect 771 eps / 633635 frames)")
print("MIX_FIN_OK" if not missing_f and not missing_p and maxrg <= 16384 and info['total_episodes'] == 771 else "MIX_FIN_FAILED", flush=True)
PYX
grep -q MIX_FIN_OK $OUT || { say "MIX_FIN_FAILED"; exit 1; }
# ---- 5. warm start, norm stats, parity, smoke -------------------------------------------------------------------------------
mkdir -p /root/ckpt_4d_init && ln -sfn /root/run3_dl/full/params /root/ckpt_4d_init/params
for n in pi05_radio_full pi05_radio_geo pi05_radio_4d pi05_radio_4d_pd pi05_radio_4d_all pi05_radio_4d_allf pi05_radio_4d_fin; do mkdir -p outputs/assets/$n && cp -r /root/run3_dl/a4/assets/* outputs/assets/$n/; done
PARITY_LOADER_CFG=pi05_radio_4d_fin PARITY_EXTRA=pi05_radio_4d_all,pi05_radio_4d_fin $PY $R3/parity_smoke_4d.py > $L/parity_fin.log 2>&1; grep -E "PARITY|Traceback|Error" $L/parity_fin.log | tail -8
grep -q "PARITY_RESULT" $L/parity_fin.log || { say "PARITY_FAILED (no PARITY_RESULT line; see parity_fin.log)"; exit 1; }
# cgroup v2 (memory.stat anon / memory.current) or v1 (memory/memory.stat total_rss / memory.usage_in_bytes)
( while true; do if [ -f /sys/fs/cgroup/memory.current ]; then a=$(grep -E "^anon " /sys/fs/cgroup/memory.stat | awk '{print $2}'); c=$(cat /sys/fs/cgroup/memory.current); else a=$(grep -E "^total_rss " /sys/fs/cgroup/memory/memory.stat | awk '{print $2}'); c=$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes); fi; echo "$a $c"; sleep 5; done > $L/smoke_fin_mem.samples ) &
SAMPLER=$!
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 B1K_STAGE_OVERSAMPLE=8 B1K_SAMPLE_WEIGHT_COL=sample_weight .venv/bin/python scripts/b1k/train_b1k.py pi05_radio_4d_fin --exp_name=smoke_fin --overwrite --num_train_steps=40 --save_interval=1000 --no-wandb-enabled > $L/smoke_fin.log 2>&1
kill $SAMPLER 2>/dev/null
grep -oE "\[stage-oversample\][^\n]{0,60}|\[sample-weight\][^\n]{0,60}" $L/smoke_fin.log | head -2
grep -oE "loss=[0-9.]+|grad_norm=[0-9.]+|Traceback|RESOURCE_EXHAUSTED|Killed" $L/smoke_fin.log | tail -6 | tr "\n" " "; echo
awk 'BEGIN{ma=0;mc=0} {if($1>ma)ma=$1; if($2>mc)mc=$2} END{printf "SMOKE_MEM max anon %.0f GB, max cgroup current %.0f GB, samples %d\n", ma/2^30, mc/2^30, NR}' $L/smoke_fin_mem.samples
rm -rf outputs/checkpoints/pi05_radio_4d_fin/smoke_fin
say PREP_FIN_DONE

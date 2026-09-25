#!/bin/bash
# 4D DATA PREP (2026-09-25): build /root/b1k_radio_mix_4d = map + factory + episodes + DART r1..r3 (approach_v2 DROPPED),
# every source carrying: sample_weight, gt_depth_ds, stage_v2/progress/toggled/target_points_v2, gist_head/hist_geo,
# cam_pose. Then hist tokens on the mix, norm stats, parity smoke, 40-step GPU smoke. Waits for dl_dart.py.
# Usage: setsid nohup bash /root/prep_4d_data.sh > /root/run3_logs/prep_4d.out 2>&1 &
set -u
PY=/root/openpi_fork/.venv/bin/python; R3=/root/run3; L=/root/run3_logs; mkdir -p $L
say(){ echo "[prep4d $(date -u +%m-%dT%H:%M:%S)] $*"; }
MAP=/root/backup/b1k_radio_map; FAC=/root/manufactured/b1k_radio_factory; EPI=/root/manufactured/b1k_radio_episodes
until grep -q DL_DART_DONE $L/dl_dart.log 2>/dev/null; do sleep 60; done; say "downloads done"
DARTS=$(ls -d /root/manufactured/b1k_radio_dart_r* | sort -V | tr "\n" " "); say "DART roots: $DARTS"
# ---- 1. sample weights (pre-registered recipe) ------------------------------------------------------------------
$PY $R3/register_feature.py --root $MAP --name gt_depth_ds --shape 768 2>&1 | tail -1
$PY $R3/add_sample_weights.py --root $MAP --poison /root/manufactured/poison_windows.json --overwrite-col | tail -1
$PY $R3/add_sample_weights.py --root $FAC --default-weight 4.78 --overwrite-col | tail -1
$PY $R3/add_sample_weights.py --root $EPI --default-weight 2.0 --overwrite-col | tail -1
for D in $DARTS; do $PY $R3/add_sample_weights.py --root $D --default-weight 4.78 --overwrite-col | tail -1; done
# ---- 2. v2 labels + gists + cam_pose: map/factory/episodes from the backed-up mix; DART computed --------------------
$PY $R3/unassemble_columns.py --mix /root/mixes_bk/mix_full --fk /root/mixes_bk/fk --sources $MAP $FAC $EPI 2>&1 | tail -4
for D in $DARTS; do
  k=${D##*_r}
  $PY $R3/relabel_v2.py --root $D --press-anchor none 2>&1 | tail -2
  XLA_PYTHON_CLIENT_PREALLOCATE=false $PY $R3/precompute_gists.py --root $D --params /root/run3_dl/a4/params 2>&1 | grep -E "PRECOMPUTE_GISTS_OK|Traceback" | tail -1
  $PY $R3/add_cam_pose_column.py --root $D --npy /root/mixes_bk/fk_dart_r$k/cam_pose.npy --index /root/mixes_bk/fk_dart_r$k/index.npy | tail -1
done
# ---- 3. canonical column order (assembler asserts schema equality), then assemble ------------------------------------
ORDER="$($PY - <<PYX
import glob, pyarrow.parquet as pq
names = pq.ParquetFile(sorted(glob.glob("$MAP/data/**/*.parquet", recursive=True))[0]).schema_arrow.names
want = [c for c in names if c not in ("sample_weight","gt_depth_ds","stage_v2","progress","toggled","target_points_v2","gist_head","hist_geo","cam_pose")] + ["sample_weight","gt_depth_ds","stage_v2","progress","toggled","target_points_v2","gist_head","hist_geo","cam_pose"]
print(" ".join(want))
PYX
)"
for D in $MAP $FAC $EPI $DARTS; do $PY $R3/canonicalize_columns.py --root $D --order $ORDER | tail -1; done
$PY $R3/assemble_run3_mix.py --overwrite --sources $MAP $FAC $EPI $DARTS --out /root/b1k_radio_mix_4d 2>&1 | grep -E "ASSEMBLED|Traceback|Error|assert" | tail -3
$PY $R3/deregister_depth_streams.py --root /root/b1k_radio_mix_4d | tail -1
$PY - <<'PYX'
import json, glob, pyarrow.parquet as pq
info = json.load(open("/root/b1k_radio_mix_4d/meta/info.json")); f = info["features"]
cols = pq.ParquetFile(sorted(glob.glob("/root/b1k_radio_mix_4d/data/**/*.parquet", recursive=True))[0]).schema_arrow.names
need = ["sample_weight","gt_depth_ds","stage_v2","progress","toggled","target_points_v2","gist_head","hist_geo","cam_pose"]
print("MIX4D_FEATURES missing_in_features", [c for c in need if c not in f], "missing_in_parquet", [c for c in need if c not in cols], "depth_streams", [k for k in f if "depth_linear" in k])
PYX
say "mix_4d assembled: $($PY -c "import json; i=json.load(open('/root/b1k_radio_mix_4d/meta/info.json')); print(i['total_episodes'], 'eps', i['total_frames'], 'frames')")"
# ---- 4. history tokens on the mix (frame cache first) -------------------------------------------------------------
mkdir -p /root/frame_cache
XLA_PYTHON_CLIENT_PREALLOCATE=false $PY $R3/precompute_gists.py --root /root/b1k_radio_mix_4d --params /root/run3_dl/a4/params --cache-frames /root/frame_cache/mix_4d.npy 2>&1 | grep -E "PRECOMPUTE_GISTS_OK|Traceback|frame cache" | tail -2
# history tokens come from the FULL tower (= the warm start; hist_in is identity-init, so the tokens must live in the
# starting tower's space; the serve wrapper computes them from the live tower). Gists stay A4 (the gist_head consumer was trained on A4 gists).
[ -f /root/run3_dl/full/params/_METADATA ] || [ -d /root/run3_dl/full/params ] || { say "FULL_PARAMS_MISSING"; exit 1; }
XLA_PYTHON_CLIENT_PREALLOCATE=false $PY $R3/precompute_hist_tokens.py --root /root/b1k_radio_mix_4d --params /root/run3_dl/full/params --from-cache /root/frame_cache/mix_4d.npy 2>&1 | grep -E "PRECOMPUTE_HIST_TOKENS_OK|Traceback" | tail -1 | tee $L/hist_tok_4d.status
grep -q PRECOMPUTE_HIST_TOKENS_OK $L/hist_tok_4d.status && rm -f /root/frame_cache/mix_4d.npy && say "frame cache removed"
# ---- 5. warm start + norm stats -------------------------------------------------------------------------------------
mkdir -p /root/ckpt_4d_init && ln -sfn /root/run3_dl/full/params /root/ckpt_4d_init/params
for n in pi05_radio_full pi05_radio_geo pi05_radio_4d; do mkdir -p /root/openpi_fork/outputs/assets/$n && cp -r /root/run3_dl/a4/assets/* /root/openpi_fork/outputs/assets/$n/; done
md5sum /root/openpi_fork/outputs/assets/pi05_radio_4d/b1k_radio/norm_stats.json
# ---- 6. parity smoke + 40-step GPU smoke --------------------------------------------------------------------------
cd /root/openpi_fork
$PY $R3/parity_smoke_4d.py 2>&1 | grep -E "PARITY|Traceback|Error" | tail -5
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 B1K_STAGE_OVERSAMPLE=8 B1K_SAMPLE_WEIGHT_COL=sample_weight .venv/bin/python scripts/b1k/train_b1k.py pi05_radio_4d --exp_name=smoke4d --overwrite --num_train_steps=40 --save_interval=1000 --no-wandb-enabled > $L/smoke4d.log 2>&1
grep -oE "loss=[0-9.]+|grad_norm=[0-9.]+|Traceback|RESOURCE_EXHAUSTED" $L/smoke4d.log | tail -6 | tr "\n" " "; echo
say PREP_4D_DONE

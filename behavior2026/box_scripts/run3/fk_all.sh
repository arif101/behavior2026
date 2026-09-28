#!/bin/bash
# FK camera poses for EVERY corrective-clip root that lacks them (2026-09-28): hand-DART r4..r9 (+ the recovered 14-ep r6 and
# the 9-ep HF r8, fetched here because the local r6/r8 dirs differ) and ODART r1..r10. ONE Isaac boot: all roots' proprio
# are concatenated (ranges.json keeps the row spans), fk_cam_poses.py runs once (its unique-joint-config cache is shared),
# then cam_pose/index are split per root into /root/fk_<root>/ and uploaded to HF b26-run3-mixes/fk_<root>/ (full root
# name -> unambiguous; fk_dart_r1..r3 keep their old names). Usage: setsid nohup bash /root/fk_all.sh > /root/fk_all.out 2>&1 &
set -u; say(){ echo "[fk_all $(date -u +%m-%dT%H:%M:%S)] $*"; }
PY=/root/miniconda3/envs/behavior/bin/python; PYO=/root/openpi_fork/.venv/bin/python; W=/root/fk_all; mkdir -p $W /root/hf_fk_fetch
$PYO - <<'PYX'
import os; os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
from huggingface_hub import snapshot_download
tok = open("/root/.hf_token").read().strip()
for r in ("b1k_radio_dart_r6_0925_1457", "b1k_radio_dart_r8"):
    snapshot_download("arif101/b26-radio-manufactured", repo_type="dataset", token=tok, local_dir="/root/hf_fk_fetch", allow_patterns=[f"{r}/data/**", f"{r}/meta/**"])
    print("FETCHED", r, flush=True)
PYX
ROOTS="/root/b1k_radio_dart_r4 /root/b1k_radio_dart_r5 /root/b1k_radio_dart_r6 /root/hf_fk_fetch/b1k_radio_dart_r6_0925_1457 /root/b1k_radio_dart_r7 /root/hf_fk_fetch/b1k_radio_dart_r8 /root/b1k_radio_dart_r8_09261035 /root/b1k_radio_dart_r9_09261201 $(ls -d /root/b1k_radio_odart_r* | sort -V | tr '\n' ' ')"
say "roots: $ROOTS"
$PYO - $W $ROOTS <<'PYX'
import sys, glob, json, numpy as np, pyarrow.parquet as pq
W = sys.argv[1]; roots = sys.argv[2:]; P = []; I = []; ranges = {}; n = 0
for root in roots:
    name = root.rstrip("/").split("/")[-1]; rows = 0
    for f in sorted(glob.glob(f"{root}/data/**/*.parquet", recursive=True)):
        t = pq.read_table(f, columns=["observation.state", "episode_index", "frame_index"])
        st = np.stack(t.column("observation.state").to_pylist()).astype(np.float32)
        P.append(st); I.append(np.stack([np.asarray(t.column("episode_index").to_pylist()), np.asarray(t.column("frame_index").to_pylist())], 1)); rows += len(st)
    ranges[name] = [n, n + rows]; n += rows; print(f"  {name}: {rows} rows", flush=True)
np.save(f"{W}/proprio.npy", np.concatenate(P)); np.save(f"{W}/index.npy", np.concatenate(I).astype(np.int64)); json.dump(ranges, open(f"{W}/ranges.json", "w"), indent=1)
print("DUMPED", n, "rows across", len(roots), "roots", flush=True)
PYX
sed "s|/root/fk/|$W/|g" /root/fk_cam_poses.py > $W/fk_cam_poses.py
OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 OMNIGIBSON_HEADLESS=1 XDG_RUNTIME_DIR=/tmp/xdg CUDA_VISIBLE_DEVICES=0 \
  timeout 14400 $PY -u $W/fk_cam_poses.py 10 > $W/fk.log 2>&1
say "FK: $(grep -E "FK_DONE|Error" $W/fk.log | tail -1 | cut -c1-100)"
[ -f $W/cam_pose.npy ] || { say FK_ALL_FAILED; exit 1; }
$PYO - $W <<'PYX'
import os, sys, json, numpy as np; os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
from huggingface_hub import HfApi
W = sys.argv[1]; api = HfApi(token=open("/root/.hf_token").read().strip())
cp = np.load(f"{W}/cam_pose.npy"); ix = np.load(f"{W}/index.npy"); ranges = json.load(open(f"{W}/ranges.json"))
assert len(cp) == len(ix) == max(e for _, e in ranges.values()), (cp.shape, ix.shape)
for name, (a, b) in ranges.items():
    out = f"/root/fk_{name}"; os.makedirs(out, exist_ok=True)
    np.save(f"{out}/cam_pose.npy", cp[a:b]); np.save(f"{out}/index.npy", ix[a:b])
    bad = int((np.abs(cp[a:b, :3]).sum(1) == 0).sum())
    for f in ("cam_pose.npy", "index.npy"):
        api.upload_file(path_or_fileobj=f"{out}/{f}", path_in_repo=f"fk_{name}/{f}", repo_id="arif101/b26-run3-mixes", repo_type="dataset")
    print(f"FK_UPLOADED fk_{name} rows={b-a} zero_pose_rows={bad}", flush=True)
print("FK_ALL_UPLOADED", flush=True)
PYX
say FK_ALL_DONE

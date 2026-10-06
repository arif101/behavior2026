#!/bin/bash
# FK (camera-pose) pass over the 17 FINISH roots (2026-10-06), derived from /root/fk_all.sh (09-28): one Isaac boot, all
# roots' proprio concatenated (ranges.json keeps the row spans), fk_cam_poses.py once (shared unique-joint-config cache),
# then cam_pose/index split per root into /root/fk_<root>/ and uploaded to HF b26-run3-mixes/fk_<root>/.
# Usage: setsid nohup bash /root/fk_finish.sh > /root/fk_finish.out 2>&1 &
set -u; say(){ echo "[fk_finish $(date -u +%m-%dT%H:%M:%S)] $*"; }
PY=/root/miniconda3/envs/behavior/bin/python; PYO=/root/openpi_fork/.venv/bin/python; W=/root/fk_finish; mkdir -p $W
ROOTS="$(ls -d /root/b1k_radio_finish_r* | sort -V | tr '\n' ' ')"
say "roots ($(echo $ROOTS | wc -w)): $ROOTS"
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
[ -f $W/proprio.npy ] || { say FK_FINISH_FAILED dump; exit 1; }
sed "s|/root/fk/|$W/|g" /root/fk_cam_poses.py > $W/fk_cam_poses.py
OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 OMNIGIBSON_HEADLESS=1 XDG_RUNTIME_DIR=/tmp/xdg CUDA_VISIBLE_DEVICES=0 \
  timeout 14400 $PY -u $W/fk_cam_poses.py 10 > $W/fk.log 2>&1
say "FK: $(grep -E "FK_DONE|Error" $W/fk.log | tail -1 | cut -c1-100)"
[ -f $W/cam_pose.npy ] || { say FK_FINISH_FAILED fk; exit 1; }
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
grep -q FK_ALL_UPLOADED /root/fk_finish.out 2>/dev/null || true
say FK_FINISH_DONE

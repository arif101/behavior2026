"""Rewrite b1k_radio2's meta/ so it describes ONLY the episodes actually present.

add_target_points.py copies meta/ from the source dataset, so info.json still claims
total_episodes=20000 / total_frames=210,916,774 while the output holds 2 shards (200 episodes).
LeRobot builds its index from that meta, cannot find the other 19,800 episodes, raises, and falls
back to a Hub lookup — which surfaces as `RepositoryNotFoundError: 401 ... b1k_radio` and hides
the real cause. Every "root is not being passed" theory I chased came from that masking.

Also links the video shards into place: the converter does not copy them, and pi0.5 needs images.
"""

import json
import pathlib
import shutil

import numpy as np
import pyarrow.parquet as pq

ROOT = pathlib.Path("/root/b1k_radio2")
SRC = pathlib.Path("/root/b1k_radio")          # where the fetched videos already live

# --- what is actually in the data shards -------------------------------------------------
eps, frames = set(), 0
for f in sorted(ROOT.glob("data/chunk-*/file-*.parquet")):
    t = pq.read_table(f, columns=["episode_index"])
    ei = np.asarray(t["episode_index"])
    eps.update(int(x) for x in np.unique(ei))
    frames += len(ei)
eps = sorted(eps)
print(f"present: {len(eps)} episodes, {frames:,} frames  (index {eps[0]}..{eps[-1]})")

# --- trim info.json ----------------------------------------------------------------------
info_p = ROOT / "meta" / "info.json"
info = json.loads(info_p.read_text())
before = (info.get("total_episodes"), info.get("total_frames"))
info["total_episodes"] = len(eps)
info["total_frames"] = frames
if "splits" in info:
    info["splits"] = {"train": f"0:{len(eps)}"}
info_p.write_text(json.dumps(info, indent=4))
print(f"info.json: total_episodes {before[0]} -> {info['total_episodes']}, "
      f"total_frames {before[1]:,} -> {info['total_frames']:,}")

# --- trim meta/episodes to the present episodes -------------------------------------------
ep_dir = ROOT / "meta" / "episodes"
kept = dropped = 0
for f in sorted(ep_dir.rglob("*.parquet")):
    t = pq.read_table(f)
    ei = np.asarray(t["episode_index"])
    m = np.isin(ei, eps)
    if not m.any():
        f.unlink()
        dropped += 1
        continue
    pq.write_table(t.filter(m), f)
    kept += int(m.sum())
print(f"meta/episodes: kept {kept} rows, removed {dropped} empty shard files")

# --- videos: the converter does not copy them, and pi0.5 needs images ----------------------
src_v, dst_v = SRC / "videos", ROOT / "videos"
if src_v.is_dir() and not dst_v.exists():
    dst_v.symlink_to(src_v)
    n = len(list(src_v.rglob("*.mp4")))
    print(f"videos: symlinked {dst_v} -> {src_v}  ({n} mp4)")
else:
    print(f"videos: {'already present' if dst_v.exists() else 'NOT FOUND at ' + str(src_v)}")

print("META_FIXED")

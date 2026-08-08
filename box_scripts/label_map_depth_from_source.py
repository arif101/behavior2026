"""Add gt_depth_ds to /root/b1k_radio_map by decoding the ORGANIZERS' depth videos.

The box copy of b1k_radio_map is meta+data only (videos died with the old A5000). The
organizers' consolidated repo (behavior-1k/2026-challenge-demos) still carries every
depth stream. Local episode -> organizer video coordinates via raw_episode_id joined
against the organizers' meta/episodes (task_index == 0). Steps:
  1. fetch organizer meta/episodes parquets (small), filter task 0 (200 rows)
  2. join to local episodes meta on raw_episode_id -> per-episode per-stream
     (organizer chunk/file, from_timestamp) + local length
  3. download the unique organizer depth files (printed budget; abort > 60G)
  4. decode + 16x16 masked patch-mean pool (dequantize_depth, add_depth_aux_labels
     conventions verbatim) -> write gt_depth_ds into the local data parquet rows
Idempotent via column presence (--overwrite-col to redo). Token file-based only.
"""

import argparse
import glob
import json
import os
import pathlib

os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from lerobot.datasets.depth_utils import dequantize_depth

SRC_REPO = "behavior-1k/2026-challenge-demos"
PATCH = 16
DEPTH_STREAMS = [
    ("observation.depth_linear.zed_link_camera_0", 0),
    ("observation.depth_linear.left_realsense_link_camera_0", 1),
    ("observation.depth_linear.right_realsense_link_camera_0", 2),
]
DEPTH_MIN = 0.01
FPS = 30


def decode_depth_window(path, from_ts, n_frames):
    frames = {}
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        c.seek(int(from_ts * av.time_base), any_frame=False, backward=True)
        for fr in c.decode(stream):
            if fr.time is None:
                continue
            rel = int(round((fr.time - from_ts) * FPS))
            if rel < 0:
                continue
            if rel >= n_frames:
                break
            frames[rel] = dequantize_depth(fr, output_unit="m")
    if not frames:
        return None, 0
    H, W = next(iter(frames.values())).shape[:2]
    out = np.zeros((n_frames, H, W), np.float32)
    for rel, arr in frames.items():
        out[rel] = np.asarray(arr, np.float32).reshape(H, W)
    return out, len(frames)


def pool_patches(depth_hw):
    H, W = depth_hw.shape
    ph, pw = H // PATCH, W // PATCH
    d = depth_hw[: ph * PATCH, : pw * PATCH].reshape(PATCH, ph, PATCH, pw)
    valid = (d > DEPTH_MIN).astype(np.float32)
    s = (d * valid).sum((1, 3))
    c = valid.sum((1, 3))
    return np.where(c > 0, s / np.maximum(c, 1.0), 0.0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/b1k_radio_map")
    ap.add_argument("--cache", default="/root/orgdepth")
    ap.add_argument("--task-index", type=int, default=0)
    ap.add_argument("--budget-gb", type=float, default=60.0)
    ap.add_argument("--overwrite-col", action="store_true")
    a = ap.parse_args()
    root = pathlib.Path(a.root)
    tok = open("/root/.hf_token").read().strip()
    api = HfApi(token=tok)

    # ---- 1. organizer episodes meta -----------------------------------------------------
    meta_files = [f for f in api.list_repo_files(SRC_REPO, repo_type="dataset")
                  if f.startswith("meta/episodes/") and f.endswith(".parquet")]
    print(f"organizer episodes-meta files: {len(meta_files)}", flush=True)
    tabs = []
    need_cols = None
    for fn in meta_files:
        p = hf_hub_download(SRC_REPO, fn, repo_type="dataset", local_dir=a.cache, token=tok)
        t = pq.read_table(p)
        if need_cols is None:
            need_cols = [c for c in t.schema.names
                         if c in ("task_index", "raw_episode_id", "episode_index", "length")
                         or any(c == f"videos/{col}/{s}" for col, _ in DEPTH_STREAMS
                                for s in ("chunk_index", "file_index",
                                          "from_timestamp", "to_timestamp"))]
        tabs.append(t.select(need_cols))
    org = pa.concat_tables(tabs).to_pydict()
    sel = [i for i, ti in enumerate(org["task_index"]) if int(ti) == a.task_index]
    print(f"organizer rows for task_index={a.task_index}: {len(sel)}", flush=True)
    org_by_raw = {}
    for i in sel:
        org_by_raw[int(org["raw_episode_id"][i])] = {
            col: {
                "chunk": int(org[f"videos/{col}/chunk_index"][i]),
                "file": int(org[f"videos/{col}/file_index"][i]),
                "from_ts": float(org[f"videos/{col}/from_timestamp"][i]),
                "to_ts": float(org[f"videos/{col}/to_timestamp"][i]),
            } for col, _ in DEPTH_STREAMS
        }

    # ---- 2. local join ------------------------------------------------------------------
    lem = pq.read_table(sorted(glob.glob(
        str(root / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))[0]).to_pydict()
    episodes = []
    for i in range(len(lem["episode_index"])):
        raw = int(lem["raw_episode_id"][i])
        if raw not in org_by_raw:
            raise SystemExit(f"local episode {lem['episode_index'][i]} raw_id {raw} "
                             "not in organizer task rows -- join broken")
        episodes.append({"ep": int(lem["episode_index"][i]), "raw": raw,
                         "len": int(lem["length"][i]), "org": org_by_raw[raw]})
    print(f"joined {len(episodes)} local episodes on raw_episode_id", flush=True)
    for e in episodes[:3] + episodes[-1:]:
        col = DEPTH_STREAMS[0][0]
        o = e["org"][col]
        span = o["to_ts"] - o["from_ts"]
        if abs(span - e["len"] / FPS) > 0.5:
            raise SystemExit(f"ep{e['ep']}: organizer span {span:.1f}s != local "
                             f"{e['len'] / FPS:.1f}s -- join misaligned")
    print("span sanity OK (organizer segment lengths match local episode lengths)", flush=True)

    # ---- 3. download unique organizer depth files ---------------------------------------
    need = sorted({(col, e["org"][col]["chunk"], e["org"][col]["file"])
                   for e in episodes for col, _ in DEPTH_STREAMS})
    print(f"unique organizer depth files needed: {len(need)}", flush=True)
    got, total_b = {}, 0
    for col, ch, fi in need:
        fn = f"videos/{col}/chunk-{ch:03d}/file-{fi:03d}.mp4"
        p = hf_hub_download(SRC_REPO, fn, repo_type="dataset", local_dir=a.cache, token=tok)
        got[(col, ch, fi)] = p
        total_b += os.path.getsize(p)
        if total_b / 1e9 > a.budget_gb:
            raise SystemExit(f"download budget exceeded ({total_b / 1e9:.1f} GB)")
    print(f"downloaded {len(got)} files, {total_b / 1e9:.2f} GB", flush=True)

    # ---- 4. decode + label --------------------------------------------------------------
    ep_info = {e["ep"]: e for e in episodes}
    stats = {"decoded": 0, "expected": 0}
    for dfp in sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True)):
        t = pq.read_table(dfp)
        if "gt_depth_ds" in t.schema.names:
            if not a.overwrite_col:
                print(f"SKIP {dfp}: gt_depth_ds already present", flush=True)
                continue
            t = t.drop(["gt_depth_ds"])
        ep_col = t["episode_index"].to_numpy()
        fr_col = t["frame_index"].to_numpy()
        gt = np.zeros((t.num_rows, 3 * PATCH * PATCH), np.float32)
        for ep in np.unique(ep_col):
            e = ep_info[int(ep)]
            rows = np.where(ep_col == ep)[0]
            for col, slot in DEPTH_STREAMS:
                o = e["org"][col]
                dw, ndec = decode_depth_window(got[(col, o["chunk"], o["file"])],
                                               o["from_ts"], e["len"])
                stats["decoded"] += ndec
                stats["expected"] += e["len"]
                if dw is None:
                    print(f"  ep{ep} {col}: DECODE EMPTY", flush=True)
                    continue
                base = slot * PATCH * PATCH
                for r in rows:
                    f = int(fr_col[r])
                    if f < e["len"]:
                        gt[r, base: base + PATCH * PATCH] = pool_patches(dw[f]).ravel()
            print(f"  ep{ep}: {len(rows)} rows labeled", flush=True)
        arr = pa.array([row for row in gt], type=pa.list_(pa.float32(), 3 * PATCH * PATCH))
        t = t.append_column(pa.field("gt_depth_ds", pa.list_(pa.float32(), 3 * PATCH * PATCH)), arr)
        pq.write_table(t, dfp)
        vf = float((gt > 0).mean())
        print(f"wrote {dfp} (+gt_depth_ds, {t.num_rows} rows, valid_frac {vf:.4f})", flush=True)
    print(json.dumps({"frames_decoded": stats["decoded"],
                      "frames_expected": stats["expected"]}), flush=True)
    print("MAP_DEPTH_LABELS_DONE", flush=True)


if __name__ == "__main__":
    main()

"""Add the gt_depth_ds column (GT-depth aux labels, patch_depth_aux.py) to a LeRobot-v3 dataset.

Per row: (768,) float32 = 3 cameras x 16x16 patch-mean METRIC depth (meters), camera order
[head zed, left wrist, right wrist] = the prefix image-token order, row-major v-then-u per
camera; 0.0 = invalid (no valid depth ray in the patch / stream missing).

Decode: pyav -> lerobot.datasets.depth_utils.dequantize_depth (the EXACT inverse of the
12-bit log encoder used by both the organizers' pipeline and convert_clips_to_parquet.py;
defaults min 0.01 / max 10.0 / shift 3.5 / gray12le), output_unit="m". Patch pooling is a
masked mean over valid pixels (depth > min), so frustum edges and dropped rays do not drag
patch means toward zero.

Idempotent: skips a data file whose schema already has gt_depth_ds (--overwrite-col redoes).
Run over EVERY dataset in a depth_aux=True training mix (b1k_radio_map, b1k_radio_corrective,
b1k_radio_rac, ...) BEFORE enabling the flag — RepackTransform KeyErrors on absent columns.

Usage (box):
  /root/miniconda3/envs/openpi/bin/python add_depth_aux_labels.py --root /root/b1k_radio_corrective
"""

import argparse
import glob
import json
import pathlib

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets.depth_utils import dequantize_depth

PATCH = 16
# video column -> camera slot (prefix image-token order)
DEPTH_STREAMS = [
    ("observation.depth_linear.zed_link_camera_0", 0),
    ("observation.depth_linear.left_realsense_link_camera_0", 1),
    ("observation.depth_linear.right_realsense_link_camera_0", 2),
]
DEPTH_MIN = 0.01  # meters; dequantize default, also the validity floor


def decode_depth_window(path, from_ts, n_frames, fps):
    """(n_frames, H, W) float32 meters; missing frames stay 0 (invalid)."""
    frames = {}
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        c.seek(int(from_ts * av.time_base), any_frame=False, backward=True)
        for fr in c.decode(stream):
            if fr.time is None:
                continue
            rel = int(round((fr.time - from_ts) * fps))
            if rel < 0:
                continue
            if rel >= n_frames:
                break
            frames[rel] = dequantize_depth(fr, output_unit="m")
    if not frames:
        return None
    H, W = next(iter(frames.values())).shape[:2]
    out = np.zeros((n_frames, H, W), np.float32)
    for rel, arr in frames.items():
        out[rel] = np.asarray(arr, np.float32).reshape(H, W)
    return out, len(frames)


def pool_patches(depth_hw):
    """(H, W) meters -> (PATCH, PATCH) masked patch means; 0 where no valid pixel."""
    H, W = depth_hw.shape
    ph, pw = H // PATCH, W // PATCH
    d = depth_hw[: ph * PATCH, : pw * PATCH].reshape(PATCH, ph, PATCH, pw)
    valid = (d > DEPTH_MIN).astype(np.float32)
    s = (d * valid).sum((1, 3))
    c = valid.sum((1, 3))
    return np.where(c > 0, s / np.maximum(c, 1.0), 0.0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--overwrite-col", action="store_true")
    a = ap.parse_args()
    root = pathlib.Path(a.root)

    emeta = pq.read_table(sorted(glob.glob(
        str(root / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))[0]).to_pydict()
    n_eps = len(emeta["episode_index"])
    print(f"{root}: {n_eps} episodes")

    # per-episode video locations
    ep_video = {}
    for i in range(n_eps):
        ep = int(emeta["episode_index"][i])
        ep_video[ep] = {
            "data": (int(emeta["data/chunk_index"][i]), int(emeta["data/file_index"][i])),
            "streams": {
                col: (int(emeta[f"videos/{col}/chunk_index"][i]),
                      int(emeta[f"videos/{col}/file_index"][i]),
                      float(emeta[f"videos/{col}/from_timestamp"][i]),
                      float(emeta[f"videos/{col}/to_timestamp"][i]))
                for col, _ in DEPTH_STREAMS if f"videos/{col}/chunk_index" in emeta
            },
            "len": int(emeta["length"][i]),
        }

    stats = {"rows": 0, "valid_frac": [], "decoded": 0, "missing": 0}
    for dfp in sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True)):
        t = pq.read_table(dfp)
        if "gt_depth_ds" in t.schema.names:
            if not a.overwrite_col:
                print(f"SKIP {dfp}: gt_depth_ds already present")
                continue
            t = t.drop(["gt_depth_ds"])
        ep_col = t["episode_index"].to_numpy()
        fr_col = t["frame_index"].to_numpy()
        gt = np.zeros((t.num_rows, 3 * PATCH * PATCH), np.float32)
        for ep in np.unique(ep_col):
            rec = ep_video[int(ep)]
            rows = np.where(ep_col == ep)[0]
            n = rec["len"]
            for col, slot in DEPTH_STREAMS:
                if col not in rec["streams"]:
                    stats["missing"] += 1
                    continue
                ch, fi, fts, _ = rec["streams"][col]
                vp = root / "videos" / col / f"chunk-{ch:03d}" / f"file-{fi:03d}.mp4"
                if not vp.exists():
                    stats["missing"] += 1
                    continue
                dec = decode_depth_window(vp, fts, n, a.fps)
                if dec is None:
                    stats["missing"] += 1
                    continue
                dw, ndec = dec
                stats["decoded"] += ndec
                base = slot * PATCH * PATCH
                for r in rows:
                    f = int(fr_col[r])
                    if f < n:
                        p = pool_patches(dw[f])
                        gt[r, base: base + PATCH * PATCH] = p.ravel()
            print(f"  ep{ep}: {len(rows)} rows done", flush=True)
        stats["rows"] += t.num_rows
        stats["valid_frac"].append(float((gt > 0).mean()))
        arr = pa.array([row for row in gt], type=pa.list_(pa.float32(), 3 * PATCH * PATCH))
        t = t.append_column(pa.field("gt_depth_ds", pa.list_(pa.float32(), 3 * PATCH * PATCH)), arr)
        pq.write_table(t, dfp)
        print(f"wrote {dfp} (+gt_depth_ds, {t.num_rows} rows)")

    vf = float(np.mean(stats["valid_frac"])) if stats["valid_frac"] else 0.0
    print(json.dumps({"rows": stats["rows"], "mean_valid_frac": round(vf, 4),
                      "frames_decoded": stats["decoded"], "streams_missing": stats["missing"]}))
    if vf < 0.5:
        print("WARNING: mean valid fraction < 0.5 -- decode path or depth videos suspect")


if __name__ == "__main__":
    main()

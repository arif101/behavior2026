"""Overlay validation: project sweep labels through the calibrated zed intrinsics onto
video frames -- drawn point must sit on the target object.

Projection convention (USD/OpenGL camera: looks along -Z, +Y up; M = camera-to-world,
column-vector): p_cam = inv(M) @ p_world; u = cx + fx*(x/-z); v = cy - fy*(y/-z).
Labels fire 2x/frame -> record index 2*f for video frame f.

Usage (behavior env):
  python overlay_check.py <task_name> <out_png> [--episode-pos 0] [--n-frames 4]
"""

import argparse
import glob
import json
import os
import re
import subprocess

import numpy as np
from PIL import Image, ImageDraw

OUT_DIR = "/root/sweep_out"
LBL_DIR = "/root/sweep_labels"
INTR = "/root/probes/zed_intrinsics_calibrated.json"
CAM_KEY = "videos/observation.rgb.zed_link_camera_0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out_png")
    ap.add_argument("--episode-pos", type=int, default=0, help="position among done episodes (ascending demo id)")
    ap.add_argument("--n-frames", type=int, default=4)
    args = ap.parse_args()

    import pandas as pd

    with open(INTR) as f:
        K = json.load(f)
    fx, fy, cx, cy = K["fx"], K["fy"], K["cx"], K["cy"]

    # --- pick episode: done labels sorted by demo id ---
    dones = sorted(
        glob.glob(os.path.join(LBL_DIR, args.task, "labels_*.jsonl.done.json")),
        key=lambda p: int(re.search(r"labels_(\d+)\.jsonl", p).group(1)),
    )
    assert dones, f"no done labels for {args.task}"
    done = json.load(open(dones[args.episode_pos]))
    demo_id = done["demo_id"]
    ep_index = args.episode_pos  # episodes appended in ascending demo order
    labels_path = os.path.join(LBL_DIR, args.task, f"labels_{demo_id}.jsonl")
    records = [json.loads(l) for l in open(labels_path)]
    print(f"episode demo_id={demo_id} ep_index={ep_index} records={len(records)}")

    # --- lerobot meta: video file + start timestamp for this episode ---
    root = os.path.join(OUT_DIR, args.task, "b1k", args.task)
    eps = pd.concat([
        pd.read_parquet(p)
        for p in sorted(glob.glob(os.path.join(root, "meta", "episodes", "**", "*.parquet"), recursive=True))
    ]).sort_values("episode_index")
    row = eps[eps.episode_index == ep_index].iloc[0]
    length = int(row["length"])
    chunk = int(row[f"{CAM_KEY}/chunk_index"])
    fidx = int(row[f"{CAM_KEY}/file_index"])
    from_ts = float(row[f"{CAM_KEY}/from_timestamp"])
    fps = json.load(open(os.path.join(root, "meta", "info.json")))["fps"]
    video = os.path.join(root, CAM_KEY, f"chunk-{chunk:03d}", f"file-{fidx:03d}.mp4")
    start_frame = round(from_ts * fps)
    print(f"video={video} length={length} start_frame={start_frame} fps={fps}")

    frames = [int(length * p) for p in np.linspace(0.15, 0.9, args.n_frames)]
    tiles = []
    for f_i in frames:
        # extract exact frame (file holds concatenated episodes; absolute frame = start + f)
        tmp = f"/tmp/ovl_{args.task}_{f_i}.png"
        n_abs = start_frame + f_i
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", video,
             "-vf", f"select=eq(n\\,{n_abs})", "-vsync", "0", "-frames:v", "1", tmp],
            check=True,
        )
        img = Image.open(tmp).convert("RGB")
        draw = ImageDraw.Draw(img)

        rec = records[min(2 * f_i, len(records) - 1)]
        M = np.array(rec["M"]).reshape(4, 4)
        Minv = np.linalg.inv(M)
        for name, pw in rec["objs"].items():
            pc = Minv @ np.array([pw[0], pw[1], pw[2], 1.0])
            x, y, z = pc[:3]
            if z >= -0.05:  # behind/too close to camera plane
                continue
            u = cx + fx * (x / -z)
            v = cy - fy * (y / -z)
            if not (0 <= u < img.width and 0 <= v < img.height):
                draw.text((5, 5), f"{name} OFFSCREEN u={u:.0f} v={v:.0f}", fill="yellow")
                continue
            r = 8
            draw.line([(u - r, v), (u + r, v)], fill="red", width=3)
            draw.line([(u, v - r), (u, v + r)], fill="red", width=3)
            draw.ellipse([u - r, v - r, u + r, v + r], outline="lime", width=2)
            draw.text((u + r + 2, v - r), name, fill="lime")
        draw.text((5, img.height - 20), f"frame {f_i}/{length}", fill="white")
        tiles.append(img)

    w, h = tiles[0].size
    grid = Image.new("RGB", (w * 2, h * ((len(tiles) + 1) // 2)))
    for i, t in enumerate(tiles):
        grid.paste(t, ((i % 2) * w, (i // 2) * h))
    grid.save(args.out_png)
    print("OVERLAY_SAVED", args.out_png)


if __name__ == "__main__":
    main()

"""GT label overlay spot-check (no model): render 8 frames/episode with all
in-frame instances: green = visible (passes occlusion rule), orange = in-frame
but occluded. Mechanical companion to mt_validate_labels.py."""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mt_common import CACHE, IMG, W, instance_thresholds, visible_mask


def render(task, file_idx, out_png, n=8):
    from PIL import Image, ImageDraw
    d = os.path.join(CACHE, task, f"ep{file_idx:03d}")
    lab = np.load(os.path.join(d, "lab.npz"))
    margin = lab["margin"].astype(np.float32)
    thr = instance_thresholds(margin, lab["inframe"],
                              np.load(os.path.join(d, "disp.npy")))
    vis = visible_mask(lab["inframe"], margin, thr)
    names = lab["obj_names"]
    N = lab["frame_k"].shape[0]
    idxs = sorted(set(np.linspace(0, N - 1, n).astype(int)))
    s = IMG / W
    tiles = []
    for i in idxs:
        im = Image.open(os.path.join(d, "frames", f"f_{i:05d}.jpg")).convert("RGB")
        dr = ImageDraw.Draw(im)
        for j in range(len(names)):
            if not lab["inframe"][i, j]:
                continue
            u, v, z = lab["uvz"][i, j]
            gu, gv = u * s, v * s
            col = (0, 255, 0) if vis[i, j] else (255, 160, 0)
            dr.ellipse([gu - 8, gv - 8, gu + 8, gv + 8], outline=col, width=2)
            m = float(lab["margin"][i, j])
            dr.text((gu + 9, gv - 8), f"{names[j]}", fill=col)
            dr.text((gu + 9, gv + 2), f"z{z:.2f} m{m:+.2f}" if np.isfinite(m)
                    else f"z{z:.2f} m?", fill=col)
        dr.rectangle([0, 0, im.width, 18], fill=(0, 0, 0))
        dr.text((4, 3), f"{task} ep{file_idx} f{int(lab['frame_k'][i])}",
                fill=(255, 255, 0))
        tiles.append(im)
    cols, rows = 4, (len(tiles) + 3) // 4
    grid = Image.new("RGB", (cols * IMG, rows * IMG), (15, 15, 15))
    for i, im in enumerate(tiles):
        grid.paste(im, ((i % cols) * IMG, (i // cols) * IMG))
    grid.save(out_png)
    print("wrote", out_png)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--file", type=int, default=0)
    ap.add_argument("--out-dir", default="/root/grounding_v05/gt_checks")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for t in args.tasks:
        render(t, args.file, os.path.join(args.out_dir, f"gt_{t}_ep{args.file:03d}.png"))

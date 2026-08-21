"""Evaluate the M1 pilot grounding head: honest held-out metrics + overlay grid.

Metrics (720p pixel frame): median/p90 pixel error of heatmap argmax vs GT,
% within 30px, median |depth error| at the predicted pixel, and median 3D error
(pred vs GT points compared in camera frame -- rigid-invariant, equals world error).
"""

import argparse
import json

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from dataset import (GroundingDataset, TRAIN_EPS, HELDOUT_EPS, DEMO_IDS,
                     FX, FY, CX, CY, W, IMG)
from model import GroundingModel


def cam_point(u, v, z):
    x = (u - CX) * z / FX
    y = (CY - v) * z / FY
    return np.stack([x, y, -z], -1)


@torch.no_grad()
def run_split(model, episodes, bs=64, workers=8, dev="cuda"):
    ds = GroundingDataset(episodes)
    dl = DataLoader(ds, batch_size=bs, num_workers=workers, pin_memory=True)
    recs = []
    for batch in dl:
        rgb = batch["rgb"].to(dev, non_blocking=True)
        depth = batch["depth"].to(dev, non_blocking=True)
        cat = batch["cat"].to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, dmap = model(rgb, depth, cat)
        logits, dmap = logits.float(), dmap.float()
        uv_pred = model.argmax_uv(logits)
        uv_soft = model.soft_argmax_uv(logits)
        z_pred = model.sample_depth(dmap, uv_pred)
        uv_gt = batch["uv"].numpy()
        z_gt = batch["z"].numpy()
        up = uv_pred.cpu().numpy()
        us = uv_soft.cpu().numpy()
        zp = z_pred.cpu().numpy()
        px = np.linalg.norm(up - uv_gt, axis=1)
        pxs = np.linalg.norm(us - uv_gt, axis=1)
        d3 = np.linalg.norm(cam_point(up[:, 0], up[:, 1], zp)
                            - cam_point(uv_gt[:, 0], uv_gt[:, 1], z_gt), axis=1)
        for i in range(len(px)):
            recs.append({"ep": int(batch["ep"][i]), "frame": int(batch["frame"][i]),
                         "px": float(px[i]), "px_soft": float(pxs[i]),
                         "dz": float(abs(zp[i] - z_gt[i])),
                         "d3": float(d3[i]),
                         "uv_gt": uv_gt[i].tolist(), "uv_pred": up[i].tolist(),
                         "z_gt": float(z_gt[i]), "z_pred": float(zp[i])})
    return recs


def summarize(recs):
    px = np.array([r["px"] for r in recs])
    dz = np.array([r["dz"] for r in recs])
    d3 = np.array([r["d3"] for r in recs])
    out = {"n": len(recs),
           "px_median": round(float(np.median(px)), 2),
           "px_p90": round(float(np.percentile(px, 90)), 2),
           "within_30px_pct": round(float((px < 30).mean() * 100), 1),
           "depth_abs_median_m": round(float(np.median(dz)), 4),
           "err3d_median_m": round(float(np.median(d3)), 4)}
    if recs and "px_soft" in recs[0]:
        pxs = np.array([r["px_soft"] for r in recs])
        out["px_soft_median"] = round(float(np.median(pxs)), 2)
        out["px_soft_p90"] = round(float(np.percentile(pxs, 90)), 2)
    return out


def overlay_grid(recs, out_png, n=6):
    """n frames spread across held-out episodes: GT green circle, pred red cross."""
    from dataset import CACHE
    by_ep = {}
    for r in recs:
        by_ep.setdefault(r["ep"], []).append(r)
    picks = []
    eps = sorted(by_ep)
    per = max(1, n // len(eps))
    for ei in eps:
        rs = sorted(by_ep[ei], key=lambda r: r["frame"])
        idxs = np.linspace(0, len(rs) - 1, per).astype(int)
        picks += [rs[i] for i in idxs]
    picks = picks[:n]

    s = IMG / W
    tiles = []
    for r in picks:
        lab = np.load(f"{CACHE}/ep{r['ep']:03d}_lab.npy")
        li = int(np.where(lab[:, 0] == r["frame"])[0][0])
        rgb = np.load(f"{CACHE}/ep{r['ep']:03d}_rgb.npy", mmap_mode="r")[li]
        im = Image.fromarray(np.ascontiguousarray(rgb))
        dr = ImageDraw.Draw(im)
        gu, gv = r["uv_gt"][0] * s, r["uv_gt"][1] * s
        pu, pv = r["uv_pred"][0] * s, r["uv_pred"][1] * s
        rad = 9
        dr.ellipse([gu - rad, gv - rad, gu + rad, gv + rad], outline=(0, 255, 0), width=3)
        dr.line([pu - rad, pv, pu + rad, pv], fill=(255, 40, 40), width=3)
        dr.line([pu, pv - rad, pu, pv + rad], fill=(255, 40, 40), width=3)
        dr.text((6, 6), f"ep{r['ep']:03d} demo{DEMO_IDS[r['ep']]} f{r['frame']} "
                        f"px={r['px']:.0f} dz={r['dz']*100:.1f}cm", fill=(255, 255, 0))
        tiles.append(im)

    cols, rows = 3, (len(tiles) + 2) // 3
    grid = Image.new("RGB", (cols * IMG, rows * IMG))
    for i, im in enumerate(tiles):
        grid.paste(im, ((i % cols) * IMG, (i // cols) * IMG))
    grid.save(out_png)
    print(f"wrote {out_png}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/root/grounding/ckpt_pilot.pt")
    ap.add_argument("--out", default="/root/grounding/results_pilot.json")
    ap.add_argument("--grid", default="/root/grounding/overlay_grid.png")
    args = ap.parse_args()

    dev = "cuda"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = GroundingModel(**{k: v for k, v in ck["config"].items() if k != "backbone"},
                           backbone=ck["config"]["backbone"]).to(dev)
    missing, unexpected = model.load_state_dict(ck["head"], strict=False)
    assert not unexpected and all(k.startswith("backbone.") for k in missing)
    model.eval()

    results = {"backbone": ck["config"]["backbone"],
               "trainable_params": ck.get("trainable_params"),
               "train_minutes": round(ck.get("train_minutes", -1), 1),
               "heldout_eps": HELDOUT_EPS,
               "heldout_demos": [DEMO_IDS[e] for e in HELDOUT_EPS]}

    ho_recs = run_split(model, HELDOUT_EPS)
    tr_recs = run_split(model, TRAIN_EPS)
    results["heldout"] = summarize(ho_recs)
    results["train"] = summarize(tr_recs)
    results["heldout_per_episode"] = {
        f"ep{ei:03d}_demo{DEMO_IDS[ei]}": summarize([r for r in ho_recs if r["ep"] == ei])
        for ei in HELDOUT_EPS}
    results["train_per_episode"] = {
        f"ep{ei:03d}_demo{DEMO_IDS[ei]}": summarize([r for r in tr_recs if r["ep"] == ei])
        for ei in TRAIN_EPS}

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(json.dumps({k: results[k] for k in
                      ["backbone", "heldout", "train", "heldout_per_episode"]}, indent=1))
    overlay_grid(ho_recs, args.grid)


if __name__ == "__main__":
    main()

"""Evaluate v0.5 multi-task grounding: per-task + per-category tables on
(a) held-out episodes of training tasks, (b) held-out tasks; overlay grids.

Multi-instance scoring (documented): the model points at ONE location per
(frame, category) query; pixel error = min over visible GT instances of the
queried category ("point at any instance is correct"). Depth error = |z_pred -
z_gt| of the pixel-nearest visible instance. within_30px on the min-distance.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mt_common import CACHE, CX, CY, FX, FY, IMG, W, heldout_ep_file, train_tasks
from mt_dataset import MTGroundingDataset
from model_mt import GroundingModelMT


class EvalQueries(Dataset):
    def __init__(self, ds, pairs):
        self.ds = ds
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        idx, cat = self.pairs[i]
        return self.ds.get(idx, cat)


def build_pairs(ds, max_per_task=3000, seed=0):
    by_task = defaultdict(list)
    for idx in range(len(ds)):
        for cat in ds.frame_cats[idx]:
            by_task[ds.item_task[idx]].append((idx, int(cat)))
    pairs = []
    for t, ps in sorted(by_task.items()):
        if len(ps) > max_per_task:
            sel = np.linspace(0, len(ps) - 1, max_per_task).astype(int)
            ps = [ps[i] for i in sel]
        pairs += ps
    return pairs


@torch.no_grad()
def run(model, ds, pairs, bs=64, workers=12, dev="cuda"):
    dl = DataLoader(EvalQueries(ds, pairs), batch_size=bs, num_workers=workers,
                    pin_memory=True)
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
        z_pred = model.sample_depth(dmap, uv_pred).cpu().numpy()
        up = uv_pred.cpu().numpy()
        us = uv_soft.cpu().numpy()
        uvs = batch["uvs"].numpy()
        zs = batch["zs"].numpy()
        mm = batch["mode_mask"].numpy()
        for i in range(len(up)):
            gt = uvs[i][mm[i]]
            gz = zs[i][mm[i]]
            d = np.linalg.norm(gt - up[i], axis=1)
            j = int(d.argmin())
            ds_ = np.linalg.norm(gt - us[i], axis=1)
            ei = int(batch["ep"][i])
            task, fi, _ = ds.eps[ei]
            recs.append({
                "task": task, "file": fi, "ep": ei,
                "frame": int(batch["frame"][i]), "idx": int(batch["idx"][i]),
                "cat": int(batch["cat"][i]), "n_modes": int(mm[i].sum()),
                "px": float(d[j]), "px_soft": float(ds_.min()),
                "dz": float(abs(z_pred[i] - gz[j])),
                "z_gt": float(gz[j]), "z_pred": float(z_pred[i]),
                "uv_pred": up[i].tolist(), "uv_gt": gt[j].tolist(),
                "gt_modes": gt.tolist()})
    return recs


def summarize(recs):
    if not recs:
        return {"n": 0}
    px = np.array([r["px"] for r in recs])
    pxs = np.array([r["px_soft"] for r in recs])
    dz = np.array([r["dz"] for r in recs])
    return {"n": len(recs),
            "px_median": round(float(np.median(px)), 2),
            "px_p90": round(float(np.percentile(px, 90)), 2),
            "within_30px_pct": round(float((px < 30).mean() * 100), 1),
            "px_soft_median": round(float(np.median(pxs)), 2),
            "depth_mae_m": round(float(dz.mean()), 4),
            "depth_median_m": round(float(np.median(dz)), 4)}


def per_key(recs, key):
    groups = defaultdict(list)
    for r in recs:
        groups[r[key]].append(r)
    return {k: summarize(v) for k, v in sorted(groups.items())}


def overlay_grid(ds, recs, task, out_png, n=8, vocab=None):
    """8-frame grid for one task: red cross = pred, green = visible GT modes of
    the queried category, orange = in-frame-but-occluded instances."""
    from PIL import Image, ImageDraw
    rs = sorted([r for r in recs if r["task"] == task],
                key=lambda r: (r["file"], r["frame"]))
    if not rs:
        print(f"no recs for {task}")
        return
    idxs = sorted(set(np.linspace(0, len(rs) - 1, n).astype(int)))
    picks = [rs[i] for i in idxs]
    s = IMG / W
    tiles = []
    for r in picks:
        ei, i = ds.items[r["idx"]]
        _, fi, d = ds.eps[ei]
        lab = ds.lab[ei]
        im = Image.open(os.path.join(d, "frames", f"f_{i:05d}.jpg")).convert("RGB")
        dr = ImageDraw.Draw(im)
        occ = (lab["obj_cats"] == r["cat"]) & lab["inframe"][i] & ~lab["vis"][i]
        for uv in lab["uvz"][i][occ][:, :2]:
            gu, gv = uv[0] * s, uv[1] * s
            dr.ellipse([gu - 7, gv - 7, gu + 7, gv + 7], outline=(255, 160, 0), width=2)
        for uv in r["gt_modes"]:
            gu, gv = uv[0] * s, uv[1] * s
            dr.ellipse([gu - 9, gv - 9, gu + 9, gv + 9], outline=(0, 255, 0), width=3)
        pu, pv = r["uv_pred"][0] * s, r["uv_pred"][1] * s
        dr.line([pu - 10, pv, pu + 10, pv], fill=(255, 40, 40), width=3)
        dr.line([pu, pv - 10, pu, pv + 10], fill=(255, 40, 40), width=3)
        cname = vocab[r["cat"]] if vocab else str(r["cat"])
        dr.rectangle([0, 0, im.width, 22], fill=(0, 0, 0))
        dr.text((4, 4), f"{task} ep{r['file']} f{r['frame']} q={cname} "
                        f"px={r['px']:.0f} dz={r['dz']*100:.1f}cm k={r['n_modes']}",
                fill=(255, 255, 0))
        tiles.append(im)
    cols, rows = 4, (len(tiles) + 3) // 4
    grid = Image.new("RGB", (cols * IMG, rows * IMG), (15, 15, 15))
    for i, im in enumerate(tiles):
        grid.paste(im, ((i % cols) * IMG, (i // cols) * IMG))
    grid.save(out_png)
    print(f"wrote {out_png}", flush=True)


def trained_categories():
    """Categories visible at least once in TRAIN episodes of train tasks."""
    meta = json.load(open(os.path.join(CACHE, "meta.json")))
    seen = set()
    for t in train_tasks():
        table = meta["episodes"][t]
        ho = heldout_ep_file(table)
        for r in table:
            if r["file"] == ho:
                continue
            d = os.path.join(CACHE, t, f"ep{r['file']:03d}")
            lab = np.load(os.path.join(d, "lab.npz"))
            from mt_common import instance_thresholds, visible_mask
            margin = lab["margin"].astype(np.float32)
            thr = instance_thresholds(margin, lab["inframe"],
                                      np.load(os.path.join(d, "disp.npy")))
            vis = visible_mask(lab["inframe"], margin, thr)
            vis &= (lab["obj_cats"] >= 0)[None, :]
            seen.update(int(c) for c in np.unique(lab["obj_cats"][vis.any(0)]))
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grids-dir", default=None)
    ap.add_argument("--grid-heldout-eps", nargs="*",
                    default=["turning_on_radio", "sorting_vegetables",
                             "hiding_Easter_eggs"])
    ap.add_argument("--grid-heldout-tasks", nargs="*",
                    default=["putting_dishes_away_after_cleaning"])
    ap.add_argument("--max-per-task", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=64)
    args = ap.parse_args()

    dev = "cuda"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    vocab = ck["vocab"]
    hf_token = open("/root/.hf_token").read().strip() if os.path.exists("/root/.hf_token") else None
    model = GroundingModelMT(n_categories=ck["config"]["n_categories"],
                             backbone=ck["config"]["backbone"],
                             hf_token=hf_token).to(dev)
    missing, unexpected = model.load_state_dict(ck["head"], strict=False)
    assert not unexpected and all(k.startswith("backbone.") for k in missing)
    model.eval()

    meta = json.load(open(os.path.join(CACHE, "meta.json")))
    holdout = ck["heldout_ep"]

    # ---- (a) held-out episodes of training tasks
    eps_a = [(t, holdout[t]) for t in sorted(holdout)]
    ds_a = MTGroundingDataset(eps_a, with_negatives=False)
    pairs_a = build_pairs(ds_a, args.max_per_task)
    recs_a = run(model, ds_a, pairs_a, bs=args.bs)
    print("held-out episodes done:", len(recs_a), flush=True)

    # ---- (b) held-out tasks (all 5 episodes)
    eps_b = []
    for t in ck["heldout_tasks"]:
        eps_b += [(t, r["file"]) for r in meta["episodes"][t]]
    ds_b = MTGroundingDataset(eps_b, with_negatives=False)
    pairs_b = build_pairs(ds_b, args.max_per_task)
    recs_b = run(model, ds_b, pairs_b, bs=args.bs)
    print("held-out tasks done:", len(recs_b), flush=True)

    seen = trained_categories()
    for r in recs_b:
        r["cat_seen_in_train"] = r["cat"] in seen

    def cat_table(recs):
        return {vocab[c]: v for c, v in per_key(recs, "cat").items()}

    results = {
        "ckpt": args.ckpt, "backbone": ck["config"]["backbone"],
        "trainable_params": ck.get("trainable_params"),
        "train_minutes": round(ck.get("train_minutes", -1), 1),
        "step": ck.get("step"), "args": ck.get("args"),
        "heldout_tasks": ck["heldout_tasks"],
        "heldout_ep_overall": summarize(recs_a),
        "heldout_ep_per_task": per_key(recs_a, "task"),
        "heldout_task_overall": summarize(recs_b),
        "heldout_task_per_task": per_key(recs_b, "task"),
        "heldout_task_seen_cats": summarize([r for r in recs_b if r["cat_seen_in_train"]]),
        "heldout_task_unseen_cats": summarize([r for r in recs_b if not r["cat_seen_in_train"]]),
        "heldout_ep_per_category": cat_table(recs_a),
        "heldout_task_per_category": cat_table(recs_b),
        "unseen_category_names": sorted({vocab[r["cat"]] for r in recs_b
                                         if not r["cat_seen_in_train"]}),
    }
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)

    print(json.dumps({k: results[k] for k in
                      ["backbone", "heldout_ep_overall", "heldout_task_overall",
                       "heldout_task_seen_cats", "heldout_task_unseen_cats"]}, indent=1))
    print("%-42s %6s %8s %8s %8s %9s" % ("task", "n", "px_med", "px_p90", "w30%", "dMAE_m"))
    for t, v in results["heldout_ep_per_task"].items():
        print("%-42s %6d %8.1f %8.1f %8.1f %9.3f" % (t, v["n"], v["px_median"],
              v["px_p90"], v["within_30px_pct"], v["depth_mae_m"]))
    print("---- held-out TASKS ----")
    for t, v in results["heldout_task_per_task"].items():
        print("%-42s %6d %8.1f %8.1f %8.1f %9.3f" % (t, v["n"], v["px_median"],
              v["px_p90"], v["within_30px_pct"], v["depth_mae_m"]))

    if args.grids_dir:
        os.makedirs(args.grids_dir, exist_ok=True)
        tag = ck["config"]["backbone"].split("_")[0]
        for t in args.grid_heldout_eps:
            overlay_grid(ds_a, recs_a, t,
                         os.path.join(args.grids_dir, f"grid_{tag}_heldoutep_{t}.png"),
                         vocab=vocab)
        for t in args.grid_heldout_tasks:
            overlay_grid(ds_b, recs_b, t,
                         os.path.join(args.grids_dir, f"grid_{tag}_heldouttask_{t}.png"),
                         vocab=vocab)


if __name__ == "__main__":
    main()

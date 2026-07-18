"""Offline stage-head metrics (spec: per-arm stage accuracy, boundary timing
error in frames, ledger F1; held-out episodes AND held-out tasks, per family).

Boundary timing: gt transitions = frames where the label seg id changes;
predicted transitions = changes of the median-filtered argmax stage track.
Each gt transition matches the nearest predicted transition of the same
(from, to) direction within +-TOL frames; report median |dt| and recall.
"""

import argparse
import json
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import FPS_CACHE
from dataset import StageWindowDataset
from model import StageHead
from train import forward

TOL = int(10 * FPS_CACHE)   # 10 s matching window


def transitions(track):
    return [(i, track[i - 1], track[i]) for i in range(1, len(track))
            if track[i] != track[i - 1]]


def median_filter(x, k=5):
    from scipy.ndimage import median_filter as mf
    return mf(x, size=k, mode="nearest")


def boundary_errors(gt_track, pred_track):
    errs, missed = [], 0
    pt = transitions(pred_track)
    for i, a, b in transitions(gt_track):
        cands = [abs(j - i) for j, pa, pb in pt if pb == b and abs(j - i) <= TOL]
        if cands:
            errs.append(min(cands))
        else:
            missed += 1
    return errs, missed


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True, help="JSON list [[task, fi], ...]")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="stage_eval.json")
    ap.add_argument("--bs", type=int, default=64)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = StageHead().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    backbone.eval().requires_grad_(False).to(device)

    eps = [tuple(e) for e in json.load(open(args.episodes))]
    per_task = defaultdict(lambda: dict(n=0, correct=0, bnd=[], missed=0,
                                        n_gt_bnd=0, led_tp=0, led_fp=0, led_fn=0))
    for task, fi in eps:
        ds = StageWindowDataset([(task, fi)], stride=1)
        dl = DataLoader(ds, batch_size=args.bs, num_workers=4)
        preds, gts = [], []
        led_p, led_g, led_m = [], [], []
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=device == "cuda"):
                out = forward(model, backbone, batch, device)
            preds.append(out["stage_logits"].argmax(-1).cpu().numpy())
            gts.append(batch["stage_soft"].argmax(-1).cpu().numpy())
            led_p.append((torch.sigmoid(out["ledger_logits"]) > 0.5).cpu().numpy())
            led_g.append(batch["ledger"].cpu().numpy() > 0.5)
            led_m.append((batch["ledger_valid"].view(-1, 1)
                          & batch["lit_mask"]).cpu().numpy())
        pred = np.concatenate(preds)          # [N,2]
        gt = np.concatenate(gts)
        r = per_task[task]
        r["n"] += gt.size
        r["correct"] += int((pred == gt).sum())
        for a in range(2):
            errs, missed = boundary_errors(gt[:, a], median_filter(pred[:, a]))
            r["bnd"] += errs
            r["missed"] += missed
            r["n_gt_bnd"] += len(errs) + missed
        lp, lg, lm = (np.concatenate(x) for x in (led_p, led_g, led_m))
        r["led_tp"] += int((lp & lg & lm).sum())
        r["led_fp"] += int((lp & ~lg & lm).sum())
        r["led_fn"] += int((~lp & lg & lm).sum())

    table = {}
    for task, r in sorted(per_task.items()):
        prec = r["led_tp"] / max(1, r["led_tp"] + r["led_fp"])
        rec = r["led_tp"] / max(1, r["led_tp"] + r["led_fn"])
        table[task] = dict(
            stage_acc=round(r["correct"] / max(1, r["n"]), 4),
            boundary_median_frames=(float(np.median(r["bnd"])) if r["bnd"] else None),
            boundary_recall=round(1 - r["missed"] / max(1, r["n_gt_bnd"]), 4),
            ledger_f1=round(2 * prec * rec / max(1e-9, prec + rec), 4))
        print(f"{task:<44} acc={table[task]['stage_acc']:.3f} "
              f"bnd={table[task]['boundary_median_frames']} "
              f"rec={table[task]['boundary_recall']:.3f} "
              f"ledF1={table[task]['ledger_f1']:.3f}")
    json.dump(table, open(args.out, "w"), indent=1)
    accs = [t["stage_acc"] for t in table.values()]
    print(f"\nmean stage-acc {np.mean(accs):.4f} over {len(table)} tasks -> {args.out}")


if __name__ == "__main__":
    main()

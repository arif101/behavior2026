"""Offline stage-head metrics -- the spec v2 section 6 gate:
  1. per-arm stage accuracy + boundary timing error (frames)
  2. ledger F1 (latched p_sat vs privileged truth)
  3. progress MAE + monotonicity-violation rate on nominal segments
  5. calibration: entropy at boundaries vs mid-segment
on held-out episodes AND held-out tasks, per task family (pass each split
separately). (4, the kill ablation vs temporal-position bins, is a separate
training run over relabeled caches.)

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
from dataset import BLEND_F, StageWindowDataset
from model import StageHead
from train import forward, load_backbone

TOL = int(10 * FPS_CACHE)   # 10 s matching window
MONO_EPS = 0.05             # progress drop that counts as a violation


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
    ap.add_argument("--backbone", default="dinov2_vitb14")
    ap.add_argument("--label_file", default="stage_labels.npz",
                    help="stage_labels_posbins.npz = score vs bin labels (v4)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    backbone = load_backbone(args.backbone, device)
    spec = backbone[1]
    model = StageHead(grid=spec["grid"], feat=spec["feat"]).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    eps = [tuple(e) for e in json.load(open(args.episodes))]
    per_task = defaultdict(lambda: dict(
        n=0, correct=0, bnd=[], missed=0, n_gt_bnd=0,
        led_tp=0, led_fp=0, led_fn=0, prog_err=[], mono_viol=0, mono_n=0,
        ent_bnd=[], ent_mid=[]))
    for task, fi in eps:
        ds = StageWindowDataset([(task, fi)], stride=1,
                                label_file=args.label_file)
        dl = DataLoader(ds, batch_size=args.bs, num_workers=4)
        preds, gts, progs, ents = [], [], [], []
        led_p, led_g, led_m = [], [], []
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=device == "cuda"):
                out = forward(model, backbone, batch, device)
            preds.append(out["stage_logits"].argmax(-1).cpu().numpy())
            gts.append(batch["stage_soft"].argmax(-1).cpu().numpy())
            sl = out["progress"].float().cpu().numpy()        # [B,2,2] sincos
            progs.append((np.arctan2(sl[..., 0], sl[..., 1])
                          / (2 * np.pi)) % 1.0)
            ents.append(StageHead.entropy(
                out["stage_logits"].float()).cpu().numpy())
            led_p.append((torch.sigmoid(out["ledger_logits"]) > 0.5).cpu().numpy())
            led_g.append(batch["ledger"].cpu().numpy() > 0.5)
            led_m.append((batch["ledger_valid"].view(-1, 1)
                          & batch["lit_mask"]
                          & batch["ledger_lit_valid"]).cpu().numpy())
        pred = np.concatenate(preds)          # [N,2]
        gt = np.concatenate(gts)
        prog = np.concatenate(progs)          # [N,2] in [0,1)
        ent = np.concatenate(ents)            # [N,2]
        r = per_task[task]
        r["n"] += gt.size
        r["correct"] += int((pred == gt).sum())
        lab = ds.eps[0]["lab"]
        for a, arm in enumerate(("left", "right")):
            errs, missed = boundary_errors(gt[:, a], median_filter(pred[:, a]))
            r["bnd"] += errs
            r["missed"] += missed
            r["n_gt_bnd"] += len(errs) + missed
            # progress MAE + monotonicity on nominal (labeled) segments
            seg = lab[f"seg_{arm}"]
            gp = lab[f"progress_{arm}"]
            act = seg >= 0
            if act.any():
                d = np.abs(prog[act, a] - gp[act])
                r["prog_err"] += np.minimum(d, 1 - d).tolist()   # sincos wrap
            same = act[1:] & act[:-1] & (seg[1:] == seg[:-1])
            r["mono_viol"] += int(((prog[1:, a] - prog[:-1, a]) < -MONO_EPS)[same].sum())
            r["mono_n"] += int(same.sum())
            # calibration: entropy near gt boundaries vs mid-segment
            bd = np.flatnonzero(np.diff(seg) != 0) + 1
            near = np.zeros(len(seg), dtype=bool)
            for b in bd:
                near[max(0, b - BLEND_F):b + BLEND_F] = True
            r["ent_bnd"] += ent[near, a].tolist()
            r["ent_mid"] += ent[act & ~near, a].tolist()
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
            ledger_f1=round(2 * prec * rec / max(1e-9, prec + rec), 4),
            progress_mae=(round(float(np.mean(r["prog_err"])), 4)
                          if r["prog_err"] else None),
            monotonicity_violation_rate=round(
                r["mono_viol"] / max(1, r["mono_n"]), 4),
            entropy_boundary=(round(float(np.mean(r["ent_bnd"])), 4)
                              if r["ent_bnd"] else None),
            entropy_midsegment=(round(float(np.mean(r["ent_mid"])), 4)
                                if r["ent_mid"] else None))
        t = table[task]
        print(f"{task:<40} acc={t['stage_acc']:.3f} "
              f"bnd={t['boundary_median_frames']} "
              f"rec={t['boundary_recall']:.3f} ledF1={t['ledger_f1']:.3f} "
              f"progMAE={t['progress_mae']} mono={t['monotonicity_violation_rate']} "
              f"H(bnd/mid)={t['entropy_boundary']}/{t['entropy_midsegment']}")
    json.dump(table, open(args.out, "w"), indent=1)
    accs = [t["stage_acc"] for t in table.values()]
    print(f"\nmean stage-acc {np.mean(accs):.4f} over {len(table)} tasks -> {args.out}")


if __name__ == "__main__":
    main()

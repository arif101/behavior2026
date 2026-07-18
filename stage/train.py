"""Train the stage head (Phase 2). bf16, frozen DINOv2 backbone on the fly for
current frames (one pass -- at serve this pass is shared with grounding).

Losses (masks make every label source optional -- official-only caches train):
  stage      soft cross-entropy vs boundary-blended targets, task-masked logits
  phase      CE, ignore -1 (needs Phase-1 extractor labels)
  progress   MSE on sincos, only where a segment is active
  active_lit CE, ignore -1
  ledger     BCE, only where ledger_valid

Also writes stage_medians.json (median stage durations in seconds per task,
from the training labels) -- serve.py's stage_age_ratio denominator.
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from common import FPS_CACHE, T_HIST
from dataset import StageWindowDataset
from model import StageHead

LOSS_W = dict(stage=1.0, phase=0.5, progress=0.5, lit=0.5, ledger=1.0)


def soft_ce(logits, soft_targets):
    return -(soft_targets * torch.log_softmax(logits, -1)).sum(-1).mean()


def compute_losses(out, batch):
    losses = {}
    losses["stage"] = soft_ce(out["stage_logits"], batch["stage_soft"])

    ph = batch["phase"]
    if (ph >= 0).any():
        losses["phase"] = F.cross_entropy(
            out["phase_logits"].flatten(0, 1), ph.flatten(), ignore_index=-1)

    pv = batch["progress_valid"].unsqueeze(-1)
    if pv.any():
        losses["progress"] = (F.mse_loss(out["progress"], batch["progress"],
                                         reduction="none") * pv).sum() / pv.sum().clamp(min=1) / 2

    al = batch["active_lit"]
    if (al >= 0).any():
        losses["lit"] = F.cross_entropy(
            out["lit_logits"].flatten(0, 1), al.flatten(), ignore_index=-1)

    lv = batch["ledger_valid"].view(-1, 1) & batch["lit_mask"]
    if lv.any():
        losses["ledger"] = (F.binary_cross_entropy_with_logits(
            out["ledger_logits"], batch["ledger"], reduction="none") * lv
        ).sum() / lv.sum()

    total = sum(LOSS_W[k] * v for k, v in losses.items())
    return total, {k: v.item() for k, v in losses.items()}


def stage_medians(ds):
    """Median official-segment duration (s) per (task, stage) from labels."""
    durs = defaultdict(list)
    for ep in ds.eps:
        for arm in ("left", "right"):
            seg = ep["lab"][f"seg_{arm}"]
            stg = ep["lab"][f"stage_{arm}"]
            i = 0
            while i < len(seg):
                if seg[i] >= 0:
                    j = i
                    while j + 1 < len(seg) and seg[j + 1] == seg[i]:
                        j += 1
                    durs[(ep["task"], int(stg[i]))].append((j - i + 1) / FPS_CACHE)
                    i = j + 1
                else:
                    i += 1
    return {f"{t}|{s}": float(np.median(v)) for (t, s), v in durs.items()}


@torch.no_grad()
def evaluate(model, backbone, loader, device):
    model.eval()
    n = correct = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = forward(model, backbone, batch, device)
        pred = out["stage_logits"].argmax(-1)
        gt = batch["stage_soft"].argmax(-1)
        correct += (pred == gt).sum().item()
        n += gt.numel()
    model.train()
    return correct / max(1, n)


def forward(model, backbone, batch, device):
    rgb = batch["rgb"].permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    with torch.no_grad():
        tok = backbone.forward_features((rgb - mean) / std)["x_norm_patchtokens"]
    return model(tok, batch["depth"], batch["hist_glob"], batch["hist_prop"],
                 batch["lit_pred"], batch["lit_tgt"], batch["lit_ref"],
                 batch["lit_mask"], batch["stage_mask"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True,
                    help="JSON: {train: [[task, fi], ...], val: [...]}")
    ap.add_argument("--out", default="/root/stage_ckpt")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--t_hist", type=int, default=T_HIST)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val_every", type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    splits = json.load(open(args.episodes))
    tr_ds = StageWindowDataset([tuple(e) for e in splits["train"]], args.t_hist)
    va_ds = StageWindowDataset([tuple(e) for e in splits["val"]], args.t_hist)
    print(f"train items {len(tr_ds)}  val items {len(va_ds)}")

    json.dump(stage_medians(tr_ds), open(os.path.join(args.out, "stage_medians.json"), "w"))

    sampler = WeightedRandomSampler(tr_ds.sampler_weights(), len(tr_ds), replacement=True)
    tr = DataLoader(tr_ds, batch_size=args.bs, sampler=sampler,
                    num_workers=args.workers, pin_memory=True, drop_last=True)
    va = DataLoader(va_ds, batch_size=args.bs, num_workers=4)

    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    backbone.eval().requires_grad_(False).to(device)
    model = StageHead().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"stage head params: {n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)

    step = 0
    best = 0.0
    log = []
    while step < args.steps:
        for batch in tr:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=device == "cuda"):
                out = forward(model, backbone, batch, device)
                loss, parts = compute_losses(out, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                msg = " ".join(f"{k}={v:.3f}" for k, v in parts.items())
                print(f"[{step}/{args.steps}] loss={loss.item():.3f} {msg}", flush=True)
                log.append(dict(step=step, loss=loss.item(), **parts))
            if step % args.val_every == 0 or step == args.steps:
                acc = evaluate(model, backbone, va, device)
                print(f"[{step}] val stage-acc {acc:.4f}", flush=True)
                torch.save(model.state_dict(), os.path.join(args.out, "last.pt"))
                if acc > best:
                    best = acc
                    torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))
                json.dump(log, open(os.path.join(args.out, "train_log.json"), "w"))
            if step >= args.steps:
                break
    print(f"done; best val stage-acc {best:.4f} -> {args.out}/best.pt")


if __name__ == "__main__":
    main()

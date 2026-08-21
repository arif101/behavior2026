"""Train the M1 pilot grounding head. Frozen backbone, bf16 autocast, single GPU.

Loss = KL(gaussian GT heatmap || softmax(logits)) + lambda_d * L1(depth @ GT pixel).
All samples are in-frame by construction (dataset filters out-of-frame frames).
"""

import argparse
import json
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import GroundingDataset, TRAIN_EPS, HELDOUT_EPS, gaussian_target
from model import GroundingModel


def heatmap_loss(logits, target):
    logp = F.log_softmax(logits.flatten(1), dim=1)
    return -(target.flatten(1) * logp).sum(1).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--lambda-d", type=float, default=1.0)
    ap.add_argument("--sigma", type=float, default=2.5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="/root/grounding/ckpt_pilot.pt")
    args = ap.parse_args()

    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_ds = GroundingDataset(TRAIN_EPS)
    print(f"train episodes {TRAIN_EPS}: {len(train_ds)} in-frame samples", flush=True)
    dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                    num_workers=args.workers, pin_memory=True, drop_last=True,
                    persistent_workers=True)

    model = GroundingModel().to(dev)
    model.backbone.eval()
    n_param = sum(p.numel() for p in model.trainable_parameters())
    print(f"backbone: {model.backbone_name} (frozen); trainable head params: "
          f"{n_param/1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr,
                            weight_decay=args.wd)
    total_steps = args.epochs * len(dl)
    warmup = min(200, total_steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / max(warmup, 1), 1.0) * 0.5 *
        (1 + torch.cos(torch.tensor(min(s / total_steps, 1.0) * 3.14159)).item()))

    t0 = time.time()
    step = 0
    for ep in range(args.epochs):
        model.train()
        model.backbone.eval()
        agg = {"hm": 0.0, "d": 0.0, "n": 0}
        for batch in dl:
            rgb = batch["rgb"].to(dev, non_blocking=True)
            depth = batch["depth"].to(dev, non_blocking=True)
            uv = batch["uv"].to(dev, non_blocking=True)
            z = batch["z"].to(dev, non_blocking=True)
            cat = batch["cat"].to(dev, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, dmap = model(rgb, depth, cat)
            logits, dmap = logits.float(), dmap.float()
            target = gaussian_target(uv, sigma=args.sigma, device=dev)
            l_hm = heatmap_loss(logits, target)
            l_d = F.l1_loss(model.sample_depth(dmap, uv), z)
            loss = l_hm + args.lambda_d * l_d

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            agg["hm"] += l_hm.item()
            agg["d"] += l_d.item()
            agg["n"] += 1
            if step % 50 == 0:
                print(f"ep {ep} step {step}/{total_steps} "
                      f"hm {agg['hm']/agg['n']:.4f} depthL1 {agg['d']/agg['n']:.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)
                agg = {"hm": 0.0, "d": 0.0, "n": 0}

    train_min = (time.time() - t0) / 60
    torch.save({"head": model.head_state_dict(),
                "config": {"backbone": model.backbone_name, "n_categories": 1},
                "args": vars(args), "train_minutes": train_min,
                "trainable_params": n_param,
                "heldout_eps": HELDOUT_EPS}, args.out)
    print(f"saved {args.out}; train time {train_min:.1f} min", flush=True)


if __name__ == "__main__":
    main()

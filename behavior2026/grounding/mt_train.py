"""Train v0.5 multi-task grounding head. Frozen backbone, bf16 autocast, A40.

Loss = KL(mixture-gaussian GT || softmax(logits)) + lambda_d * L1(depth at each
visible GT mode). Sampler = uniform-per-task (see mt_dataset).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mt_common import CACHE, HELDOUT_TASKS, heldout_ep_file, train_tasks
from mt_dataset import (MTGroundingDataset, load_vocab, mixture_gaussian_target,
                        sample_depth_multi)
from model_mt import GroundingModelMT


def train_episode_list():
    meta = json.load(open(os.path.join(CACHE, "meta.json")))
    eps, holdout = [], {}
    for t in train_tasks():
        table = meta["episodes"][t]
        ho = heldout_ep_file(table)
        holdout[t] = ho
        eps += [(t, r["file"]) for r in table if r["file"] != ho]
    return eps, holdout


def heatmap_loss(logits, target):
    logp = F.log_softmax(logits.flatten(1), dim=1)
    return -(target.flatten(1) * logp).sum(1).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="dinov2_vitb14",
                    choices=["dinov2_vitb14", "dinov3_vitb16"])
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--lambda-d", type=float, default=1.0)
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-tasks", nargs="*", default=None,
                    help="restrict training to these tasks (single-task control)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    vocab, n_base = load_vocab()
    eps, holdout = train_episode_list()
    if args.only_tasks:
        eps = [(t, f) for t, f in eps if t in args.only_tasks]
        holdout = {t: h for t, h in holdout.items() if t in args.only_tasks}
    ds = MTGroundingDataset(eps, rng_seed=args.seed)
    w = ds.task_weights()
    n_tasks = len(set(ds.item_task))
    print(f"train: {len(eps)} episodes / {n_tasks} tasks / {len(ds)} frame-items; "
          f"vocab {len(vocab)} ({n_base} base)", flush=True)

    sampler = WeightedRandomSampler(torch.from_numpy(w), num_samples=args.steps * args.bs,
                                    replacement=True,
                                    generator=torch.Generator().manual_seed(args.seed))
    def winit(wid):
        info = torch.utils.data.get_worker_info()
        info.dataset.rng = np.random.RandomState(args.seed * 1000 + wid)

    dl = DataLoader(ds, batch_size=args.bs, sampler=sampler,
                    num_workers=args.workers, pin_memory=True, drop_last=True,
                    persistent_workers=True, prefetch_factor=4,
                    worker_init_fn=winit)

    hf_token = open("/root/.hf_token").read().strip() if os.path.exists("/root/.hf_token") else None
    model = GroundingModelMT(n_categories=len(vocab), backbone=args.backbone,
                             hf_token=hf_token).to(dev)
    model.backbone.eval()
    n_param = sum(p.numel() for p in model.trainable_parameters())
    print(f"backbone {args.backbone} (frozen); trainable {n_param/1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr,
                            weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / args.warmup, 1.0) * 0.5 *
        (1 + np.cos(min(s / args.steps, 1.0) * np.pi)))

    def save(path, step):
        torch.save({"head": model.head_state_dict(),
                    "config": {"backbone": args.backbone, "n_categories": len(vocab)},
                    "vocab": vocab, "n_base_vocab": n_base,
                    "heldout_tasks": HELDOUT_TASKS, "heldout_ep": holdout,
                    "args": vars(args), "step": step,
                    "trainable_params": n_param,
                    "train_minutes": (time.time() - t0) / 60}, path)

    t0 = time.time()
    model.train(); model.backbone.eval()
    agg = {"hm": 0.0, "d": 0.0, "n": 0}
    for step, batch in enumerate(dl, 1):
        rgb = batch["rgb"].to(dev, non_blocking=True)
        depth = batch["depth"].to(dev, non_blocking=True)
        cat = batch["cat"].to(dev, non_blocking=True)
        uvs = batch["uvs"].to(dev, non_blocking=True)
        zs = batch["zs"].to(dev, non_blocking=True)
        mm = batch["mode_mask"].to(dev, non_blocking=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, dmap = model(rgb, depth, cat)
        logits, dmap = logits.float(), dmap.float()
        target = mixture_gaussian_target(uvs, mm, sigma=args.sigma, device=dev)
        l_hm = heatmap_loss(logits, target)
        if mm.any():   # depth loss only at visible GT modes (negatives have none)
            l_d = F.l1_loss(sample_depth_multi(dmap, uvs, mm), zs[mm])
        else:
            l_d = torch.zeros((), device=dev)
        loss = l_hm + args.lambda_d * l_d

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
        opt.step()
        sched.step()
        agg["hm"] += l_hm.item(); agg["d"] += l_d.item(); agg["n"] += 1
        if step % 100 == 0:
            el = (time.time() - t0) / 60
            eta = el / step * (args.steps - step)
            print(f"step {step}/{args.steps} hm {agg['hm']/agg['n']:.4f} "
                  f"depthL1 {agg['d']/agg['n']:.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"({el:.1f} min, eta {eta:.0f})", flush=True)
            agg = {"hm": 0.0, "d": 0.0, "n": 0}
        if step % 5000 == 0:
            save(args.out + f".step{step}", step)

    save(args.out, args.steps)
    print(f"saved {args.out}; {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

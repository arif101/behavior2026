"""Affordance head v1: frozen DINOv2 ViT-S/14 + heatmap/offset/confidence head (PREP c).

THE LEGAL POINT SOURCE. Replaces oracle target points at eval: head RGB (-> 518x518) through
frozen DINOv2 -> 37x37 patch tokens -> tiny conv head ->
    heatmap  (37x37 logits, softmax = where the togglebutton is)
    offset   (2 per patch, sub-patch refinement)
    conf     (1 logit, "is the button actually visible" — gates HANDOFF; trained on the
              occlusion-checked visibility labels, so it must NOT fire when the button is
              hidden or out of frame)
3D at inference: soft-argmax pixel -> measured depth at pixel -> unproject (calibrated zed
intrinsics) -> base frame -> map.write_target().

Labels: 200-episode metalink projections (oracle-grade, validated 2.2 cm). Split BY EPISODE:
every 10th demo id held out (20 eps). Gates (spec): 3D error < 2 cm median on held-out VISIBLE
frames within 1.5 m (the SERVO/COMMIT range); report all ranges + conf ROC-AUC.

Run: OMP_NUM_THREADS=4 python train_affordance.py --epochs 3
"""

import argparse
import glob
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

FX, FY, CX, CY = 238.9, 315.8, 364.7, 356.2   # zed @ 720
IN = 518                                       # 37 * 14
P = 37
IMNET_M = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_S = np.array([0.229, 0.224, 0.225], np.float32)


class AffDataset(Dataset):
    def __init__(self, demos):
        self.items = []
        for d in demos:
            rows = np.load(f"/root/aff_data/ep{d}_labels.npz")["rows"]
            for r in rows:
                self.items.append((d, r))
        print(f"dataset: {len(self.items)} samples from {len(demos)} eps")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image
        d, r = self.items[i]
        t = int(r[0])
        img = Image.open(f"/root/aff_data/frames/ep{d}_f{t}.jpg").resize((IN, IN))
        x = np.asarray(img, np.float32) / 255.0
        x = (x - IMNET_M) / IMNET_S
        u, v, z, vis = r[1] * IN / 720.0, r[2] * IN / 720.0, r[3], r[4]
        return (torch.from_numpy(x.transpose(2, 0, 1)),
                torch.tensor([u, v, z, vis], dtype=torch.float32),
                torch.tensor([d, t], dtype=torch.int64))


class AffHead(nn.Module):
    def __init__(self, dim=384):
        super().__init__()
        self.trunk = nn.Sequential(nn.Conv2d(dim, 256, 1), nn.GELU(),
                                   nn.Conv2d(256, 256, 3, padding=1), nn.GELU())
        self.heat = nn.Conv2d(256, 1, 1)
        self.off = nn.Conv2d(256, 2, 1)
        self.conf = nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, feats):                  # (B, P*P, dim)
        B = feats.shape[0]
        f = feats.permute(0, 2, 1).reshape(B, -1, P, P)
        h = self.trunk(f)
        return (self.heat(h).reshape(B, -1),   # (B, P*P) logits
                self.off(h),                   # (B, 2, P, P) sub-patch offsets in [0,1] units
                self.conf(h.mean((2, 3))).squeeze(-1))


def losses(heat, off, conf, lab):
    u, v, vis = lab[:, 0], lab[:, 1], lab[:, 3]
    pu, pv = (u / 14).long().clamp(0, P - 1), (v / 14).long().clamp(0, P - 1)
    tgt = pv * P + pu
    m = vis > 0.5
    l_heat = F.cross_entropy(heat[m], tgt[m]) if m.any() else heat.sum() * 0
    if m.any():
        b = torch.arange(len(u), device=u.device)[m]
        po = off[b, :, pv[m], pu[m]]
        frac = torch.stack([(u[m] / 14 - pu[m].float()), (v[m] / 14 - pv[m].float())], 1)
        l_off = F.l1_loss(torch.sigmoid(po), frac)
    else:
        l_off = off.sum() * 0
    l_conf = F.binary_cross_entropy_with_logits(conf, vis)
    return l_heat, l_off, l_conf


@torch.no_grad()
def evaluate(dino, head, loader, dev, dump_dir=None):
    from PIL import Image
    head.eval()
    errs, rngs, confs, viss = [], [], [], []
    n_dump = 0
    for x, lab, ids in loader:
        x = x.to(dev, non_blocking=True)
        feats = dino.forward_features(x)["x_norm_patchtokens"]
        heat, off, conf = head(feats)
        prob = heat.softmax(-1)
        idx = prob.argmax(-1)
        pu, pv = (idx % P), (idx // P)
        b = torch.arange(len(idx), device=dev)
        frac = torch.sigmoid(off[b, :, pv, pu])
        u518 = (pu.float() + frac[:, 0]) * 14
        v518 = (pv.float() + frac[:, 1]) * 14
        u720, v720 = u518 * 720 / IN, v518 * 720 / IN
        confs.append(torch.sigmoid(conf).cpu().numpy())
        viss.append(lab[:, 3].numpy())
        for k in range(len(idx)):
            if lab[k, 3] < 0.5:
                continue
            d, t = int(ids[k, 0]), int(ids[k, 1])
            dep = np.asarray(Image.open(f"/root/aff_data/frames/ep{d}_f{t}_d.png"),
                             np.float32) / 1000.0
            uu, vv = float(u720[k]), float(v720[k])
            meas = dep[min(int(vv / 4), 179), min(int(uu / 4), 179)]
            if not np.isfinite(meas) or meas < 0.05:
                continue
            xc = (uu - CX) / FX * meas
            yc = (vv - CY) / FY * meas
            p_pred_cam = np.array([xc, -yc, -meas])
            zl = float(lab[k, 2])
            ul, vl = float(lab[k, 0]) * 720 / IN, float(lab[k, 1]) * 720 / IN
            p_lab_cam = np.array([(ul - CX) / FX * zl, -(vl - CY) / FY * zl, -zl])
            e = float(np.linalg.norm(p_pred_cam - p_lab_cam))
            errs.append(e)
            rngs.append(zl)
            if dump_dir and n_dump < 24:
                img = Image.open(f"/root/aff_data/frames/ep{d}_f{t}.jpg").convert("RGB")
                from PIL import ImageDraw
                dr = ImageDraw.Draw(img)
                dr.ellipse([ul - 8, vl - 8, ul + 8, vl + 8], outline=(0, 255, 60), width=3)
                dr.ellipse([uu - 8, vv - 8, uu + 8, vv + 8], outline=(255, 40, 40), width=3)
                dr.text((4, 4), f"ep{d} f{t} err={e * 100:.1f}cm", fill=(255, 255, 0))
                img.save(f"{dump_dir}/pred_ep{d}_f{t}.png")
                n_dump += 1
    errs, rngs = np.array(errs), np.array(rngs)
    confs, viss = np.concatenate(confs), np.concatenate(viss)
    # conf ROC-AUC (rank-based)
    order = np.argsort(confs)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(confs))
    pos, neg = viss > 0.5, viss <= 0.5
    auc = ((ranks[pos].mean() - (pos.sum() - 1) / 2) / max(neg.sum(), 1)) if pos.any() and neg.any() else float("nan")
    out = {"n_visible_evald": int(len(errs)), "conf_auc": round(float(auc), 3)}
    for name, m in (("all", np.ones_like(rngs, bool)), ("near_1p5m", rngs < 1.5),
                    ("far", rngs >= 1.5)):
        if m.any():
            out[name] = {"median_cm": round(float(np.median(errs[m]) * 100), 2),
                         "p90_cm": round(float(np.percentile(errs[m], 90) * 100), 2),
                         "n": int(m.sum())}
    head.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()
    dev = "cuda"

    demos = sorted(int(f.split("ep")[-1].split("_")[0])
                   for f in glob.glob("/root/aff_data/ep*_labels.npz"))
    held = set(demos[::10])
    train_d = [d for d in demos if d not in held]
    print(f"train {len(train_d)} eps / held-out {len(held)}: {sorted(held)[:6]}...")

    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(dev).eval()
    for p in dino.parameters():
        p.requires_grad = False
    head = AffHead().to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler()

    dl = DataLoader(AffDataset(train_d), batch_size=a.bs, shuffle=True, num_workers=6,
                    pin_memory=True, drop_last=True)
    dl_ev = DataLoader(AffDataset(sorted(held)), batch_size=a.bs, num_workers=6)

    step = 0
    for ep in range(a.epochs):
        for x, lab, _ in dl:
            x, lab = x.to(dev, non_blocking=True), lab.to(dev, non_blocking=True)
            with torch.autocast("cuda", torch.bfloat16):
                with torch.no_grad():
                    feats = dino.forward_features(x)["x_norm_patchtokens"]
                heat, off, conf = head(feats)
                lh, lo, lc = losses(heat, off, conf, lab)
                loss = lh + lo + 0.5 * lc
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if step % 100 == 0:
                print(f"ep{ep} step{step} heat={lh.item():.3f} off={lo.item():.3f} "
                      f"conf={lc.item():.3f}", flush=True)
            step += 1
        import os
        os.makedirs("/root/aff_out/preds", exist_ok=True)
        metrics = evaluate(dino, head, dl_ev, dev,
                           dump_dir="/root/aff_out/preds" if ep == a.epochs - 1 else None)
        print(f"EVAL epoch {ep}: {json.dumps(metrics)}", flush=True)
        torch.save(head.state_dict(), "/root/aff_out/aff_head.pt")
    with open("/root/aff_out/metrics.json", "w") as f:
        json.dump(metrics, f, indent=1)
    print("DONE -> /root/aff_out/")


if __name__ == "__main__":
    main()

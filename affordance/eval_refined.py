"""Eval-only decoding refinement for the trained affordance head — no retrain.

The v2 error tail (p90 8.6 cm ~= one patch at 1 m) is argmax quantization: ~10% of frames pick
a patch adjacent to the button's. Replace hard argmax with a probability-weighted centroid over
the 3x3 neighborhood around the argmax (pixel AND dz), which is how the head will be decoded at
serving. Reports hard vs soft side by side on the held-out split.
"""

import glob
import json

import numpy as np
import torch

import sys
sys.path.insert(0, "/root")
from train_affordance import (FX, FY, CX, CY, IN, P, AffDataset, AffHead)  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


@torch.no_grad()
def main():
    dev = "cuda"
    demos = sorted(int(f.split("ep")[-1].split("_")[0])
                   for f in glob.glob("/root/aff_data/ep*_labels.npz"))
    held = sorted(set(demos[::10]))
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(dev).eval()
    head = AffHead().to(dev)
    head.load_state_dict(torch.load("/root/aff_out/aff_head.pt", map_location=dev))
    head.eval()
    dl = DataLoader(AffDataset(held), batch_size=48, num_workers=6)

    from PIL import Image
    res = {"hard": [], "soft": []}
    rngs = []
    for x, lab, ids in dl:
        x = x.to(dev)
        feats = dino.forward_features(x)["x_norm_patchtokens"]
        heat, off, dz, conf = head(feats)
        prob2 = heat.softmax(-1).reshape(-1, P, P)
        idx = heat.argmax(-1)
        pu, pv = (idx % P), (idx // P)
        b = torch.arange(len(idx), device=dev)

        # hard: argmax patch + its offset + its dz
        frac = torch.sigmoid(off[b, :, pv, pu])
        u_h = (pu.float() + frac[:, 0]) * 14
        v_h = (pv.float() + frac[:, 1]) * 14
        dz_h = dz[b, 0, pv, pu]

        # soft: 3x3 neighborhood, prob-weighted centroid of per-patch refined pixels + dz
        us, vs, ds, ws = [], [], [], []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                qu = (pu + dx).clamp(0, P - 1)
                qv = (pv + dy).clamp(0, P - 1)
                w = prob2[b, qv, qu]
                fr = torch.sigmoid(off[b, :, qv, qu])
                us.append((qu.float() + fr[:, 0]) * 14 * w)
                vs.append((qv.float() + fr[:, 1]) * 14 * w)
                ds.append(dz[b, 0, qv, qu] * w)
                ws.append(w)
        W = torch.stack(ws).sum(0)
        u_s = torch.stack(us).sum(0) / W
        v_s = torch.stack(vs).sum(0) / W
        dz_s = torch.stack(ds).sum(0) / W

        for k in range(len(idx)):
            if lab[k, 3] < 0.5:
                continue
            d, t = int(ids[k, 0]), int(ids[k, 1])
            dep = np.asarray(Image.open(f"/root/aff_data/frames/ep{d}_f{t}_d.png"),
                             np.float32) / 1000.0
            zl = float(lab[k, 2])
            ul, vl = float(lab[k, 0]) * 720 / IN, float(lab[k, 1]) * 720 / IN
            p_lab = np.array([(ul - CX) / FX * zl, -(vl - CY) / FY * zl, -zl])
            for name, uu5, vv5, dzp in (("hard", u_h[k], v_h[k], dz_h[k]),
                                        ("soft", u_s[k], v_s[k], dz_s[k])):
                uu, vv = float(uu5) * 720 / IN, float(vv5) * 720 / IN
                meas = dep[min(int(vv / 4), 179), min(int(uu / 4), 179)]
                if not np.isfinite(meas) or meas < 0.05:
                    continue
                zp = meas + float(dzp)
                pp = np.array([(uu - CX) / FX * zp, -(vv - CY) / FY * zp, -zp])
                res[name].append(float(np.linalg.norm(pp - p_lab)))
            rngs.append(zl)

    out = {}
    for name, errs in res.items():
        e = np.array(errs)
        out[name] = {"median_cm": round(float(np.median(e) * 100), 2),
                     "p90_cm": round(float(np.percentile(e, 90) * 100), 2),
                     "frac_under_2cm": round(float((e < 0.02).mean()), 3),
                     "frac_under_5cm": round(float((e < 0.05).mean()), 3),
                     "n": len(e)}
    print(json.dumps(out, indent=1))
    with open("/root/aff_out/metrics_refined.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()

"""Label-validity gate (M1 discipline, mechanical): per episode, the projected
GT of small TARGET-category instances must agree with measured depth at the
projected pixel. Median |margin| large on a whole episode = wrong demo->file
mapping / extrinsics bug. Reports per task + per episode + per category.

Gate (pre-registered): per episode, frac(|margin| < 0.15m) >= 0.60 over
in-frame target-category instance-frames (looser than the pilot's 5cm point
gate because object CENTERS sit up to half an extent behind the surface; the
0.15m/0.60 rule still separates mapping bugs, which are metres off, cleanly).
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mt_common import CACHE, TASK_TARGETS, all_tasks

tt = json.load(open(TASK_TARGETS))
vocab = json.load(open(os.path.join(CACHE, "vocab.json")))["vocab"]
meta = json.load(open(os.path.join(CACHE, "meta.json")))

fails = []
percat = {}
for t in all_tasks():
    tgt_ids = {i for i, c in enumerate(vocab) if c in set(tt[t]["targets"])}
    for r in meta["episodes"][t]:
        d = os.path.join(CACHE, t, f"ep{r['file']:03d}")
        lab = np.load(os.path.join(d, "lab.npz"))
        cats = lab["obj_cats"]
        is_tgt = np.isin(cats, list(tgt_ids))
        m = lab["margin"].astype(np.float32)
        infr = lab["inframe"]
        sel = infr & is_tgt[None, :] & np.isfinite(m)
        vals = np.abs(m[sel])
        if len(vals) == 0:
            print(f"{t}/ep{r['file']:03d} demo{r['demo']}: no target-cat in-frame frames")
            continue
        frac15 = float((vals < 0.15).mean())
        frac05 = float((vals < 0.05).mean())
        med = float(np.median(vals))
        ok = frac15 >= 0.60
        if not ok:
            fails.append((t, r["file"], r["demo"], med, frac15))
        print(f"{t}/ep{r['file']:03d} demo{r['demo']}: n={len(vals)} "
              f"med|m|={med:.3f} frac<5cm={frac05:.2f} frac<15cm={frac15:.2f} "
              f"{'OK' if ok else 'FAIL'}", flush=True)
        for j, c in enumerate(cats):
            if c < 0:
                continue
            s = infr[:, j] & np.isfinite(m[:, j])
            if s.sum() == 0:
                continue
            key = vocab[c]
            percat.setdefault(key, []).append(float(np.median(np.abs(m[s, j]))))

print("\n==== per-category median |margin| (median over episodes) ====")
for k in sorted(percat):
    v = np.array(percat[k])
    print(f"{k:35s} med={np.median(v):.3f} max={v.max():.3f} n_eps={len(v)}")

print(f"\nGATE: {len(fails)} episode failures")
for f in fails:
    print("  FAIL", f)

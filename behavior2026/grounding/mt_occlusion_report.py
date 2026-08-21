"""Occlusion-gate report: per task, the fraction of in-frame instance-frame
labels EXCLUDED by the final static/moving visibility gate (mt_common), the
fixed-0.15m gate for comparison, and the fraction of (frame,category) queries
that are all-occluded (-> empty-target negatives). Flags tasks >60% excluded.

Also prints the container verification: per-instance kept-fraction over time
(first vs last quarter) for two fridge/cabinet tasks -- MOVING objects placed
inside a closed container must be gated OUT late; STATIC containers must stay
IN (visible) in both their open and closed states.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mt_common import CACHE, MOVE_DISP, all_tasks, instance_thresholds, visible_mask

meta = json.load(open(os.path.join(CACHE, "meta.json")))
vocab = json.load(open(os.path.join(CACHE, "vocab.json")))["vocab"]

out = {}
for t in all_tasks():
    n_in = n_vis = n_vis_fixed = 0
    q_tot = q_neg = 0
    for r in meta["episodes"][t]:
        d = os.path.join(CACHE, t, "ep%03d" % r["file"])
        lab = np.load(os.path.join(d, "lab.npz"))
        margin = lab["margin"].astype(np.float32)
        infr = lab["inframe"] & (lab["obj_cats"] >= 0)[None, :]
        thr = instance_thresholds(margin, lab["inframe"],
                                  np.load(os.path.join(d, "disp.npy")))
        vis = visible_mask(lab["inframe"], margin, thr) & infr
        vis_fixed = lab["inframe"] & (~np.isfinite(margin) | (margin < 0.15)) & infr
        n_in += int(infr.sum())
        n_vis += int(vis.sum())
        n_vis_fixed += int(vis_fixed.sum())
        cats = lab["obj_cats"]
        for i in range(infr.shape[0]):
            pres = np.unique(cats[infr[i]])
            visc = set(np.unique(cats[vis[i]]))
            q_tot += len(pres)
            q_neg += sum(1 for c in pres if c not in visc)
    excl = 100.0 * (1 - n_vis / max(n_in, 1))
    excl_fixed = 100.0 * (1 - n_vis_fixed / max(n_in, 1))
    out[t] = {"inframe_labels": n_in,
              "excluded_pct": round(excl, 1),
              "excluded_pct_fixed15": round(excl_fixed, 1),
              "queries": q_tot,
              "all_occluded_query_pct": round(100.0 * q_neg / max(q_tot, 1), 1),
              "flag_over60": excl > 60.0}

print("%-46s %8s %8s %8s %7s" % ("task", "excl%", "excl15%", "negq%", ">60%"))
for t, v in out.items():
    print("%-46s %8.1f %8.1f %8.1f %7s" % (
        t, v["excluded_pct"], v["excluded_pct_fixed15"],
        v["all_occluded_query_pct"], "FLAG" if v["flag_over60"] else ""))

os.makedirs("/root/grounding_v05", exist_ok=True)
with open("/root/grounding_v05/occlusion_exclusion.json", "w") as f:
    json.dump(out, f, indent=1)
print("\nwrote /root/grounding_v05/occlusion_exclusion.json")

# ---- container verification: early vs late kept-fraction
for t in ["clearing_food_from_table_into_fridge", "storing_food"]:
    print(f"\n==== {t}: per-instance kept fraction, first vs last quarter ====")
    r = meta["episodes"][t][0]
    d = os.path.join(CACHE, t, "ep%03d" % r["file"])
    lab = np.load(os.path.join(d, "lab.npz"))
    margin = lab["margin"].astype(np.float32)
    disp = np.load(os.path.join(d, "disp.npy"))
    thr = instance_thresholds(margin, lab["inframe"], disp)
    vis = visible_mask(lab["inframe"], margin, thr)
    N = margin.shape[0]
    q1, q4 = slice(0, N // 4), slice(3 * N // 4, N)
    for j, name in enumerate(lab["obj_names"]):
        inf1, inf4 = lab["inframe"][q1, j], lab["inframe"][q4, j]
        if inf1.sum() < 5 and inf4.sum() < 5:
            continue
        k1 = vis[q1, j][inf1].mean() if inf1.sum() else float("nan")
        k4 = vis[q4, j][inf4].mean() if inf4.sum() else float("nan")
        kind = "MOVING" if disp[j] >= MOVE_DISP else "static"
        print("  %-28s %6s disp=%.2f thr=%.2f kept_q1=%.2f kept_q4=%.2f" % (
            name, kind, disp[j], thr[j], k1, k4))

"""Probe harness stage 2: fit ridge probes per (layer, target) on extracted features.

Outputs probe_results.json: R^2 per layer per target (+ per-dim breakdown), ready for the
layer x concept heatmap visualization. CPU-only, runs anywhere.
"""

import argparse
import glob
import json
import numpy as np


def ridge_r2(X, Y, alpha=10.0, val_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    n = len(X)
    perm = rng.permutation(n)
    nv = int(n * val_frac)
    va, tr = perm[:nv], perm[nv:]
    Xm, Xs = X[tr].mean(0), X[tr].std(0) + 1e-6
    Ym, Ys = Y[tr].mean(0), Y[tr].std(0) + 1e-6
    Xtr, Xva = (X[tr] - Xm) / Xs, (X[va] - Xm) / Xs
    Ytr, Yva = (Y[tr] - Ym) / Ys, (Y[va] - Ym) / Ys
    d = Xtr.shape[1]
    A = Xtr.T @ Xtr + alpha * np.eye(d)
    W = np.linalg.solve(A, Xtr.T @ Ytr)
    pred = Xva @ W
    ss_res = ((Yva - pred) ** 2).sum(0)
    ss_tot = (Yva ** 2).sum(0) + 1e-9
    return 1.0 - ss_res / ss_tot  # per-dim R^2 (Y standardized)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/workspace/probe_data")
    ap.add_argument("--out", default="/workspace/probe_results.json")
    ap.add_argument("--max-feat-dim", type=int, default=4096, help="random-project features above this dim")
    args = ap.parse_args()

    lab = dict(np.load(f"{args.data}/labels.npz"))
    # phase_frac from (ep_idx, frame_idx): frame / max_frame_in_episode
    ep, fr = lab["ep_idx"].astype(int), lab["frame_idx"].astype(float)
    max_fr = {e: fr[ep == e].max() + 1.0 for e in np.unique(ep)}
    lab["phase_frac"] = np.array([fr[i] / max_fr[ep[i]] for i in range(len(ep))], dtype=np.float32)
    targets = {k: v.reshape(len(v), -1).astype(np.float32)
               for k, v in lab.items() if k not in ("ep_idx", "frame_idx")}

    results = {}
    for fp in sorted(glob.glob(f"{args.data}/feat_*.npz")):
        lname = fp.split("feat_")[-1][:-4]
        X = np.load(fp)["X"].astype(np.float32)
        if X.shape[1] > args.max_feat_dim:
            rng = np.random.default_rng(1)
            proj = rng.standard_normal((X.shape[1], args.max_feat_dim)).astype(np.float32)
            proj /= np.sqrt(X.shape[1])
            X = X @ proj
        results[lname] = {}
        for tname, Y in targets.items():
            r2 = ridge_r2(X, Y)
            results[lname][tname] = {"mean_r2": float(np.mean(r2)), "per_dim": [float(x) for x in r2]}
        print(lname, {t: round(results[lname][t]["mean_r2"], 3) for t in targets})

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print("PROBES_DONE ->", args.out)


if __name__ == "__main__":
    main()

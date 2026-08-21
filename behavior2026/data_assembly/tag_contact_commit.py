#!/usr/bin/env python3
"""COMMIT-INSTANT tagging — the definition the Phase-A treatment actually used.

This reconstructs the script that produced /root/phaseA/contact_commit.json, which was run
ad-hoc and never committed. That was a reproducibility gap: the file defines which frames get
upweighted 10.36x, i.e. the entire independent variable of the Phase-A A/B, and nothing in the
repo could regenerate it.

DEFINITION (deliberately narrower than tag_contact_fine.py):
  commit frames = sign changes on the binary gripper action dims (14=left, 22=right), +/- pad.
  NOTHING ELSE. tag_contact_fine.py takes the UNION of this with an "EE settled while gripper
  closed" signal, and that second signal fires through the whole CARRY phase -- 48.95% on its
  own, which swamps the thing we care about. Measured on the Phase-A demos:
      commit instants  3.12%   <- this file
      carry/settle    48.95%   <- the part tag_contact_fine.py adds
  Using the union as a sampling weight is useless (a ~50% class is already balanced).

Reproduces exactly: commit_frac 0.0312, weight_25pct 10.36, weight_40pct 20.73, pad 12.

Usage:
    python tag_contact_commit.py --data /root/valA/demos/data --out /root/valA/contact_commit.json
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import pyarrow.parquet as pq

GRIP_L, GRIP_R = 14, 22  # verified binary gripper action dims


def episode_commit_mask(A: np.ndarray, pad: int) -> np.ndarray:
    """Frames within +/-pad of a gripper open<->close transition."""
    n = len(A)
    m = np.zeros(n, bool)
    for g in (GRIP_L, GRIP_R):
        if g >= A.shape[1]:
            continue
        sign = np.sign(A[:, g])
        for i in np.flatnonzero(np.abs(np.diff(sign)) > 0):
            m[max(0, i - pad) : min(n, i + pad + 1)] = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/root/phaseA/demos/data")
    ap.add_argument("--pad", type=int, default=12)
    ap.add_argument("--out", default="/root/phaseA/contact_commit.json")
    ap.add_argument("--limit-files", type=int, default=0, help="0 = all; >0 samples N files (validation)")
    a = ap.parse_args()

    files = sorted(glob.glob(f"{a.data}/chunk-*/file-*.parquet"))
    if a.limit_files:
        files = files[: a.limit_files]
    out, tot_c, tot_n = {}, 0, 0
    for k, f in enumerate(files):
        t = pq.read_table(f, columns=["episode_index", "action"])
        ep = np.array(t.column("episode_index").to_pylist())
        for e in sorted(set(ep.tolist())):
            idx = np.flatnonzero(ep == e)
            A = np.stack([np.asarray(t.column("action")[int(i)].as_py(), float) for i in idx])
            m = episode_commit_mask(A, a.pad)
            rs, cur = [], None
            for i, v in enumerate(m):
                if v and cur is None:
                    cur = i
                elif not v and cur is not None:
                    rs.append((cur, i))
                    cur = None
            if cur is not None:
                rs.append((cur, len(m)))
            out[str(int(e))] = {
                "n_frames": int(len(m)),
                "commit_frames": int(m.sum()),
                "frac": round(float(m.mean()), 4),
                "ranges": rs,
            }
            tot_c += int(m.sum())
            tot_n += len(m)
        if (k + 1) % 20 == 0:
            print(f"  {k+1}/{len(files)} files, running commit frac {tot_c/max(1,tot_n):.4f}", flush=True)

    frac = tot_c / max(1, tot_n)

    def weight_for(target):
        return target * (1 - frac) / ((1 - target) * frac) if 0 < frac < 1 else 1.0

    meta = {
        "commit_frac": round(frac, 4),
        "pad": a.pad,
        "weight_25pct": round(weight_for(0.25), 2),
        "weight_40pct": round(weight_for(0.40), 2),
        "grip_dims": [GRIP_L, GRIP_R],
        "n_files": len(files),
    }
    json.dump({"meta": meta, "by_episode": out}, open(a.out, "w"))
    print(f"\nepisodes tagged: {len(out)}")
    print(f"COMMIT fraction: {frac*100:.2f}%  -> weight {meta['weight_25pct']}x for 25% of draws")
    print("wrote", a.out)


if __name__ == "__main__":
    main()

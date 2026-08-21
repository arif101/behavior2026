#!/usr/bin/env python3
"""Fine-grained contact tagging from the ACTUAL data (not annotation phases).

Why: annotation segments span whole manipulation PHASES (measured 55% of frames — already
balanced, useless as a weight). The frames that matter are the COMMIT INSTANTS: the gripper
open<->close transition and the moments the EE is on the object. Scaling episodes 6.7x does
NOT change the transit:contact ratio — re-weighting does.

Signals (union), all read from the fetched parquet:
  1. GRIPPER TRANSITION: sign change on the binary gripper action dims (VERIFIED: dim14=left,
     dim22=right; base(3)+torso(4)+armL(7)+gripL(1)+armR(7)+gripR(1) = 23), +/- a window.
  2. EE SETTLE: frames where an end-effector is nearly stationary while the gripper is closed
     (holding/manipulating), from observation.state eef positions ([17:20] L, [42:45] R).
Writes per-episode frame ranges + a sample_weight recommendation.

Usage: python tag_contact_fine.py --out contact_fine.json [--pad 12] [--target-frac 0.25]
"""
from __future__ import annotations
import argparse, glob, json
import numpy as np, pyarrow.parquet as pq

GRIP_L, GRIP_R = 14, 22          # verified binary gripper action dims
EEF_L, EEF_R = slice(17, 20), slice(42, 45)


def episode_contact_mask(A: np.ndarray, S: np.ndarray, pad: int) -> np.ndarray:
    n = len(A)
    m = np.zeros(n, bool)
    # 1) gripper transitions (the literal commit instants)
    for g in (GRIP_L, GRIP_R):
        if g >= A.shape[1]:
            continue
        sign = np.sign(A[:, g])
        tr = np.flatnonzero(np.abs(np.diff(sign)) > 0)
        for i in tr:
            m[max(0, i - pad): min(n, i + pad + 1)] = True
    # 2) EE settled while gripper closed (holding / fine manipulation)
    if S is not None and S.shape[1] >= 45:
        for sl, g in ((EEF_L, GRIP_L), (EEF_R, GRIP_R)):
            if g >= A.shape[1]:
                continue
            ee = S[:, sl]
            v = np.zeros(len(ee))
            v[1:] = np.linalg.norm(np.diff(ee, axis=0), axis=1)
            closed = np.sign(A[:len(ee), g]) < 0
            slow = v < 0.004
            m[: len(ee)] |= (closed & slow)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/root/phaseA/demos/data")
    ap.add_argument("--pad", type=int, default=12)
    ap.add_argument("--target-frac", type=float, default=0.25)
    ap.add_argument("--out", default="/root/phaseA/contact_fine.json")
    a = ap.parse_args()

    files = sorted(glob.glob(f"{a.data}/chunk-*/file-*.parquet"))
    out, tot_c, tot_n = {}, 0, 0
    for k, f in enumerate(files):
        t = pq.read_table(f, columns=["episode_index", "action", "observation.state"])
        ep = np.array(t.column("episode_index").to_pylist())
        for e in sorted(set(ep.tolist())):
            idx = np.flatnonzero(ep == e)
            A = np.stack([np.asarray(t.column("action")[int(i)].as_py(), float) for i in idx])
            S = np.stack([np.asarray(t.column("observation.state")[int(i)].as_py(), float) for i in idx])
            m = episode_contact_mask(A, S, a.pad)
            rs, cur = [], None
            for i, v in enumerate(m):
                if v and cur is None: cur = i
                elif not v and cur is not None: rs.append((cur, i)); cur = None
            if cur is not None: rs.append((cur, len(m)))
            out[str(int(e))] = {"n_frames": int(len(m)), "contact_frames": int(m.sum()),
                                "frac": round(float(m.mean()), 4), "ranges": rs}
            tot_c += int(m.sum()); tot_n += len(m)
        if (k + 1) % 20 == 0:
            print(f"  {k+1}/{len(files)} files, running frac {tot_c/max(1,tot_n):.3f}", flush=True)

    frac = tot_c / max(1, tot_n)
    w = a.target_frac * (1 - frac) / ((1 - a.target_frac) * frac) if 0 < frac < 1 else 1.0
    meta = {"contact_frac": round(frac, 4), "episodes": len(out),
            "sample_weight_contact": round(w, 2), "sample_weight_other": 1.0,
            "target_contact_frac": a.target_frac, "pad": a.pad,
            "grip_dims": [GRIP_L, GRIP_R]}
    json.dump({"meta": meta, "by_episode": out}, open(a.out, "w"))
    print(f"\nepisodes tagged: {len(out)}")
    print(f"FINE contact fraction: {frac*100:.2f}%   (annotation-phase version was 55%)")
    print(f"-> weight {w:.2f}x on contact frames to reach {a.target_frac*100:.0f}% of samples")
    print("wrote", a.out)


if __name__ == "__main__":
    main()

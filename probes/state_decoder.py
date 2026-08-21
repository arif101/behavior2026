"""CPU state-array object-pose decoder — positions without rendering.

Validated 2026-07-19 (picking_up_trash, 5 blind episodes incl. cross-instance):
cans median 1-4 mm / p95 < 10 mm; container ~26 mm (origin convention, systematic);
zero unresolved rows; NO per-episode calibration.

Structural rule (challenge rawdata HDF5, OmniGibson sparse state serialization):
  - each state row: [robot/base block | moving-object blocks, 15 dims each | 123-dim tail]
  - full keyframe rows (size == episode max, ~13/episode incl. row 0): ALL K objects,
    sorted-name registry order, blocks ending TAIL dims before row end
  - sparse rows: only objects in motion, same order; membership resolved by
    order-constrained min-cost continuity assignment against last-known positions,
    with refuse-on-ambiguity (no update) so errors cannot cascade past a keyframe

KNOWN OPEN (validate before trusting beyond rigid relocation tasks):
  - articulated objects (joints extend their blocks)
  - per-task TAIL variance (radio sparse rows suggest an extra entity / TAIL=124)

Usage (validation against replay labels):
  python state_decoder.py --hdf5 episode_00010020.hdf5 --objects can_of_soda_113 ... \
      --labels labels_10020.jsonl
Usage (decode only):
  python state_decoder.py --hdf5 ep.hdf5 --objects a b c --out poses.npz
"""
from __future__ import annotations

import argparse
import json

import h5py
import numpy as np

TAIL, STEP, BLOCK0 = 123, 15, 28


def assign(blocks, order, last, gate=0.06):
    """Order-preserving min-cost assignment of sparse blocks to objects (DP).
    Returns None when any block cannot be explained within the gate."""
    B, O = len(blocks), len(order)
    INF = 1e9
    cost = np.full((B, O), INF)
    for b in range(B):
        for o in range(O):
            if order[o] in last:
                d = np.linalg.norm(blocks[b] - last[order[o]])
                if d < gate:
                    cost[b, o] = d
    dp = np.full((B + 1, O + 1), INF)
    dp[0, :] = 0.0
    choice = np.zeros((B + 1, O + 1), dtype=int)
    for b in range(1, B + 1):
        for o in range(b, O + 1):
            skip = dp[b, o - 1]
            take = dp[b - 1, o - 1] + cost[b - 1, o - 1]
            dp[b, o], choice[b, o] = (take, 1) if take < skip else (skip, 0)
    if dp[B, O] >= INF:
        return None
    out, b, o = {}, B, O
    while b > 0:
        if choice[b, o]:
            out[order[o - 1]] = blocks[b - 1]
            b -= 1
        o -= 1
    return out


def decode(st: np.ndarray, ss: np.ndarray, objs: list[str]):
    """objs MUST be sorted by name (registry order). Returns ({obj: (N,3)}, n_unresolved)."""
    objs = sorted(objs)
    mx = int(ss.max())
    K = len(objs)
    full_start = mx - TAIL - STEP * K
    last: dict[str, np.ndarray] = {}
    out = {o: np.full((len(st), 3), np.nan) for o in objs}
    unresolved = 0
    for t in range(len(st)):
        size = int(ss[t])
        if size == mx:
            for k, o in enumerate(objs):
                last[o] = st[t, full_start + k * STEP: full_start + k * STEP + 3].copy()
        else:
            nb = max(0, size - TAIL - BLOCK0) // STEP
            if nb:
                blocks = [st[t, BLOCK0 + b * STEP: BLOCK0 + b * STEP + 3] for b in range(nb)]
                got = assign(blocks, objs, last)
                if got is None:
                    unresolved += 1
                else:
                    for o, v in got.items():
                        last[o] = v.copy()
        for o in objs:
            if o in last:
                out[o][t] = last[o]
    return out, unresolved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--objects", nargs="+", required=True)
    ap.add_argument("--labels", default=None, help="replay labels jsonl to validate against")
    ap.add_argument("--out", default=None, help="write decoded poses npz")
    args = ap.parse_args()

    f = h5py.File(args.hdf5, "r")
    g = f["data"][list(f["data"].keys())[0]]
    st = g["state"][:].astype(np.float64)
    ss = g["state_size"][:]
    dec, unres = decode(st, ss, args.objects)
    print(f"decoded {len(st)} frames, unresolved rows: {unres}")

    if args.labels:
        rows = [json.loads(l) for l in open(args.labels)]
        n = min(len(st), len(rows) // 2)
        for o in sorted(args.objects):
            lab = np.array([r["objs"][o] for r in rows[::2]])[:n]
            d = np.linalg.norm(dec[o][:n] - lab, axis=1)
            print(f"  {o}: median {np.nanmedian(d)*1000:.1f} mm  p95 {np.nanpercentile(d, 95)*1000:.1f} mm")
    if args.out:
        np.savez(args.out, **{o: v for o, v in dec.items()})
        print("wrote", args.out)


if __name__ == "__main__":
    main()

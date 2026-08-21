"""Segment an OmniGibson serialized `state` row into per-object blocks — WITHOUT a registry.

Why this exists
---------------
`probes/state_decoder.py` assumes every object block is STEP=15 wide. That is false. Read from
the OmniGibson v1.1.1 source:

    XFormPrim.serialize      -> [pos(3), ori(4)]                                    =  7
    EntityPrim.serialize     -> root_link(7) + joint_pos/vel/eff(3N)                =  7 + 3N
    StatefulObject.serialize -> super() + each *stateful* object state concatenated =  7 + 3N + S

so width is OBJECT-DEPENDENT (joints, and BEHAVIOR abilities like ToggledOn/Temperature). A radio
episode shows widths 14/15/16/17/20/24 in a single scene. The fixed-stride decoder silently
mis-parses everything after the first non-15 block; it only validated on `picking_up_trash`
because that scene happens to be uniform.

The handle
----------
`ori` is a UNIT QUATERNION at offset +3 of every block. That is a checksum: "could a block start
here?" is testable without knowing any object names. Local tests alone give false positives (a
quaternion with near-zero components makes i, i+1, i+2 all look valid), so we add two constraints:

  1. TEMPORAL  — the quaternion must be unit-norm at EVERY full-keyframe row, not just one.
  2. GLOBAL    — blocks must tile the region EXACTLY: no gaps, no overlaps.

Together these pin the segmentation. No object registry is required, which matters because
`behavior-1k/2026-challenge-task-instances` 404s even authenticated.

Caveat: the ~13 "full" rows are all at t=0..12 (an initialisation burst, not periodic snapshots),
so there is exactly ONE full world snapshot per episode. That is sufficient for our use: the
target object is static until first manipulated, and the phase we care about (navigation) precedes
manipulation.
"""

from __future__ import annotations

import argparse

import h5py
import numpy as np

QUAT_TOL = 2e-3       # |‖q‖ − 1| tolerance
POS_LIMIT = 50.0      # a scene coordinate beyond this is not a position
MIN_W, MAX_W = 7, 120  # 7 = rigid; R1Pro (~23 DOF) reaches 7 + 69 + states


def full_rows(path: str) -> np.ndarray:
    """Return the (K, D) stack of FULL keyframe rows (state_size == max)."""
    with h5py.File(path, "r") as f:
        g = f["data/demo_0"]
        st = g["state"][:]
        ss = g["state_size"][:]
    d = int(ss.max())
    return st[ss == d][:, :d]


def valid_starts(F: np.ndarray) -> np.ndarray:
    """Boolean mask: index i could begin a block, checked across ALL keyframes."""
    D = F.shape[1]
    ok = np.zeros(D, dtype=bool)
    for i in range(D - 7 + 1):
        q = F[:, i + 3 : i + 7]
        if not np.all(np.abs(np.linalg.norm(q, axis=1) - 1.0) < QUAT_TOL):
            continue
        p = F[:, i : i + 3]
        if np.any(np.abs(p) > POS_LIMIT):
            continue
        ok[i] = True
    return ok


def tile(ok: np.ndarray, start: int, end: int) -> list[int] | None:
    """Exact cover of [start, end) by blocks whose starts satisfy `ok`. Returns widths."""
    memo: dict[int, list[int] | None] = {}

    def go(i: int) -> list[int] | None:
        if i == end:
            return []
        if i > end or not ok[i]:
            return None
        if i in memo:
            return memo[i]
        memo[i] = None  # guard against cycles
        for w in range(MIN_W, MAX_W + 1):
            if i + w > end:
                break
            rest = go(i + w)
            if rest is not None:
                memo[i] = [w] + rest
                return memo[i]
        return memo[i]

    return go(start)


def solve(path: str, verbose: bool = True):
    F = full_rows(path)
    D = F.shape[1]
    ok = valid_starts(F)
    if verbose:
        print(f"keyframes {F.shape}  candidate starts {int(ok.sum())}/{D}")

    # The object region need not span the whole row: there may be a leading header and/or a
    # trailing section. Search outward from the most permissive hypothesis.
    for start in range(0, 64):
        if not ok[start]:
            continue
        widths = tile(ok, start, D)
        if widths:
            return start, D, widths, F
    return None, None, None, F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--max-show", type=int, default=40)
    a = ap.parse_args()

    start, end, widths, F = solve(a.hdf5)
    if widths is None:
        print("NO EXACT TILING FOUND")
        return

    import collections

    print(f"\nTILED [{start}, {end}) into {len(widths)} blocks")
    print("width histogram:", dict(sorted(collections.Counter(widths).items())))
    print(f"\n{'idx':>4} {'off':>5} {'w':>4}  position (t=0)                 drift over keyframes")
    off = start
    for k, w in enumerate(widths[: a.max_show]):
        p0 = F[0, off : off + 3]
        drift = float(np.abs(F[:, off : off + 3] - p0).max())
        print(f"{k:>4} {off:>5} {w:>4}  ({p0[0]:8.3f},{p0[1]:8.3f},{p0[2]:7.3f})  {drift:.5f}")
        off += w


if __name__ == "__main__":
    main()

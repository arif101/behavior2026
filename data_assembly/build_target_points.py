"""Turn replayed world poses into per-frame `target_points` for AdaLN conditioning.

Contract (must match scripts/b1k/add_target_points.py and B1KInputs):
    target_points       float32, flat list[6]  ->  [left_xyz, right_xyz]
    target_points_mask  bool,    list[2]       ->  False = that arm has no target this frame

The value is the EE-to-target DISPLACEMENT expressed in the BASE frame — NOT a world coordinate.
That matters twice over: a world point is unusable at eval (robot global pose is prohibited), and
displacement is the form that carried the +25-46 point gains in arXiv:2606.27663.

    p_obj_base = R(base_quat)^-1 . (p_obj_world - p_base_world)      <- replayed poses
    d_arm      = p_obj_base - p_ee_base                              <- eef_*_pos from proprio

Why this exists: coverage. The Phase-A dataset had real points on 2.67% of frames, so the channel
was masked on 36 of every 37 samples and the action-expert probe reads R2 0.369 for decoding the
point but 0.046 for decoding the future action — present, never converted to intent. This raises
coverage to ~100% on the episodes it processes.

FRAME ALIGNMENT IS THE HAZARD. Replay produced 1957 frames where the parquet had 1956; an
off-by-one silently shifts every label by one timestep across the whole dataset. We therefore
truncate to the common length and record the discrepancy per episode rather than assuming.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def quat_to_R(q: np.ndarray) -> np.ndarray:
    """Rotation matrix from an (x, y, z, w) quaternion."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses_dir", required=True, help="dir of ep*.json from replay_poses_batch")
    ap.add_argument("--target", required=True, help="scene object name to condition on, e.g. radio_89")
    ap.add_argument("--out", required=True, help="output npz of per-episode point arrays")
    ap.add_argument("--eef_left_idx", type=int, default=-1,
                    help="index of eef_left_pos within observation.state; -1 = emit base-frame "
                         "object position and let the converter subtract the EE")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.poses_dir, "ep*.json")))
    print(f"{len(files)} episodes in {a.poses_dir}")

    out: dict[str, np.ndarray] = {}
    n_ok = n_skip = 0
    mags: list[float] = []

    for f in files:
        d = json.load(open(f))
        F = d["frames"]
        if not F:
            n_skip += 1
            continue
        if a.target not in F[0].get("objects", {}):
            print(f"  SKIP ep{d['demo_id']}: {a.target!r} not tracked "
                  f"(have {list(F[0].get('objects', {}))})")
            n_skip += 1
            continue

        pts = np.zeros((len(F), 3), dtype=np.float32)
        msk = np.zeros(len(F), dtype=bool)
        for i, fr in enumerate(F):
            o = fr["objects"].get(a.target)
            if not o:                       # prim destroyed (sliced/diced) -> genuinely absent
                continue
            R = quat_to_R(np.asarray(fr["base_quat"], dtype=float))
            p = R.T @ (np.asarray(o["pos"], dtype=float) - np.asarray(fr["base_pos"], dtype=float))
            pts[i] = p
            msk[i] = True

        out[f"ep{d['demo_id']}_pts"] = pts
        out[f"ep{d['demo_id']}_mask"] = msk
        mags.extend(np.linalg.norm(pts[msk], axis=1).tolist())
        n_ok += 1

    np.savez_compressed(a.out, **out)
    m = np.asarray(mags)
    print(f"\nwrote {a.out}: {n_ok} episodes, {n_skip} skipped")
    print(f"coverage: {100.0 * (m.size / max(sum(len(json.load(open(f))['frames']) for f in files), 1)):.1f}% of frames")
    print(f"|p_obj_base|: median {np.median(m):.3f} m  p5 {np.percentile(m, 5):.3f}  "
          f"p95 {np.percentile(m, 95):.3f}  max {m.max():.3f}")
    # Sanity band: the target should sit within a few metres of the base for a household task.
    # Anything beyond ~10 m means a frame convention is wrong, not that the robot is far away.
    if m.max() > 10.0:
        print("  !! WARNING: distances >10 m — check the base_quat convention (x,y,z,w vs w,x,y,z)")


if __name__ == "__main__":
    main()

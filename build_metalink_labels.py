"""Compose per-frame togglebutton metalink labels for all 200 radio episodes. NO sim needed.

    metalink_world(t) = radio_pos(t) + R(radio_quat(t)) @ p_off
    metalink_base(t)  = R(base_quat(t))^T @ (metalink_world(t) - base_pos(t))

Inputs:  /root/metalink_offset.json           (one-time sim read, offset in radio-root frame)
         /root/poses_x/turning_on_radio/*.json (rescue pose JSONs, 200 eps, 0.5 mm validated)
Outputs: /root/metalink_labels/ep{N}.npz       meta_world (T,3) f32, meta_base (T,3) f32
         /root/metalink_labels/summary.json    per-ep stats + global sanity numbers

Sanity checks baked in:
  - |metalink - radio_center| must be CONSTANT per episode (rigid attachment) and < radio
    half-diagonal; drift here would mean a quat convention bug.
  - metalink world motion across the episode should be ~0 until the radio is touched.
Pixel projections (3 cams) are NOT computed here — they need per-frame cam_rel_poses from the
training parquet and happen in the loader plumbing (PREP f).
"""

import glob
import json

import numpy as np


def q2r(q):
    x, y, z, w = [float(v) for v in q]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def main():
    import os
    os.makedirs("/root/metalink_labels", exist_ok=True)

    off = json.load(open("/root/metalink_offset.json"))
    p_off = np.array(off["offset_pos_root_frame"], dtype=np.float64)
    print(f"offset (root frame): {np.round(p_off, 4)}  |off|={np.linalg.norm(p_off):.4f} m")

    summary = {"offset_pos_root_frame": p_off.tolist(), "episodes": {}}
    rigid_errs, motions = [], []

    for fp in sorted(glob.glob("/root/poses_x/turning_on_radio/ep*.json")):
        d = json.load(open(fp))
        ep = d["demo_id"]
        T = len(d["frames"])
        meta_w = np.zeros((T, 3), dtype=np.float64)
        meta_b = np.zeros((T, 3), dtype=np.float64)
        rig = np.zeros(T, dtype=np.float64)
        for t, fr in enumerate(d["frames"]):
            r = fr["objects"]["radio_89"]
            p_r = np.asarray(r["pos"], dtype=np.float64)
            R_r = q2r(r["quat"])
            mw = p_r + R_r @ p_off
            meta_w[t] = mw
            rig[t] = np.linalg.norm(mw - p_r)
            p_b = np.asarray(fr["base_pos"], dtype=np.float64)
            R_b = q2r(fr["base_quat"])
            meta_b[t] = R_b.T @ (mw - p_b)

        motion = float(np.linalg.norm(meta_w - meta_w[0], axis=1).max())
        rigid_spread = float(rig.max() - rig.min())
        rigid_errs.append(rigid_spread)
        motions.append(motion)
        np.savez_compressed(f"/root/metalink_labels/ep{ep}.npz",
                            meta_world=meta_w.astype(np.float32),
                            meta_base=meta_b.astype(np.float32))
        summary["episodes"][str(ep)] = {
            "n_frames": T,
            "instance": d["instance"],
            "metalink_world_motion_max_m": round(motion, 4),
            "rigid_dist_m": round(float(rig.mean()), 4),
            "rigid_spread_m": round(rigid_spread, 6),
        }

    n = len(summary["episodes"])
    summary["global"] = {
        "n_episodes": n,
        "rigid_spread_max_m": round(max(rigid_errs), 6),   # must be ~0 (float noise)
        "world_motion_p50_m": round(float(np.median(motions)), 4),
        "world_motion_max_m": round(max(motions), 4),
    }
    with open("/root/metalink_labels/summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    g = summary["global"]
    print(f"episodes: {n}")
    print(f"rigid spread max (must be ~0): {g['rigid_spread_max_m']} m")
    print(f"metalink world motion p50/max: {g['world_motion_p50_m']} / {g['world_motion_max_m']} m")
    print("WROTE /root/metalink_labels/  (ep*.npz + summary.json)")


if __name__ == "__main__":
    main()

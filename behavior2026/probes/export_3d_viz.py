"""Export a REAL 3D scene cloud + foveation-level assignment for the architecture visualization.

Uses the captured oracle-rollout frames (/root/pcl_frames) with the CORRECTED camera mapping
(cam_rel_poses order is [wrist, wrist, HEAD] — index 2 is the head, z=1.39 m; the earlier probe's
registration failure came from assuming index 0).

Single-frame reconstruction (frames sit at different base poses and we have no per-frame base
pose to fuse them): pick a late frame where the robot is at the radio, unproject the head 720x720
RGB-D through the calibrated intrinsics into the BASE frame, and assign each point a foveation
level exactly as the architecture defines them:

    L2 fine   2 cm  — inside a 0.4 m cube ahead of either gripper (toward the target)
    L1 mid    10 cm — within 1.5 m of either end-effector
    L0 coarse 50 cm — everything else (room memory)

Outputs:
  /root/viz_cloud.json — downsampled points, RGB colors, per-point level, level-snapped
                          coordinates (position quantized to that level's voxel grid, which is
                          what makes resolution VISIBLE), plus radio/EE/wrist-cube markers
  /root/viz_render_*.png — matplotlib renders: RGB view, foveation view, top-down
"""

import glob
import json

import numpy as np

FX, FY, CX, CY = 238.9, 315.8, 364.7, 356.2   # calibrated zed @ 720
HEAD_IDX = 2                                   # cam_rel_poses: [wrist, wrist, HEAD]
FRAME = -6                                     # late frame: robot at the radio
STRIDE = 5
MAX_PTS = 9000

VOX = {2: 0.02, 1: 0.10, 0: 0.50}
CUBE = 0.40
MID_R = 1.5


def q2r(q):
    x, y, z, w = [float(v) for v in q]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def main():
    frames = sorted(glob.glob("/root/pcl_frames/frame_*.npz"))
    fp = frames[FRAME]
    z = np.load(fp, allow_pickle=True)
    rgb = np.asarray(z["robot_r1::robot_r1:zed_link:Camera:0::rgb"])[..., :3]
    depth = np.asarray(z["robot_r1::robot_r1:zed_link:Camera:0::depth_linear"]).astype(np.float64)
    prop = np.asarray(z["robot_r1::proprio"]).astype(np.float64)
    cams = np.asarray(z["robot_r1::cam_rel_poses"]).astype(np.float64)
    tp = np.asarray(z["target_points"]).astype(np.float64)

    cam_t = cams[HEAD_IDX * 7: HEAD_IDX * 7 + 3]
    cam_R = q2r(cams[HEAD_IDX * 7 + 3: HEAD_IDX * 7 + 7])
    print(f"frame {fp.split('/')[-1]}  head z={cam_t[2]:.3f} (expect ~1.39)")

    ee_l, ee_r = prop[17:20], prop[42:45]
    radio = tp[0] + ee_l
    print(f"radio (base frame): {np.round(radio, 3)}")

    H, W = depth.shape
    pts, cols = [], []
    for v in range(0, H, STRIDE):
        for u in range(0, W, STRIDE):
            d = depth[v, u]
            if not np.isfinite(d) or d < 0.05 or d > 7.0:
                continue
            xc, yc = (u - CX) / FX * d, (v - CY) / FY * d
            p = cam_R @ np.array([xc, -yc, -d]) + cam_t     # USD: -Z forward, +Y up
            pts.append(p)
            cols.append(rgb[v, u])
    pts = np.array(pts)
    cols = np.array(cols)
    print(f"unprojected {len(pts)} points")

    if len(pts) > MAX_PTS:
        sel = np.random.default_rng(0).choice(len(pts), MAX_PTS, replace=False)
        pts, cols = pts[sel], cols[sel]

    # foveation cubes sit ahead of each gripper, toward the target
    def cube_center(ee):
        d = radio - ee
        n = np.linalg.norm(d)
        return ee + (d / n) * (CUBE / 2) if n > 1e-6 else ee

    c_l, c_r = cube_center(ee_l), cube_center(ee_r)

    def level(p):
        for c in (c_l, c_r):
            if np.all(np.abs(p - c) <= CUBE / 2):
                return 2
        if min(np.linalg.norm(p - ee_l), np.linalg.norm(p - ee_r)) <= MID_R:
            return 1
        return 0

    levels = np.array([level(p) for p in pts])
    snapped = np.array([np.round(p / VOX[l]) * VOX[l] for p, l in zip(pts, levels)])
    n2, n1, n0 = int((levels == 2).sum()), int((levels == 1).sum()), int((levels == 0).sum())
    print(f"levels: L2 fine {n2}  L1 mid {n1}  L0 coarse {n0}")

    out = {
        "pts": np.round(pts, 3).tolist(),
        "snap": np.round(snapped, 3).tolist(),
        "col": cols.astype(int).tolist(),
        "lvl": levels.tolist(),
        "radio": np.round(radio, 3).tolist(),
        "ee_l": np.round(ee_l, 3).tolist(),
        "ee_r": np.round(ee_r, 3).tolist(),
        "cube_l": np.round(c_l, 3).tolist(),
        "cube_r": np.round(c_r, 3).tolist(),
        "cube_size": CUBE,
        "head_cam": np.round(cam_t, 3).tolist(),
    }
    with open("/root/viz_cloud.json", "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print("wrote /root/viz_cloud.json")

    # ---- matplotlib renders -------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    LVL_COLORS = {0: "#4a6fa5", 1: "#e8a33d", 2: "#e84b3d"}

    def scatter3(ax, P, C, s):
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=C, s=s, depthshade=False)
        ax.scatter(*radio, marker="*", s=350, c="red", edgecolors="black", linewidths=1.2)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
        ax.view_init(elev=22, azim=-60)

    fig = plt.figure(figsize=(16, 6.5))
    ax = fig.add_subplot(121, projection="3d")
    scatter3(ax, pts, cols / 255.0, 2)
    ax.set_title("REAL scene cloud — head RGB-D, base frame (radio ★)")
    ax = fig.add_subplot(122, projection="3d")
    order = np.argsort(levels)  # draw fine last
    scatter3(ax, snapped[order], [LVL_COLORS[l] for l in levels[order]],
             np.where(levels[order] == 2, 8, np.where(levels[order] == 1, 4, 2)))
    ax.set_title("FOVEATED memory — same points at L0 50cm / L1 10cm / L2 2cm")
    plt.tight_layout()
    plt.savefig("/root/viz_render_iso.png", dpi=110)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    axes[0].scatter(pts[:, 0], pts[:, 1], c=cols / 255.0, s=2)
    axes[0].scatter(*radio[:2], marker="*", s=300, c="red", edgecolors="black")
    axes[0].set_title("top-down, RGB"); axes[0].set_aspect("equal")
    o = np.argsort(levels)
    axes[1].scatter(snapped[o, 0], snapped[o, 1], c=[LVL_COLORS[l] for l in levels[o]],
                    s=np.where(levels[o] == 2, 10, np.where(levels[o] == 1, 5, 2)))
    axes[1].scatter(*radio[:2], marker="*", s=300, c="red", edgecolors="black")
    axes[1].set_title("top-down, foveation levels"); axes[1].set_aspect("equal")
    plt.tight_layout()
    plt.savefig("/root/viz_render_top.png", dpi=110)
    print("wrote /root/viz_render_iso.png /root/viz_render_top.png")


if __name__ == "__main__":
    main()

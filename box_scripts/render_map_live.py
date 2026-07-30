"""Render the LIVE 3D map snapshots from an eval run into a video + keyframe montage.

This is "what the VLA's map sees" during a real rollout: L0 room memory (faint, 0.5 m),
L1 mid (RGB, 0.1 m), L2 wrist contact cubes (red, 2 cm), the affordance-written target (star),
and the odometry trail (where the robot believes it has been — dead-reckoned, no sim pose).

Input: /root/map_live/snap_*.npz (dumped every 100 steps by AffordanceMapFullRes).
Output: /root/map_live.mp4 + /root/map_live_montage.png
"""

import glob

import numpy as np


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snaps = sorted(glob.glob("/root/map_live/snap_*.npz"))
    if not snaps:
        raise SystemExit("no snapshots in /root/map_live")
    print(f"{len(snaps)} snapshots")

    # global bounds for a stable camera
    allpts = []
    data = []
    for f in snaps:
        z = np.load(f)
        d = {k: z[k] for k in z.files}
        data.append(d)
        for lv in ("l0_pts", "l1_pts"):
            if lv in d and len(d[lv]):
                allpts.append(d[lv].astype(np.float32))
    P = np.concatenate(allpts)
    lo, hi = np.percentile(P, 1, 0) - 0.3, np.percentile(P, 99, 0) + 0.3
    trail = np.array([d["odom"][:2] for d in data])

    frames = []
    for i, d in enumerate(data):
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        if "l0_pts" in d and len(d["l0_pts"]):
            p = d["l0_pts"].astype(np.float32)
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=d["l0_rgb"] / 255.0, s=10, alpha=0.18,
                       depthshade=False)
        if "l1_pts" in d and len(d["l1_pts"]):
            p = d["l1_pts"].astype(np.float32)
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=d["l1_rgb"] / 255.0, s=3, depthshade=False)
        for lv in ("l2_left_pts", "l2_right_pts"):
            if lv in d and len(d[lv]):
                p = d[lv].astype(np.float32)
                ax.scatter(p[:, 0], p[:, 1], p[:, 2], c="red", s=5, depthshade=False)
        if "target" in d:
            t = d["target"]
            ax.scatter([t[0]], [t[1]], [t[2]], marker="*", s=420, c="lime",
                       edgecolors="black", linewidths=1.2)
        od = d["odom"]
        ax.scatter([od[0]], [od[1]], [0.05], marker="s", s=120, c="blue",
                   edgecolors="black")
        k = i + 1
        ax.plot(trail[:k, 0], trail[:k, 1], np.full(k, 0.05), c="blue", lw=1.5, alpha=0.7)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(0, hi[2])
        ax.view_init(elev=32, azim=-55 + i * 1.5)      # slow orbit
        ax.set_title(f"live foveated map — step {int(d['step'])}  "
                     f"(L0 faint / L1 rgb / L2 red / target ★ / odom ■)")
        fig.tight_layout()
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(img)
        plt.close(fig)

    import av
    H, W = frames[0].shape[:2]
    H2, W2 = H - H % 2, W - W % 2
    with av.open("/root/map_live.mp4", "w") as c:
        st = c.add_stream("h264", rate=4)
        st.width, st.height = W2, H2
        st.pix_fmt = "yuv420p"
        for f in frames:
            vf = av.VideoFrame.from_ndarray(f[:H2, :W2], format="rgb24")
            for pkt in st.encode(vf):
                c.mux(pkt)
        for pkt in st.encode():
            c.mux(pkt)

    picks = [0, len(frames) // 3, 2 * len(frames) // 3, len(frames) - 1]
    from PIL import Image
    tiles = [frames[i][:H2, :W2] for i in picks]
    row1 = np.concatenate(tiles[:2], axis=1)
    row2 = np.concatenate(tiles[2:], axis=1)
    Image.fromarray(np.concatenate([row1, row2], axis=0)).save("/root/map_live_montage.png")
    print(f"WROTE /root/map_live.mp4 ({len(frames)} frames) + /root/map_live_montage.png")


if __name__ == "__main__":
    main()

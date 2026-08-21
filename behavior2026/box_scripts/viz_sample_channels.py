"""Visualize every supervision channel of ONE real training sample (demo 10, frame 1125 —
the first grasp-descent initiation). Six panels:

  A/B/C  the three cameras with aux_pixels drawn (head + wristR see the button; wristL blind)
  D      target_points as 3D arrows: EE-L and EE-R -> button, in the BASE frame + T0 decode
  E      T1 room-polar token as a radar chart (12 bearings x 3 range rings around the robot)
  F      T4 L2-left contact cube as 4x4x4 voxels + T5 nearest-surface vector

Output: /root/sample_channels.png
"""

import glob
import json

import numpy as np

DEMO_EP_IDX = 0
FRAME = None  # resolved to first ACQUIRE->MANIP initiation


def main():
    import av
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pyarrow.parquet as pq

    t = pq.read_table(sorted(glob.glob("/root/b1k_radio_map/data/**/*.parquet", recursive=True))[0],
                      columns=["episode_index", "frame_index", "target_points", "stage",
                               "aux_pixels", "map_tokens_full", "observation.state"])
    st = t["stage"].to_numpy()
    eps = t["episode_index"].to_numpy()
    tr = np.where((st[1:] == 2) & (st[:-1] == 1) & (eps[1:] == DEMO_EP_IDX))[0]
    r = int(tr[0] + 1)
    row = t.slice(r, 1)
    fi = int(row["frame_index"][0].as_py())
    tp = np.asarray(row["target_points"][0].as_py()).reshape(2, 3)
    ap = np.asarray(row["aux_pixels"][0].as_py())
    mt = np.asarray(row["map_tokens_full"][0].as_py(), np.float32).reshape(8, 72)
    state = np.asarray(row["observation.state"][0].as_py())
    ee_l, ee_r = state[17:20], state[42:45]
    button = mt[0, :3]

    emap = json.load(open("/root/episode_map.json"))
    seg = emap["segments"][str(DEMO_EP_IDX)]
    fps = 30

    def grab(key):
        ch, f_ = seg[f"videos/{key}/chunk_index"], seg[f"videos/{key}/file_index"]
        from_ts = float(seg[f"videos/{key}/from_timestamp"])
        path = f"/root/b1k_radio_map/videos/{key}/chunk-{ch:03d}/file-{f_:03d}.mp4"
        with av.open(path) as c:
            c.seek(int(from_ts * av.time_base), any_frame=False, backward=True)
            for frv in c.decode(c.streams.video[0]):
                if frv.time is None:
                    continue
                rel = int(round((frv.time - from_ts) * fps))
                if rel >= fi:
                    return frv.to_ndarray(format="rgb24")
        return None

    head = grab("observation.rgb.zed_link_camera_0")
    wl = grab("observation.rgb.left_realsense_link_camera_0")
    wr = grab("observation.rgb.right_realsense_link_camera_0")

    fig = plt.figure(figsize=(17, 10))
    fig.suptitle(f"ONE TRAINING SAMPLE, EVERY CHANNEL — demo 10, frame {fi} "
                 f"(first ACQUIRE→MANIPULATE initiation; one of the 258 oversampled frames)",
                 fontsize=13)

    # A/B/C cameras + aux_pixels
    for k, (img, name, uv) in enumerate((
            (head, "HEAD (zed)", ap[0:3]), (wl, "WRIST LEFT", ap[3:6]), (wr, "WRIST RIGHT", ap[6:9]))):
        ax = fig.add_subplot(2, 3, k + 1)
        ax.imshow(img)
        H = img.shape[0]
        if uv[2] > 0.5:
            ax.scatter([uv[0] * H], [uv[1] * H], s=250, facecolors="none",
                       edgecolors="lime", linewidths=3)
            ax.set_title(f"{name} — aux_pixel ({uv[0]:.2f}, {uv[1]:.2f}) VISIBLE", fontsize=10)
        else:
            ax.set_title(f"{name} — button NOT visible (vis=0, trains the negative)", fontsize=10)
        ax.axis("off")

    # D target_points 3D
    ax = fig.add_subplot(2, 3, 4, projection="3d")
    ax.scatter(0, 0, 0, c="blue", s=90, marker="s", label="base origin")
    ax.scatter(*ee_l, c="orange", s=70, label="EE left")
    ax.scatter(*ee_r, c="brown", s=70, label="EE right")
    ax.scatter(*button, c="lime", s=220, marker="*", edgecolors="black", label="button (T0)")
    for ee, v, col in ((ee_l, tp[0], "orange"), (ee_r, tp[1], "brown")):
        ax.quiver(*ee, *v, color=col, arrow_length_ratio=0.08, linewidth=2)
    ax.text(*button, f"  |dL|={np.linalg.norm(tp[0]):.2f}m |dR|={np.linalg.norm(tp[1]):.2f}m",
            fontsize=8)
    ax.set_title("target_points: EE→button displacement vectors (base frame, raw meters)\n"
                 f"T0 token: pos={np.round(button, 2)} conf={mt[0, 3]:.2f} "
                 f"stale={mt[0, 4]:.1f} range={mt[0, 5]:.2f}m", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")

    # E room polar radar
    ax = fig.add_subplot(2, 3, 5, projection="polar")
    occ = mt[1, :36].reshape(12, 3)
    theta_edges = np.linspace(-np.pi, np.pi, 13)
    r_edges = np.array([0.0, 1.5, 3.0, 6.0])
    T_, R_ = np.meshgrid(theta_edges, r_edges, indexing="ij")
    pc = ax.pcolormesh(T_, R_, occ, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_theta_zero_location("N")
    ax.set_title("T1 room-polar token: occupancy 12 bearings × 3 ranges\n"
                 "(robot at center, facing up — walls/furniture as the model reads them)",
                 fontsize=9)
    fig.colorbar(pc, ax=ax, shrink=0.6)

    # F L2 cube voxels
    ax = fig.add_subplot(2, 3, 6, projection="3d")
    occ3 = mt[4, :64].reshape(4, 4, 4) > 0
    colors = np.empty(occ3.shape, dtype=object)
    colors[occ3] = "#e85d4b"
    ax.voxels(occ3, facecolors=colors, edgecolor="#00000030")
    ns = mt[5, 24:27]
    ax.quiver(2, 2, 2, ns[0] * 30, ns[1] * 30, ns[2] * 30, color="blue", linewidth=2.5,
              arrow_length_ratio=0.15)
    ax.set_title(f"T4 L2-LEFT contact cube (40cm @ 2cm→4³ pooled): {occ3.mean():.0%} filled\n"
                 f"T5 nearest-surface vector {np.round(ns, 2)} m (→ the TABLE 6cm below; ×30 for display)",
                 fontsize=9)

    plt.tight_layout()
    plt.savefig("/root/sample_channels.png", dpi=110)
    print(f"WROTE /root/sample_channels.png (frame {fi})")


if __name__ == "__main__":
    main()

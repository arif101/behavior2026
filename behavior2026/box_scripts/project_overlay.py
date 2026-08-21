"""Project composed metalink labels into HEAD-camera pixels and overlay on real demo frames.

Validation for PREP(f) projection plumbing: if the green cross sits on the radio's button in
every frame — including AFTER pickup, when the radio is in the hand — the whole chain
(pose JSONs -> offset compose -> base frame -> robot2cam pose -> intrinsics) is correct.

Uses the VERIFIED episode map (/root/episode_map.json, raw_episode_id + 5-point physical check;
the naive id assumption was wrong for 192/200 episodes), the v3 chunked-video segment offsets
(from_timestamp seek), and the calibrated zed intrinsics rescaled to stored resolution.
Camera convention (validated in export_3d_viz.py): looks down -Z, +Y up.

Output: /root/proj_overlay_ep{N}.png — 2x4 montage, early -> late frames.
"""

import glob
import json

import numpy as np

ROOT = "/root/b1k_radio2"
FX, FY, CX, CY = 238.9, 315.8, 364.7, 356.2  # zed @ 720
KEY = "observation.rgb.zed_link_camera_0"
CAM_COL = "observation.robot2cam_pose.zed_link_camera_0"
DEMOS = [10, 990, 3000]


def q2r(q):
    x, y, z, w = [float(v) for v in q]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def main():
    import av
    import pyarrow.parquet as pq
    from PIL import Image, ImageDraw

    emap = json.load(open("/root/episode_map.json"))
    inv = {int(d): int(i) for i, d in emap["mapping"].items()}
    fps = json.load(open(f"{ROOT}/meta/info.json")).get("fps", 30)
    data_files = sorted(glob.glob(f"{ROOT}/data/**/*.parquet", recursive=True))

    for demo in DEMOS:
        ep_idx = inv[demo]
        seg = emap["segments"][str(ep_idx)]
        chunk, fil = seg[f"videos/{KEY}/chunk_index"], seg[f"videos/{KEY}/file_index"]
        from_ts = float(seg[f"videos/{KEY}/from_timestamp"])
        vp = f"{ROOT}/videos/{KEY}/chunk-{chunk:03d}/file-{fil:03d}.mp4"

        cam = None
        for df in data_files:
            t = pq.read_table(df, columns=["episode_index", CAM_COL])
            m = t["episode_index"].to_numpy() == ep_idx
            if m.any():
                cam = np.stack(t[CAM_COL].to_numpy()[m])
                break
        lbl = np.load(f"/root/metalink_labels/ep{demo}.npz")["meta_base"]
        T = min(len(cam), len(lbl))
        picks = sorted(int(T * f) for f in (0.05, 0.25, 0.45, 0.6, 0.75, 0.85, 0.93, 0.99))
        print(f"demo {demo} (ep_idx {ep_idx}): T={T}, video {vp} @ {from_ts:.2f}s")

        frames = {}
        with av.open(vp) as c:
            stream = c.streams.video[0]
            c.seek(int(from_ts * av.time_base), any_frame=False, backward=True)
            for fr in c.decode(stream):
                if fr.time is None:
                    continue
                rel = int(round((fr.time - from_ts) * fps))
                if rel in picks and rel not in frames:
                    frames[rel] = fr.to_ndarray(format="rgb24")
                if rel > picks[-1]:
                    break
        missing = [i for i in picks if i not in frames]
        if missing:
            print(f"  WARN: missing frames {missing}")
            picks = [i for i in picks if i in frames]

        H, W = next(iter(frames.values())).shape[:2]
        sx, sy = W / 720.0, H / 720.0
        fx, fy, cx, cy = FX * sx, FY * sy, CX * sx, CY * sy

        tiles = []
        for i in picks:
            img = Image.fromarray(frames[i]).convert("RGB")
            dr = ImageDraw.Draw(img)
            t_c, q_c = cam[i, :3], cam[i, 3:7]
            qv = q2r(q_c).T @ (lbl[i] - t_c)      # point in camera frame
            xc, yc, d = qv[0], -qv[1], -qv[2]      # USD: -Z forward, +Y up
            status = ""
            if d > 0.05:
                u, v = xc / d * fx + cx, yc / d * fy + cy
                if 0 <= u < W and 0 <= v < H:
                    r = max(4, int(6 * sx))
                    dr.ellipse([u - r, v - r, u + r, v + r], outline=(0, 255, 60), width=2)
                    dr.line([u - 2 * r, v, u + 2 * r, v], fill=(0, 255, 60), width=1)
                    dr.line([u, v - 2 * r, u, v + 2 * r], fill=(0, 255, 60), width=1)
                else:
                    status = "OFF-IMG"
            else:
                status = "BEHIND"
            dr.text((3, 3), f"f{i} d={max(d, 0):.2f}m {status}", fill=(255, 255, 0))
            tiles.append(np.asarray(img))

        while len(tiles) < 8:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.concatenate(tiles[:4], axis=1), np.concatenate(tiles[4:8], axis=1)]
        Image.fromarray(np.concatenate(rows, axis=0)).save(f"/root/proj_overlay_ep{demo}.png")
        print(f"  WROTE /root/proj_overlay_ep{demo}.png")


if __name__ == "__main__":
    main()

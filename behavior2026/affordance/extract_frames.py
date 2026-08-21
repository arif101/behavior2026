"""Extract affordance-head training data from the radio demos (PREP c, stage 1).

Per episode, every 8th frame: head RGB as JPG, depth downsampled 4x as 16-bit PNG (for 3D-error
eval unprojection), and the label row — button pixel (validated projection chain), camera-frame
depth of the label, and a VISIBILITY flag (in-image AND measured depth at the pixel agrees with
the label's depth within 15 cm; disagreement = the button is occluded). Non-visible frames are
kept: they are the negatives that teach the confidence output when NOT to fire — that
confidence is what makes the HANDOFF state-machine condition safe.

Outputs: /root/aff_data/frames/ep{demo}_f{t}.jpg + _d.png, /root/aff_data/ep{demo}_labels.npz
"""

import argparse
import glob
import json

import numpy as np

ROOT = "/root/b1k_radio2"
FX, FY, CX, CY = 238.9, 315.8, 364.7, 356.2  # zed @ 720
K_RGB = "observation.rgb.zed_link_camera_0"
K_DEP = "observation.depth_linear.zed_link_camera_0"
CAM_COL = "observation.robot2cam_pose.zed_link_camera_0"
STRIDE = 8
DEPTH_SCALE = 1.0 / 1000.0


def q2r(q):
    x, y, z, w = [float(v) for v in q]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def pose7_compose(base7, rel7):
    Rb = q2r(base7[3:7])
    t = np.asarray(base7[:3]) + Rb @ np.asarray(rel7[:3])
    R = Rb @ q2r(rel7[3:7])
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w > 1e-6:
        q = [(R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w),
             (R[1, 0] - R[0, 1]) / (4 * w), w]
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(max(1e-12, 1 + R[i, i] - R[j, j] - R[k, k])) * 2
        q = [0, 0, 0, 0]
        q[i], q[j], q[k], q[3] = s / 4, (R[j, i] + R[i, j]) / s, (R[k, i] + R[i, k]) / s, \
            (R[k, j] - R[j, k]) / s
    return np.array([*t, *q])


def decode_pair(seg, T, fps):
    import av

    def gen(key, depth):
        ch, fi = seg[f"videos/{key}/chunk_index"], seg[f"videos/{key}/file_index"]
        path = f"{ROOT}/videos/{key}/chunk-{ch:03d}/file-{fi:03d}.mp4"
        from_ts = float(seg[f"videos/{key}/from_timestamp"])
        with av.open(path) as c:
            c.seek(int(from_ts * av.time_base), any_frame=False, backward=True)
            for fr in c.decode(c.streams.video[0]):
                if fr.time is None:
                    continue
                rel = int(round((fr.time - from_ts) * fps))
                if rel < 0:
                    continue
                if rel >= T:
                    break
                if depth:
                    a = fr.to_ndarray()
                    if a.ndim == 3:
                        a = a[..., 0]
                    yield rel, a.astype(np.float32) * DEPTH_SCALE
                else:
                    yield rel, fr.to_ndarray(format="rgb24")
    return gen(K_RGB, False), gen(K_DEP, True)


def main():
    import os

    import pyarrow.parquet as pq
    from PIL import Image

    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", type=int, required=True)
    a = ap.parse_args()
    demo = a.demo

    os.makedirs("/root/aff_data/frames", exist_ok=True)
    emap = json.load(open("/root/episode_map.json"))
    ep_idx = {int(d): int(i) for i, d in emap["mapping"].items()}[demo]
    seg = emap["segments"][str(ep_idx)]
    fps = json.load(open(f"{ROOT}/meta/info.json")).get("fps", 30)

    cam = None
    for df in sorted(glob.glob(f"{ROOT}/data/**/*.parquet", recursive=True)):
        t = pq.read_table(df, columns=["episode_index", CAM_COL])
        m = t["episode_index"].to_numpy() == ep_idx
        if m.any():
            cam = np.stack(t[CAM_COL].to_numpy()[m])
            break
    pj = json.load(open(f"/root/poses_x/turning_on_radio/ep{demo}.json"))
    base = np.array([[*f["base_pos"], *f["base_quat"]] for f in pj["frames"]])
    meta_b = np.load(f"/root/metalink_labels/ep{demo}.npz")["meta_base"]
    T = min(len(cam), len(base), len(meta_b))

    rgb_it, dep_it = decode_pair(seg, T, fps)
    rgb_buf, dep_buf = {}, {}
    rows = []
    keep = set(range(0, T, STRIDE))
    for t in range(T):
        for buf, it in ((rgb_buf, rgb_it), (dep_buf, dep_it)):
            while t not in buf:
                k, v = next(it)
                buf[k] = v
        rgb, dep = rgb_buf.pop(t), dep_buf.pop(t)
        if t not in keep:
            continue
        # label pixel via the validated chain (all in BASE frame -> camera frame)
        t_c, q_c = cam[t, :3], cam[t, 3:7]
        qv = q2r(q_c).T @ (meta_b[t] - t_c)
        xc, yc, d = qv[0], -qv[1], -qv[2]
        H, W = dep.shape
        u = xc / max(d, 1e-6) * FX + CX
        v = yc / max(d, 1e-6) * FY + CY
        vis = False
        if d > 0.05 and 0 <= u < W and 0 <= v < H:
            meas = dep[int(v), int(u)]
            vis = bool(np.isfinite(meas) and abs(meas - d) < 0.15)
        Image.fromarray(rgb).save(f"/root/aff_data/frames/ep{demo}_f{t}.jpg", quality=90)
        d16 = (dep[::4, ::4] * 1000).clip(0, 65535).astype(np.uint16)
        Image.fromarray(d16, mode="I;16").save(f"/root/aff_data/frames/ep{demo}_f{t}_d.png")
        rows.append([t, u, v, d, float(vis), *meta_b[t], *base[t], *cam[t]])

    arr = np.array(rows, dtype=np.float32)
    np.savez_compressed(f"/root/aff_data/ep{demo}_labels.npz", rows=arr)
    nv = int(arr[:, 4].sum())
    print(f"ep{demo}: {len(arr)} samples, visible {nv} ({nv / max(len(arr), 1):.0%})")


if __name__ == "__main__":
    main()

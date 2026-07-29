"""Offline map driver — build foveated-map token streams from the training dataset (PREP b).

Per episode: decode head RGB + depth videos in lockstep (v3 chunked files, from_timestamp seek),
read per-frame robot2cam poses (parquet), base pose (replay-validated pose JSONs), target =
composed metalink label. Runs TWO map instances — with and without target writes — because
anti-shortcut dropout at train time needs a target-blind token stream.

Outputs per episode: /root/map_tokens/ep{demo}.npz  (tokens_full, tokens_notarget: (T,8,72) f16)
Smoke mode (--smoke): ep10 only, profile ms/frame, dump level renders at 4 checkpoints.

Depth video probe: the depth stream's pixel format + scale are DISCOVERED (gray16 -> mm assumed,
8-bit -> rejected loudly: 2.7 cm quantization would poison L2).
"""

import argparse
import glob
import json
import time

import numpy as np

import sys
sys.path.insert(0, "/root")
from foveated_map import FoveatedMap, MapFrame, q2r  # noqa: E402

ROOT = "/root/b1k_radio2"
FX, FY, CX, CY = 238.9, 315.8, 364.7, 356.2  # zed @ 720
K_RGB = "observation.rgb.zed_link_camera_0"
K_DEP = "observation.depth_linear.zed_link_camera_0"
CAMS = {"head": "observation.robot2cam_pose.zed_link_camera_0",
        "left": "observation.robot2cam_pose.left_realsense_link_camera_0",
        "right": "observation.robot2cam_pose.right_realsense_link_camera_0"}


def pose7_compose(base7, rel7):
    """world_pose of a frame given base world pose and pose relative to base."""
    Rb = q2r(base7[3:7])
    t = np.asarray(base7[:3]) + Rb @ np.asarray(rel7[:3])
    Rr = q2r(rel7[3:7])
    R = Rb @ Rr
    # R -> quat (xyzw)
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w > 1e-6:
        q = [(R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w),
             (R[1, 0] - R[0, 1]) / (4 * w), w]
    else:  # fallback for near-pi rotations
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(max(1e-12, 1 + R[i, i] - R[j, j] - R[k, k])) * 2
        q = [0, 0, 0, 0]
        q[i] = s / 4
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
        q[3] = (R[k, j] - R[j, k]) / s
    return np.array([*t, *q])


def decode_video(path, from_ts, n_frames, fps, to_float_depth=None):
    """Yield (rel_idx, ndarray) for rel_idx in [0, n_frames)."""
    import av
    with av.open(path) as c:
        stream = c.streams.video[0]
        c.seek(int(from_ts * av.time_base), any_frame=False, backward=True)
        for fr in c.decode(stream):
            if fr.time is None:
                continue
            rel = int(round((fr.time - from_ts) * fps))
            if rel < 0:
                continue
            if rel >= n_frames:
                break
            if to_float_depth is not None:
                arr = fr.to_ndarray()
                if arr.ndim == 3:
                    arr = arr[..., 0] if arr.shape[-1] in (3, 4) else arr[0]
                yield rel, arr.astype(np.float32) * to_float_depth
            else:
                yield rel, fr.to_ndarray(format="rgb24")


def probe_depth(path, from_ts, fps):
    import av
    with av.open(path) as c:
        stream = c.streams.video[0]
        fmt = stream.codec_context.pix_fmt
        c.seek(int((from_ts + 5.0) * av.time_base), any_frame=False, backward=True)
        for fr in c.decode(stream):
            arr = fr.to_ndarray()
            print(f"depth probe: pix_fmt={fmt} dtype={arr.dtype} shape={arr.shape} "
                  f"min={arr.min()} max={arr.max()} mean={arr.mean():.1f}")
            if arr.dtype == np.uint16:
                scale = 1.0 / 1000.0     # mm -> m
            elif "16" in str(fmt) or arr.max() > 300:
                scale = 1.0 / 1000.0
            else:
                raise SystemExit(f"REJECT: depth video looks 8-bit (max={arr.max()}) — "
                                 "2.7 cm quantization poisons L2; need raw depth source")
            m = arr.astype(np.float32) * scale
            print(f"  as meters: min={m.min():.3f} max={m.max():.3f} median={np.median(m):.3f}"
                  f" (room scale expected 0.5-7)")
            return scale


def run_episode(demo, emap, smoke=False):
    import pyarrow.parquet as pq

    inv = {int(d): int(i) for i, d in emap["mapping"].items()}
    ep_idx = inv[demo]
    seg = emap["segments"][str(ep_idx)]
    fps = json.load(open(f"{ROOT}/meta/info.json")).get("fps", 30)

    cam = {}
    for df in sorted(glob.glob(f"{ROOT}/data/**/*.parquet", recursive=True)):
        t = pq.read_table(df, columns=["episode_index", *CAMS.values()])
        m = t["episode_index"].to_numpy() == ep_idx
        if m.any():
            for k, col in CAMS.items():
                cam[k] = np.stack(t[col].to_numpy()[m])
            break
    pj = json.load(open(f"/root/poses_x/turning_on_radio/ep{demo}.json"))
    base = np.array([[*f["base_pos"], *f["base_quat"]] for f in pj["frames"]])
    meta_w = np.load(f"/root/metalink_labels/ep{demo}.npz")["meta_world"]
    T = min(len(cam["head"]), len(base), len(meta_w))
    if smoke:
        T = min(T, 600)

    def vpath(key):
        ch, fi = seg[f"videos/{key}/chunk_index"], seg[f"videos/{key}/file_index"]
        return f"{ROOT}/videos/{key}/chunk-{ch:03d}/file-{fi:03d}.mp4", \
            float(seg[f"videos/{key}/from_timestamp"])

    dp, dts = vpath(K_DEP)
    scale = probe_depth(dp, dts, fps) if smoke else getattr(run_episode, "_scale", None)
    if scale is None:
        scale = probe_depth(dp, dts, fps)
    run_episode._scale = scale

    rp, rts = vpath(K_RGB)
    rgb_it = decode_video(rp, rts, T, fps)
    dep_it = decode_video(dp, dts, T, fps, to_float_depth=scale)

    m_full = FoveatedMap()
    toks_f = np.zeros((T, 8, FoveatedMap.TOK_D), np.float16)
    toks_b = np.zeros((T, 8, FoveatedMap.TOK_D), np.float16)
    times, snaps, prof_acc = [], {}, {}
    snap_at = {int(T * f) for f in (0.05, 0.4, 0.7, 0.97)} if smoke else set()

    rgb_buf, dep_buf = dict(rgb_it.__next__() for _ in range(0)), {}
    ri, di = iter(rgb_it), iter(dep_it)
    for t in range(T):
        while t not in rgb_buf:
            k, v = next(ri)
            rgb_buf[k] = v
        while t not in dep_buf:
            k, v = next(di)
            dep_buf[k] = v
        rgb, dep = rgb_buf.pop(t), dep_buf.pop(t)
        H = rgb.shape[0]
        s = H / 720.0
        intr = (FX * s, FY * s, CX * s, CY * s)
        cam_w = pose7_compose(base[t], cam["head"][t])
        wrists = {sd: pose7_compose(base[t], cam[sd][t]) for sd in ("left", "right")}
        fr = MapFrame(rgb, dep, cam_w, base[t], wrists, intr, t)

        t0 = time.perf_counter()
        m_full.update(fr)
        m_full.write_target(meta_w[t], 1.0)
        tq = time.perf_counter()
        toks_f[t] = m_full.query()
        tq2 = time.perf_counter()
        times.append((tq2 - t0) * 1000)
        prof = dict(m_full.prof)
        prof["query"] = tq2 - tq
        for k, v in prof.items():
            prof_acc[k] = prof_acc.get(k, 0.0) + v * 1000

        toks_b[t] = FoveatedMap.blind_view(toks_f[t].astype(np.float32))

        if t in snap_at:
            snaps[t] = m_full.snapshot()

    import os
    os.makedirs("/root/map_tokens", exist_ok=True)
    np.savez_compressed(f"/root/map_tokens/ep{demo}.npz",
                        tokens_full=toks_f, tokens_notarget=toks_b)
    tm = np.array(times)
    print(f"ep{demo}: T={T}  update+query ms/frame p50={np.median(tm):.1f} "
          f"p95={np.percentile(tm, 95):.1f} max={tm.max():.1f}  (gate <20)")
    print("  breakdown ms/frame:", {k: round(v / T, 1) for k, v in prof_acc.items()})
    return snaps, tm


def render_snaps(demo, snaps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for t, s in snaps.items():
        fig = plt.figure(figsize=(14, 7))
        for ci, (view, elev, azim) in enumerate((("iso", 25, -60), ("top", 88, -90))):
            ax = fig.add_subplot(1, 2, ci + 1, projection="3d")
            l0, l1 = s["l0"], s["l1"]
            ax.scatter(*l0["pts"].T, c=l0["rgb"] / 255, s=14, alpha=0.25, depthshade=False)
            ax.scatter(*l1["pts"].T, c=l1["rgb"] / 255, s=4, depthshade=False)
            for side, mk in (("left", "^"), ("right", "v")):
                g = s.get(f"l2_{side}")
                if g is not None and len(g["pts"]):
                    ax.scatter(*g["pts"].T, c="red", s=2, marker=mk, depthshade=False)
            if s["target"] is not None:
                ax.scatter(*s["target"], marker="*", s=380, c="lime",
                           edgecolors="black", linewidths=1.2)
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"ep{demo} f{t} {view} — L0 faint / L1 solid / L2 red / target ★")
        plt.tight_layout()
        plt.savefig(f"/root/map_viz/ep{demo}_f{t}.png", dpi=100)
        plt.close(fig)
    print(f"renders -> /root/map_viz/ep{demo}_f*.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    import os
    os.makedirs("/root/map_viz", exist_ok=True)
    emap = json.load(open("/root/episode_map.json"))
    snaps, tm = run_episode(a.demo, emap, smoke=a.smoke)
    if snaps:
        render_snaps(a.demo, snaps)

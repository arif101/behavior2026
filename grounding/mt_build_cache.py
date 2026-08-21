"""Build the v0.5 multi-task frame cache (parallel across episodes).

MEMORY-SAFE: the box cgroup caps RAM at 50GB (banked gotcha: `free` shows the
host's 503GB). RGB is written by ffmpeg straight to JPEG files (no buffer);
depth is STREAMED from ffmpeg stdout in 256-frame chunks (~0.5GB peak/worker).

Per (task, file_idx) under CACHE/<task>/ep{f:03d}/:
  frames/f_{i:05d}.jpg   518x518 RGB (ffmpeg area-downscaled, q~95)
  depth.npy              float16 [N,148,148]  meters (area-pooled from 720)
  lab.npz                frame_k [N] i32; uvz [N,K,3] f32 (u,v @720p, z m);
                         margin [N,K] f16 (z_proj - depth720@pixel, NaN unknown);
                         inframe [N,K] bool; obj_names [K] str; obj_cats [K] i32
                         (global vocab id, -1 = unmapped)
CACHE/meta.json: per-episode table; CACHE/vocab.json: global vocabulary.

Every 5th video frame (FRAME_STRIDE).
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mt_common import (CACHE, DEPTH_BIAS, DEPTH_SCALE, DGRID, FRAME_STRIDE, H,
                       IMG, W, all_tasks, build_vocab, episode_table,
                       labels_path, map_category, project, video_path)

CHUNK = 256  # depth frames per streamed chunk (256*720*720*4B = 0.53GB)


def build_episode(task, file_idx, demo, n_video, vocab_index):
    import torch
    import torch.nn.functional as Fnn

    out_dir = os.path.join(CACHE, task, f"ep{file_idx:03d}")
    done_flag = os.path.join(out_dir, ".done")
    if os.path.exists(done_flag):
        return f"{task}/ep{file_idx:03d}: cached"
    fdir = os.path.join(out_dir, "frames")
    os.makedirs(fdir, exist_ok=True)

    ks = list(range(0, n_video, FRAME_STRIDE))
    N = len(ks)

    # ---- labels first (needs no video)
    rows = [json.loads(l) for l in open(labels_path(task, demo))]
    obj_names = list(rows[0]["objs"].keys())
    K = len(obj_names)
    cats = [map_category(o, vocab_index) for o in obj_names]
    obj_cats = np.array([vocab_index.get(c, -1) if c is not None else -1
                         for c in cats], dtype=np.int32)
    uvz = np.zeros((N, K, 3), dtype=np.float32)
    margin = np.full((N, K), np.nan, dtype=np.float32)
    inframe = np.zeros((N, K), dtype=bool)
    for i, k in enumerate(ks):
        ridx = min(int((k + 1) * len(rows) / n_video) - 1, len(rows) - 1)
        r = rows[ridx]
        for j, o in enumerate(obj_names):
            u, v, z, infr = project(r["M"], r["objs"][o])
            uvz[i, j] = (u, v, z)
            inframe[i, j] = infr
    del rows

    # ---- RGB: ffmpeg decodes straight to JPEG files (no python-side buffer)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-threads", "4",
         "-i", video_path(task, "rgb", file_idx),
         "-vf", f"select=not(mod(n\\,{FRAME_STRIDE})),scale={IMG}:{IMG}:flags=area",
         "-vsync", "0", "-start_number", "0", "-q:v", "2",
         os.path.join(fdir, "f_%05d.jpg")],
        check=True)
    n_jpg = len([f for f in os.listdir(fdir) if f.endswith(".jpg")])
    assert n_jpg == N, f"{task}/ep{file_idx}: jpg {n_jpg} != expected {N}"

    # ---- depth: streamed gray16le chunks
    dep_small = np.zeros((N, DGRID, DGRID), dtype=np.float16)
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-threads", "4",
         "-i", video_path(task, "depth", file_idx),
         "-vf", f"select=not(mod(n\\,{FRAME_STRIDE}))", "-vsync", "0",
         "-f", "rawvideo", "-pix_fmt", "gray16le", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    fb = H * W * 2
    i0 = 0
    while True:
        want = min(CHUNK, N - i0)
        if want <= 0:
            extra = proc.stdout.read(fb)
            assert not extra, f"{task}/ep{file_idx}: extra depth frames beyond {N}"
            break
        buf = proc.stdout.read(want * fb)
        if not buf:
            break
        got = len(buf) // fb
        assert len(buf) % fb == 0, f"{task}/ep{file_idx}: partial depth frame"
        dep = np.frombuffer(buf, dtype=np.uint16).reshape(got, H, W)
        dep_m = dep.astype(np.float32) * DEPTH_SCALE + DEPTH_BIAS
        # margins at full res for in-frame instances of these frames
        for ii in range(got):
            i = i0 + ii
            for j in range(K):
                if inframe[i, j]:
                    u, v, z = uvz[i, j]
                    d = float(dep_m[ii, min(int(v), H - 1), min(int(u), W - 1)])
                    if d > 0.05:
                        margin[i, j] = z - d
        dep_small[i0:i0 + got] = Fnn.interpolate(
            torch.from_numpy(dep_m).unsqueeze(1), size=(DGRID, DGRID),
            mode="area").squeeze(1).numpy().astype(np.float16)
        i0 += got
    proc.stdout.close()
    proc.wait()
    assert i0 == N, f"{task}/ep{file_idx}: depth {i0} != expected {N}"

    np.save(os.path.join(out_dir, "depth.npy"), dep_small)
    np.savez(os.path.join(out_dir, "lab.npz"),
             frame_k=np.array(ks, dtype=np.int32), uvz=uvz,
             margin=margin.astype(np.float16), inframe=inframe,
             obj_names=np.array(obj_names), obj_cats=obj_cats)
    open(done_flag, "w").write("ok")
    n_in = int(inframe.any(1).sum())
    return f"{task}/ep{file_idx:03d} demo{demo}: {N} samples, {n_in} frames w/ inframe obj"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tasks", nargs="*", default=None)
    args = ap.parse_args()

    vocab, vocab_index, n_base = build_vocab()
    os.makedirs(CACHE, exist_ok=True)

    tasks = args.tasks or all_tasks()
    jobs = []
    meta = {"vocab_size": len(vocab), "n_base_vocab": n_base, "episodes": {}}
    for t in tasks:
        table = episode_table(t)
        meta["episodes"][t] = table
        for r in table:
            jobs.append((t, r["file"], r["demo"], r["n_video"]))
    with open(os.path.join(CACHE, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    with open(os.path.join(CACHE, "vocab.json"), "w") as f:
        json.dump({"vocab": vocab, "n_base": n_base}, f)

    # biggest first for better parallel packing
    jobs.sort(key=lambda j: -j[3])
    print(f"{len(jobs)} episodes, vocab {len(vocab)} ({n_base} base)", flush=True)
    n_fail = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(build_episode, t, fi, d, nv, vocab_index): (t, fi)
                for t, fi, d, nv in jobs}
        for fu in as_completed(futs):
            try:
                print(fu.result(), flush=True)
            except Exception as e:
                n_fail += 1
                print(f"FAILED {futs[fu]}: {e!r}", flush=True)
    print(f"done, {n_fail} failures", flush=True)


if __name__ == "__main__":
    main()

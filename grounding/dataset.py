"""M1 pilot grounding dataset: BEHAVIOR-2026 turning_on_radio, ZED head camera.

Frame cache layout (built by `python dataset.py --build`):
  /root/frame_cache/ep{ei:03d}_rgb.npy    uint8   [N, 518, 518, 3]   (every 5th frame, ffmpeg area-downscaled)
  /root/frame_cache/ep{ei:03d}_depth.npy  float16 [N, 148, 148]      (meters, area-pooled from 720)
  /root/frame_cache/ep{ei:03d}_lab.npy    float32 [N, 5]             (k, u720, v720, z_m, inframe)
  /root/frame_cache/meta.json

Coordinate conventions (all GT/eval in ORIGINAL 720p pixel frame — no intrinsics
rescale is ever applied; the 518 resize only feeds the backbone, and heatmap /
depth grids are mapped back to 720p by uniform cell-center scaling):
  p_cam = R.T @ (p_world - t),  R = M[:3,:3], t = M[:3,3]
  z = -p_cam[2];  u = cx + fx*x/z;  v = cy - fy*y/z
Depth video: HEVC gray12le, decoded via ffmpeg to gray16le.
  z_meters = DEPTH_SCALE * raw16 + DEPTH_BIAS
  (label-anchored fit, median residual 0.1 cm over 5 episodes, z in 0.7-1.5 m;
   consistent with a 6m-range/12-bit encoding: 6/4096/16.008 = 9.15e-5)
"""

import argparse
import json
import os
import subprocess

import numpy as np
import torch
from torch.utils.data import Dataset

# Calibrated intrinsics at 720x720 (fitted upstream, 3.7px residual -- do not recompute).
FX, FY, CX, CY = 238.9, 315.8, 364.7, 356.2
W = H = 720
IMG = 518            # backbone input (37x37 patches @14px)
DGRID = 148          # cached depth grid (= 4 * 37; cell j center at u720 = (j+0.5)*720/148)
HM = 180             # output heatmap (cell j center at u720 = (j+0.5)*4)
FRAME_STRIDE = 5
DEPTH_SCALE = 8.928079e-05
DEPTH_BIAS = -0.0119

DEMO_IDS = [10, 20, 30, 40, 50, 60, 70, 80, 100, 110, 120, 130, 140,
            160, 170, 180, 190, 200, 210, 220]
# Held out by DEMO id {50, 130} -> file indices {4, 11}.
# (Task text said "file-012 (demo 130)" but per the demo-id list file-011 = demo 130.)
HELDOUT_EPS = [4, 11]
TRAIN_EPS = [i for i in range(20) if i not in HELDOUT_EPS]

ROOT = "/root/replay_out/b1k/turning_on_radio"
RGB_TPL = ROOT + "/videos/observation.rgb.zed_link_camera_0/chunk-000/file-{:03d}.mp4"
DEP_TPL = ROOT + "/videos/observation.depth_linear.zed_link_camera_0/chunk-000/file-{:03d}.mp4"
LAB_TPL = "/root/labels/labels_{}.jsonl"
META_DIR = ROOT + "/meta/episodes/chunk-000"
CACHE = "/root/frame_cache"
CATEGORY_IDS = {"radio": 0}


def episode_lengths():
    import pandas as pd
    lens = []
    for ei in range(20):
        df = pd.read_parquet(os.path.join(META_DIR, f"file-{ei:03d}.parquet"))
        lens.append(int(df["length"].iloc[0]))
    return lens


def _ffmpeg_frames(path, stride, vf_extra, pix_fmt, shape):
    """Decode every `stride`-th frame in one pass; return [N, *shape] array."""
    vf = f"select=not(mod(n\\,{stride}))"
    if vf_extra:
        vf += "," + vf_extra
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf", vf, "-vsync", "0",
           "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    dtype = np.uint16 if pix_fmt == "gray16le" else np.uint8
    arr = np.frombuffer(out, dtype=dtype)
    return arr.reshape(-1, *shape)


def project(M_flat, p_world):
    M = np.asarray(M_flat, dtype=np.float64).reshape(4, 4)
    R, t = M[:3, :3], M[:3, 3]
    pc = R.T @ (np.asarray(p_world) - t)
    z = -pc[2]
    if z <= 1e-6:
        return -1.0, -1.0, z, False
    u = CX + FX * pc[0] / z
    v = CY - FY * pc[1] / z
    return u, v, z, bool(0 <= u < W and 0 <= v < H)


def build_cache():
    os.makedirs(CACHE, exist_ok=True)
    lens = episode_lengths()
    meta = {"episodes": []}
    for ei in range(20):
        demo, n_video = DEMO_IDS[ei], lens[ei]
        rows = [json.loads(l) for l in open(LAB_TPL.format(demo))]
        ks = list(range(0, n_video, FRAME_STRIDE))

        rgb = _ffmpeg_frames(RGB_TPL.format(ei), FRAME_STRIDE,
                             f"scale={IMG}:{IMG}:flags=area", "rgb24", (IMG, IMG, 3))
        dep_raw = _ffmpeg_frames(DEP_TPL.format(ei), FRAME_STRIDE, None, "gray16le", (H, W))
        assert len(rgb) == len(ks) and len(dep_raw) == len(ks), \
            f"ep{ei}: decoded {len(rgb)}/{len(dep_raw)} vs expected {len(ks)}"

        dep_m = dep_raw.astype(np.float32) * DEPTH_SCALE + DEPTH_BIAS
        dep_m = torch.nn.functional.interpolate(
            torch.from_numpy(dep_m).unsqueeze(1), size=(DGRID, DGRID), mode="area"
        ).squeeze(1).numpy().astype(np.float16)

        lab = np.zeros((len(ks), 5), dtype=np.float32)
        for i, k in enumerate(ks):
            ridx = min(int((k + 1) * len(rows) / n_video) - 1, len(rows) - 1)
            r = rows[ridx]
            u, v, z, inframe = project(r["M"], r["objs"]["radio_89"])
            lab[i] = (k, u, v, z, float(inframe))

        np.save(f"{CACHE}/ep{ei:03d}_rgb.npy", rgb)
        np.save(f"{CACHE}/ep{ei:03d}_depth.npy", dep_m)
        np.save(f"{CACHE}/ep{ei:03d}_lab.npy", lab)
        n_in = int(lab[:, 4].sum())
        meta["episodes"].append({"ep": ei, "demo": demo, "n_video": n_video,
                                 "n_samples": len(ks), "n_inframe": n_in})
        print(f"ep{ei:03d} demo {demo}: {len(ks)} samples, {n_in} in-frame", flush=True)
    with open(f"{CACHE}/meta.json", "w") as f:
        json.dump(meta, f, indent=1)


class GroundingDataset(Dataset):
    """Returns per-sample: rgb uint8 [518,518,3], depth float32 [148,148] (m),
    uv float32 [2] (720p GT pixel), z float32 (m), cat int64, ep/frame ids."""

    def __init__(self, episodes, inframe_only=True):
        self.items = []   # (ep, local_idx)
        self.rgb, self.dep, self.lab = {}, {}, {}
        for ei in episodes:
            self.rgb[ei] = np.load(f"{CACHE}/ep{ei:03d}_rgb.npy", mmap_mode="r")
            self.dep[ei] = np.load(f"{CACHE}/ep{ei:03d}_depth.npy", mmap_mode="r")
            lab = np.load(f"{CACHE}/ep{ei:03d}_lab.npy")
            self.lab[ei] = lab
            for i in range(len(lab)):
                if not inframe_only or lab[i, 4] > 0.5:
                    self.items.append((ei, i))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ei, i = self.items[idx]
        lab = self.lab[ei][i]
        return {
            "rgb": torch.from_numpy(np.ascontiguousarray(self.rgb[ei][i])),
            "depth": torch.from_numpy(self.dep[ei][i].astype(np.float32)),
            "uv": torch.tensor(lab[1:3], dtype=torch.float32),
            "z": torch.tensor(lab[3], dtype=torch.float32),
            "cat": torch.tensor(CATEGORY_IDS["radio"], dtype=torch.long),
            "ep": torch.tensor(ei, dtype=torch.long),
            "frame": torch.tensor(int(lab[0]), dtype=torch.long),
        }


def gaussian_target(uv, hm=HM, sigma=2.5, device="cpu"):
    """Normalized gaussian heatmap targets at HM res. uv: [B,2] in 720p coords."""
    scale = hm / float(W)
    g = torch.arange(hm, device=device, dtype=torch.float32) + 0.5
    gx = g.view(1, 1, hm)
    gy = g.view(1, hm, 1)
    cx = (uv[:, 0] * scale).view(-1, 1, 1)
    cy = (uv[:, 1] * scale).view(-1, 1, 1)
    t = torch.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * sigma ** 2))
    return t / t.sum(dim=(1, 2), keepdim=True).clamp_min(1e-8)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()
    if args.build:
        build_cache()

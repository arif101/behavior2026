"""On-distribution control for the instance-transfer gate: evaluate on UNSEEN
frames (stride offset +2) from TRAIN episodes, decoded directly from video.
Gives the honest denominator for held-out-episode vs on-distribution comparison
(train-cache numbers are on the exact trained frames = memorization + argmax
quantization floor)."""

import argparse
import json
import subprocess

import numpy as np
import torch

from dataset import (DEMO_IDS, DEPTH_BIAS, DEPTH_SCALE, DGRID, FRAME_STRIDE,
                     IMG, LAB_TPL, H, W, project, episode_lengths,
                     RGB_TPL, DEP_TPL)
from eval import cam_point, summarize
from model import GroundingModel


def decode_offset(path, stride, offset, vf_extra, pix_fmt, shape):
    vf = f"select=not(mod(n-{offset}\\,{stride}))*gte(n\\,{offset})"
    if vf_extra:
        vf += "," + vf_extra
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf", vf, "-vsync", "0",
           "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    dtype = np.uint16 if pix_fmt == "gray16le" else np.uint8
    return np.frombuffer(out, dtype=dtype).reshape(-1, *shape)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/root/grounding/ckpt_pilot.pt")
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 5, 9, 14, 18])
    ap.add_argument("--offset", type=int, default=2)
    ap.add_argument("--max-per-ep", type=int, default=120)
    ap.add_argument("--out", default="/root/grounding/results_offstride.json")
    args = ap.parse_args()

    dev = "cuda"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = GroundingModel(backbone=ck["config"]["backbone"]).to(dev)
    model.load_state_dict(ck["head"], strict=False)
    model.eval()

    lens = episode_lengths()
    recs = []
    for ei in args.episodes:
        demo, n_video = DEMO_IDS[ei], lens[ei]
        rows = [json.loads(l) for l in open(LAB_TPL.format(demo))]
        rgb = decode_offset(RGB_TPL.format(ei), FRAME_STRIDE, args.offset,
                            f"scale={IMG}:{IMG}:flags=area", "rgb24",
                            (IMG, IMG, 3))
        dep = decode_offset(DEP_TPL.format(ei), FRAME_STRIDE, args.offset,
                            None, "gray16le", (H, W))
        dep = dep.astype(np.float32) * DEPTH_SCALE + DEPTH_BIAS
        dep = torch.nn.functional.interpolate(
            torch.from_numpy(dep).unsqueeze(1), size=(DGRID, DGRID),
            mode="area").squeeze(1)

        ks = [args.offset + i * FRAME_STRIDE for i in range(len(rgb))]
        keep, uvz = [], []
        for i, k in enumerate(ks):
            ridx = min(int((k + 1) * len(rows) / n_video) - 1, len(rows) - 1)
            u, v, z, inframe = project(rows[ridx]["M"], rows[ridx]["objs"]["radio_89"])
            if inframe:
                keep.append(i)
                uvz.append((u, v, z))
        if len(keep) > args.max_per_ep:   # spread evenly across the episode
            sel = np.linspace(0, len(keep) - 1, args.max_per_ep).astype(int)
            keep = [keep[j] for j in sel]
            uvz = [uvz[j] for j in sel]
        uvz = np.array(uvz, dtype=np.float32)

        for s in range(0, len(keep), 64):
            sel = keep[s:s + 64]
            rb = torch.from_numpy(np.ascontiguousarray(rgb[sel])).to(dev)
            db = dep[sel].to(dev)
            cat = torch.zeros(len(sel), dtype=torch.long, device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, dmap = model(rb, db, cat)
            uv_pred = model.argmax_uv(logits.float()).cpu().numpy()
            uv_soft = model.soft_argmax_uv(logits.float()).cpu().numpy()
            z_pred = model.sample_depth(dmap.float(),
                                        torch.from_numpy(uv_pred).to(dev)).cpu().numpy()
            gt = uvz[s:s + 64]
            px = np.linalg.norm(uv_pred - gt[:, :2], axis=1)
            pxs = np.linalg.norm(uv_soft - gt[:, :2], axis=1)
            d3 = np.linalg.norm(cam_point(uv_pred[:, 0], uv_pred[:, 1], z_pred)
                                - cam_point(gt[:, 0], gt[:, 1], gt[:, 2]), axis=1)
            for i in range(len(px)):
                recs.append({"ep": ei, "px": float(px[i]), "px_soft": float(pxs[i]),
                             "dz": float(abs(z_pred[i] - gt[i, 2])),
                             "d3": float(d3[i])})
        print(f"ep{ei:03d} demo {demo}: {len(keep)} off-stride frames", flush=True)

    res = {"offset": args.offset, "episodes": args.episodes,
           "overall": summarize(recs),
           "per_episode": {f"ep{ei:03d}": summarize([r for r in recs if r["ep"] == ei])
                           for ei in args.episodes}}
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

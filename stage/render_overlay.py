"""Render a live overlay video of the stage head running on one episode.

Streams the cached frames of (task, file_idx) through the serve-time
StageEstimator exactly as the director would call it (set_task once, then
update() per 6 Hz perception step), and renders each frame with the model's
live outputs next to the ground-truth labels:

  per arm:  predicted stage (green = matches GT, red = mismatch), GT stage,
            top-3 stage_dist bars, progress bar (pred fill + GT tick), entropy
  shared:   p_sat bar per goal literal with GT ledger dot, stage_age_ratio

Usage (on the box, after build_cache):
  python3 render_overlay.py --task turning_on_radio --fi 3 \
      --ckpt /root/stage_ckpt/best.pt --medians /root/stage_ckpt/stage_medians.json \
      --out /root/overlay_radio.mp4 --speed 2
"""

import argparse
import json
import os
import subprocess

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from common import FPS_CACHE, IMG, TASK_TARGETS, cache_dir
from serve import StageEstimator
from taxonomy import literals_from_task_targets, stage_name
from train import load_backbone

W_PANEL = 506
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

GREEN = (80, 200, 90)
RED = (230, 90, 80)
GRAY = (150, 150, 150)
WHITE = (235, 235, 235)
BLUE = (90, 150, 235)
YELLOW = (235, 200, 80)
BG = (16, 16, 18)


def precompute_tokens(frames_dir, n, backbone, device, batch=32):
    """All-frame DINO patch tokens, kept on CPU f16 ([N, g*g, feat])."""
    import torch.nn.functional as Fnn
    tokens_fn, spec = backbone
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    out = None
    with torch.no_grad():
        for a in range(0, n, batch):
            b = min(n, a + batch)
            imgs = np.stack([
                np.asarray(Image.open(os.path.join(frames_dir, f"f_{i:05d}.jpg"))
                           .convert("RGB"), dtype=np.uint8) for i in range(a, b)])
            x = torch.from_numpy(imgs).to(device).permute(0, 3, 1, 2).float() / 255
            if x.shape[-1] != spec["img"]:
                x = Fnn.interpolate(x, size=(spec["img"], spec["img"]),
                                    mode="bilinear", align_corners=False,
                                    antialias=True)
            x = (x - mean) / std
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=device != "cpu"):
                f = tokens_fn(x)
            if out is None:
                out = torch.empty((n,) + f.shape[1:], dtype=torch.float16)
            out[a:b] = f.half().cpu()
    return out


def bar(draw, x, y, w, h, frac, color, outline=GRAY):
    draw.rectangle([x, y, x + w, y + h], outline=outline)
    draw.rectangle([x + 1, y + 1, x + 1 + max(0, int((w - 2) * min(1, frac))),
                    y + h - 1], fill=color)


def render_frame(rgb, res, gt, literals, t, fnt, fnt_b, fnt_s):
    canvas = Image.new("RGB", (IMG + W_PANEL, IMG), BG)
    canvas.paste(rgb, (0, 0))
    d = ImageDraw.Draw(canvas)
    x0 = IMG + 12
    y = 8
    d.text((x0, y), f"t = {t / FPS_CACHE:6.1f}s", font=fnt_b, fill=WHITE)
    y += 26

    for arm in ("left", "right"):
        r = res[arm]
        gt_sid = int(gt[f"stage_{arm}"][t])
        ok = r["stage_id"] == gt_sid
        d.text((x0, y), f"{arm.upper()} ARM", font=fnt_b, fill=BLUE)
        d.text((x0 + 110, y), f"H={res['entropy'][arm]:.2f}  "
               f"age={res['stage_age_ratio'][arm]:.2f}", font=fnt_s, fill=GRAY)
        y += 20
        d.text((x0, y), f"pred {r['stage_name']}", font=fnt,
               fill=GREEN if ok else RED)
        y += 17
        d.text((x0, y), f"gt   {stage_name(gt_sid)}", font=fnt, fill=GRAY)
        y += 19
        dist = np.asarray(r["stage_dist"])
        for sid in dist.argsort()[::-1][:3]:
            bar(d, x0, y, 150, 11, dist[sid], BLUE)
            d.text((x0 + 156, y - 1), f"{dist[sid]:.2f} {stage_name(int(sid))[:28]}",
                   font=fnt_s, fill=WHITE)
            y += 15
        gp = float(gt[f"progress_{arm}"][t])
        seg_ok = int(gt[f"seg_{arm}"][t]) >= 0
        bar(d, x0, y, 150, 11, r["progress"], YELLOW)
        if seg_ok:
            tx = x0 + 1 + int(148 * gp)
            d.line([tx, y - 2, tx, y + 13], fill=GREEN, width=2)
        d.text((x0 + 156, y - 1),
               f"prog {r['progress']:.2f}" + (f" (gt {gp:.2f})" if seg_ok else ""),
               font=fnt_s, fill=WHITE)
        y += 16
        q = r.get("grounding_query")
        if q:
            d.text((x0, y), f"-> grounding: \"{q['instruction'][:44]}\"",
                   font=fnt_s, fill=YELLOW)
        y += 20

    d.text((x0, y), "p_sat  (dot = GT satisfied)", font=fnt_b, fill=BLUE)
    y += 20
    led = np.asarray(gt["ledger"][t])
    for i, lit in enumerate(literals):
        if i >= len(res["p_sat"]):
            break
        p = res["p_sat"][i]
        bar(d, x0, y, 100, 10, p, GREEN if p > 0.5 else GRAY)
        if i < len(led) and led[i] > 0.5:
            d.ellipse([x0 + 106, y + 1, x0 + 114, y + 9], fill=GREEN)
        name = f"{lit['predicate']}({lit.get('target', '?')})"
        d.text((x0 + 120, y - 2), f"{p:.2f} {name[:40]}", font=fnt_s, fill=WHITE)
        y += 14
        if y > IMG - 16:
            break
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--fi", type=int, required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--medians", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--speed", type=float, default=2.0, help="x real-time")
    ap.add_argument("--backbone", default="dinov2_vitb14")
    ap.add_argument("--max_frames", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d = cache_dir(args.task, args.fi)
    lab = dict(np.load(os.path.join(d, "stage_labels.npz"), allow_pickle=False))
    depth = np.load(os.path.join(d, "depth.npy"))
    prop = np.load(os.path.join(d, "proprio.npy"))
    n = len(lab["stage_left"])
    if args.max_frames:
        n = min(n, args.max_frames)

    tt = json.load(open(TASK_TARGETS))[args.task]
    literals = literals_from_task_targets(tt)

    backbone = load_backbone(args.backbone, device)
    spec = backbone[1]
    print(f"precomputing DINO tokens for {n} frames ...", flush=True)
    toks = precompute_tokens(os.path.join(d, "frames"), n, backbone, device)

    est = StageEstimator(args.ckpt, args.medians, device=device,
                         grid=spec["grid"], feat=spec["feat"])
    est.set_task(args.task, literals)

    fnt = ImageFont.truetype(FONT, 14)
    fnt_b = ImageFont.truetype(FONT_B, 15)
    fnt_s = ImageFont.truetype(FONT, 12)

    fps_out = FPS_CACHE * args.speed
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{IMG + W_PANEL}x{IMG}", "-r", f"{fps_out:.3f}", "-i", "-",
         "-c:v", "libx264", "-preset", "fast", "-crf", "23",
         "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
         args.out], stdin=subprocess.PIPE)

    correct = np.zeros(2, dtype=np.int64)
    for t in range(n):
        rgb = Image.open(os.path.join(d, "frames", f"f_{t:05d}.jpg")).convert("RGB")
        tok = toks[t:t + 1].to(device).float()
        dep = torch.from_numpy(depth[t:t + 1].astype(np.float32)).to(device)
        pr = torch.from_numpy(prop[t:t + 1].astype(np.float32)).to(device)
        res = est.update(tok, dep, pr)
        for a, arm in enumerate(("left", "right")):
            correct[a] += int(res[arm]["stage_id"] == int(lab[f"stage_{arm}"][t]))
        frame = render_frame(rgb, res, lab, literals, t, fnt, fnt_b, fnt_s)
        enc.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
        if t % 200 == 0:
            print(f"  frame {t}/{n}", flush=True)
    enc.stdin.close()
    enc.wait()
    print(f"done -> {args.out}  live acc L={correct[0] / n:.3f} R={correct[1] / n:.3f}")


if __name__ == "__main__":
    main()

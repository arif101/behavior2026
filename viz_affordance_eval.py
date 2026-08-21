"""Visualize eval-time affordance: run the head over a completed closed-loop eval video.

Input: the tiled eval mp4 (672x448: left column = wrist cams stacked 224^2, right pane 448^2 =
head camera). Crops the head pane, runs DINOv2-S + AffHead on CPU (GPU busy with the live
campaign), overlays the soft-decoded point + confidence per sampled frame.

Green ring = conf > TAU (would inject / write_target); gray ring = below threshold (policy
gets the sentinel). Output: /root/viz_affordance_eval.png (4x5 montage, early -> late).
"""

import sys

import numpy as np

sys.path.insert(0, "/root")

VIDEO = "/root/rate/run_1/videos/turning_on_radio_301_0.mp4"
IN, P, TAU = 518, 37, 0.5
IMNET_M = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_S = np.array([0.229, 0.224, 0.225], np.float32)


def main():
    import av
    import torch
    from PIL import Image, ImageDraw
    from train_affordance import AffHead

    torch.set_num_threads(4)
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval()
    head = AffHead()
    head.load_state_dict(torch.load("/root/aff_out/aff_head.pt", map_location="cpu"))
    head.eval()

    c = av.open(VIDEO)
    s = c.streams.video[0]
    n = s.frames
    picks = sorted({int(n * f / 20) for f in range(1, 21)} - {0})
    frames = {}
    for i, fr in enumerate(c.decode(s)):
        if i in picks:
            frames[i] = fr.to_ndarray(format="rgb24")[:, 224:, :]   # head pane 448^2
        if i > picks[-1]:
            break

    tiles = []
    with torch.no_grad():
        for i in picks:
            if i not in frames:
                continue
            pane = frames[i]
            H = pane.shape[0]
            x = np.asarray(Image.fromarray(pane).resize((IN, IN)), np.float32) / 255.0
            x = (x - IMNET_M) / IMNET_S
            t = torch.from_numpy(x.transpose(2, 0, 1))[None]
            feats = dino.forward_features(t)["x_norm_patchtokens"]
            heat, off, dz, conf = head(feats)
            prob2 = heat.softmax(-1).reshape(1, P, P)
            idx = int(heat.argmax(-1)[0])
            pu, pv = idx % P, idx // P
            us = vs = ws = 0.0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    qu = min(max(pu + dx, 0), P - 1)
                    qv = min(max(pv + dy, 0), P - 1)
                    w = float(prob2[0, qv, qu])
                    fr_ = torch.sigmoid(off[0, :, qv, qu])
                    us += (qu + float(fr_[0])) * 14 * w
                    vs += (qv + float(fr_[1])) * 14 * w
                    ws += w
            u = us / ws * H / IN
            v = vs / ws * H / IN
            cf = float(torch.sigmoid(conf)[0])

            img = Image.fromarray(pane).convert("RGB")
            dr = ImageDraw.Draw(img)
            col = (0, 255, 60) if cf > TAU else (150, 150, 150)
            dr.ellipse([u - 7, v - 7, u + 7, v + 7], outline=col, width=3)
            dr.line([u - 14, v, u + 14, v], fill=col, width=1)
            dr.line([u, v - 14, u, v + 14], fill=col, width=1)
            dr.rectangle([0, 0, H - 1, H - 1], outline=col, width=3)
            dr.text((6, 6), f"f{i} conf={cf:.2f}{'' if cf > TAU else ' (no inject)'}",
                    fill=(255, 255, 0))
            tiles.append(np.asarray(img))

    rows = [np.concatenate(tiles[r * 5:(r + 1) * 5], axis=1) for r in range(4) if len(tiles[r * 5:(r + 1) * 5]) == 5]
    Image.fromarray(np.concatenate(rows, axis=0)).save("/root/viz_affordance_eval.png")
    print(f"WROTE /root/viz_affordance_eval.png ({len(tiles)} tiles)")


if __name__ == "__main__":
    main()

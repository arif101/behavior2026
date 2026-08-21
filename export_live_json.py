"""Export live-map snapshots + affordance-heatmap evolution as one compact JSON for the
interactive debug artifact (separate L0/L1/L2 viewers + time scrubber).

Map levels: from /root/map_live/snap_*.npz (the in-flight legal episode).
Heatmap strip: affordance 37x37 heatmaps + confidences computed over the COMPLETED legal run_2
video's head pane at matching steps (the in-flight episode's mp4 has no index until it closes) —
labeled as such in the artifact.

Budget: <= 14 timesteps, L0 <= 3000 pts, L1 <= 7000 pts, L2 all; coords rounded to cm,
colors packed as one int per point.
"""

import glob
import json

import numpy as np

MAX_T = 14
CAPS = {"l0": 3000, "l1": 7000, "l2_left": 4000, "l2_right": 4000}


def pack(pts, rgb, cap):
    if len(pts) > cap:
        sel = np.random.default_rng(0).choice(len(pts), cap, replace=False)
        pts, rgb = pts[sel], rgb[sel]
    p = np.round(np.asarray(pts, np.float64), 2)
    c = np.asarray(rgb, np.int64)
    packed = (c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]
    return {"p": p.ravel().tolist(), "c": packed.tolist()}


def main():
    snaps = sorted(glob.glob("/root/map_live/snap_*.npz"))
    idx = np.linspace(0, len(snaps) - 1, min(MAX_T, len(snaps))).astype(int)
    out = {"timesteps": [], "meta": {"n_snapshots": len(snaps)}}
    for i in idx:
        z = np.load(snaps[i])
        d = {k: z[k] for k in z.files}
        entry = {"step": int(d["step"]), "odom": np.round(d["odom"][:3], 3).tolist()}
        for lv in ("l0", "l1", "l2_left", "l2_right"):
            if f"{lv}_pts" in d and len(d[f"{lv}_pts"]):
                entry[lv] = pack(d[f"{lv}_pts"].astype(np.float32),
                                 d[f"{lv}_rgb"], CAPS[lv])
            else:
                entry[lv] = {"p": [], "c": []}
        if "target" in d:
            entry["target"] = np.round(np.asarray(d["target"], np.float64), 3).tolist()
        out["timesteps"].append(entry)

    # ---- affordance heatmap evolution (completed run_2 video, head pane) -------------------
    try:
        import sys
        sys.path.insert(0, "/root")
        import av
        import torch
        from PIL import Image
        from train_affordance import AffHead

        torch.set_num_threads(2)
        dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval()
        head = AffHead()
        head.load_state_dict(torch.load("/root/aff_out/aff_head.pt", map_location="cpu"))
        head.eval()
        IMNET_M = np.array([0.485, 0.456, 0.406], np.float32)
        IMNET_S = np.array([0.229, 0.224, 0.225], np.float32)

        c = av.open("/root/rate_legal/run_2/videos/turning_on_radio_301_0.mp4")
        s_ = c.streams.video[0]
        n = s_.frames
        picks = sorted({int(n * k / 12) for k in range(1, 13)} - {0})
        heat_ts = []
        with torch.no_grad():
            for i, fr in enumerate(c.decode(s_)):
                if i in picks:
                    pane = fr.to_ndarray(format="rgb24")[:, 224:, :]
                    x = np.asarray(Image.fromarray(pane).resize((518, 518)), np.float32) / 255.0
                    x = (x - IMNET_M) / IMNET_S
                    t = torch.from_numpy(x.transpose(2, 0, 1))[None]
                    feats = dino.forward_features(t)["x_norm_patchtokens"]
                    heat, off, dz, conf = head(feats)
                    prob = heat.softmax(-1).reshape(37, 37).numpy()
                    prob8 = (prob / prob.max() * 255).astype(np.uint8)
                    thumb = np.asarray(Image.fromarray(pane).resize((112, 112)), np.uint8)
                    tp = ((thumb[:, :, 0].astype(int) << 16) |
                          (thumb[:, :, 1].astype(int) << 8) | thumb[:, :, 2].astype(int))
                    heat_ts.append({
                        "frame": i,
                        "conf": round(float(torch.sigmoid(conf)[0]), 3),
                        "heat": prob8.ravel().tolist(),
                        "thumb": tp.ravel().tolist(),
                    })
                if i > picks[-1]:
                    break
        out["heatmaps"] = {"source": "legal run 2 (completed episode)", "frames": heat_ts}
    except Exception as e:  # noqa: BLE001
        out["heatmaps"] = {"error": str(e)[:200]}

    with open("/root/map_debug_data.json", "w") as f:
        json.dump(out, f, separators=(",", ":"))
    import os
    print(f"WROTE /root/map_debug_data.json ({os.path.getsize('/root/map_debug_data.json')/1e6:.1f} MB, "
          f"{len(out['timesteps'])} timesteps, {len(out.get('heatmaps', {}).get('frames', []))} heat frames)")


if __name__ == "__main__":
    main()

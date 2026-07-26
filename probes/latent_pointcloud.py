"""Back-project the policy's per-patch latents into 3D and render them as a coloured point cloud.

Question it answers: does the policy's visual representation LOCALISE THE TARGET in metric space?
The policy drove to a wall-mounted TV instead of the radio, so the sharp version is: does the
high-similarity region sit on the radio, or on the TV?

This is NOT gaussian splatting, deliberately. 3DGS buys novel-view synthesis and a
differentiably-optimised field; we need neither, and its per-scene fitting cost is incompatible
with a control loop. Back-projection through the depth and camera poses we already capture gives
the same insight for essentially no cost. Anisotropic gaussians would be decoration.

HOW THE PATCH -> 3D MAPPING WORKS
    embed_prefix concatenates image tokens FIRST (each image a fixed grid, in obs.images order),
    then text. So prefix positions partition cleanly:
        [0, P)      -> image_0 patches, reshape to (g, g)
        [P, 2P)     -> image_1
        [2P, 3P)    -> image_2
        [3P, ...)   -> text tokens
    Each patch centre maps to a pixel, the pixel + depth un-projects to a camera-frame ray point,
    and the camera pose lifts it to world/base frame.

COLOURING
    cosine similarity between each patch's latent and the mean TEXT-token latent (the prompt
    names the target). High similarity should concentrate on the referred object if the
    representation grounds language spatially.

Ground truth is rendered alongside: the captured target_world position is marked, so the picture
is falsifiable rather than decorative.
"""

import argparse
import glob
import json
import os

import numpy as np


def _quat_to_rot(q):
    x, y, z, w = [float(v) for v in np.asarray(q).reshape(4)]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def unproject(depth, K, cam_pos, cam_quat, stride):
    """Pixel grid + depth -> world points, subsampled to the patch grid."""
    H, W = depth.shape
    ys = (np.arange(0, H, stride) + stride // 2).clip(0, H - 1)
    xs = (np.arange(0, W, stride) + stride // 2).clip(0, W - 1)
    uu, vv = np.meshgrid(xs, ys)
    z = depth[vv, uu].astype(np.float64)
    fx, fy, cx, cy = K
    x = (uu - cx) / fx * z
    y = (vv - cy) / fy * z
    # OmniGibson cameras look down -Z with +Y up (USD convention)
    cam_pts = np.stack([x, -y, -z], axis=-1)
    R = _quat_to_rot(cam_quat)
    world = cam_pts.reshape(-1, 3) @ R.T + np.asarray(cam_pos).reshape(1, 3)
    return world, z.reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", default="/root/capture")
    ap.add_argument("--features", required=True, help="npz with per-patch latents for these frames")
    ap.add_argument("--out", default="/root/latent_cloud.npz")
    ap.add_argument("--max-depth", type=float, default=6.0, help="drop points beyond this (metres)")
    args = ap.parse_args()

    frames = sorted(glob.glob(os.path.join(args.capture_dir, "frame_*.npz")))
    if not frames:
        raise SystemExit(f"no captured frames in {args.capture_dir}")
    feats = np.load(args.features, allow_pickle=True)
    print(f"frames: {len(frames)}  feature keys: {list(feats)[:6]}", flush=True)

    all_pts, all_sim, all_rgb = [], [], []
    target = None
    for fi, fp in enumerate(frames):
        d = np.load(fp, allow_pickle=True)
        cams = json.load(open(fp.replace("frame_", "cams_").replace(".npz", ".json")))
        if "target0_world" in d:
            target = np.asarray(d["target0_world"]).reshape(3)

        key = f"frame{fi}"
        if key not in feats:
            continue
        patch = feats[key]                      # (n_images, g, g, D)
        text = feats.get(f"text{fi}")           # (D,) mean text-token latent
        n_img, g, _, D = patch.shape

        cam_names = [k.split("::")[0] for k in d.files if k.endswith("::rgb")]
        for ci, cam in enumerate(cam_names[:n_img]):
            dep = np.asarray(d.get(f"{cam}::depth_linear"))
            rgb = np.asarray(d.get(f"{cam}::rgb"))
            if dep is None or dep.ndim != 2:
                continue
            cp = cams.get(cam)
            if cp is None:
                continue
            H, W = dep.shape
            f_px = (cp["focal"] / max(cp["h_aperture"], 1e-6)) * W if cp["h_aperture"] else W * 1.2
            K = (f_px, f_px, W / 2.0, H / 2.0)
            pts, z = unproject(dep, K, cp["pos"], cp["quat"], stride=max(1, H // g))
            v = patch[ci].reshape(-1, D)
            n = min(len(pts), len(v))
            pts, z, v = pts[:n], z[:n], v[:n]
            if text is not None:
                a = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
                b = text / (np.linalg.norm(text) + 1e-8)
                sim = a @ b
            else:
                sim = np.linalg.norm(v, axis=1)
            keep = np.isfinite(z) & (z > 0.05) & (z < args.max_depth)
            all_pts.append(pts[keep])
            all_sim.append(sim[keep])
            if rgb is not None and rgb.ndim == 3:
                s = max(1, H // g)
                small = rgb[s // 2 :: s, s // 2 :: s][..., :3].reshape(-1, 3)[:n][keep]
                all_rgb.append(small)

    P = np.concatenate(all_pts) if all_pts else np.zeros((0, 3))
    S = np.concatenate(all_sim) if all_sim else np.zeros((0,))
    C = np.concatenate(all_rgb) if all_rgb else np.zeros((0, 3))
    print(f"points: {len(P):,}   sim range [{S.min():.3f}, {S.max():.3f}]" if len(P) else "no points")
    if target is not None and len(P):
        dist = np.linalg.norm(P - target[None], axis=1)
        near = dist < 0.35
        if near.any():
            print(f"TARGET LOCALISATION: mean sim within 35cm of target = {S[near].mean():.4f} "
                  f"vs {S[~near].mean():.4f} elsewhere  (n_near={int(near.sum())})")
            print(f"  -> ratio {S[near].mean() / max(1e-9, S[~near].mean()):.3f}  "
                  f"(>1 means the representation IS concentrating on the target)")
        else:
            print("target not visible in captured views (no points within 35cm)")
    np.savez_compressed(args.out, points=P, sim=S, rgb=C,
                        target=(target if target is not None else np.zeros(3)))
    print("wrote", args.out)


if __name__ == "__main__":
    main()

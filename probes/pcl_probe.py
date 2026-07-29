"""THE latent point-cloud probe, finally run for real: back-project the policy's per-patch
prefix latents into metric 3D and ask whether language-similarity mass sits ON the radio.

Question it answers (unchanged since Jul 26): does the policy's visual representation LOCALISE
THE TARGET in metric space, or is high similarity smeared over distractors (the TV it drove to
in G3)? This is the SEMANTIC-layer diagnostic; today's fall is a sensorimotor-layer failure --
the probe tells us whether the semantic layer is solid enough to build on.

Inputs: /root/pcl_frames/frame_XXXX.npz captured live at the policy server during an
oracle-conditioned rollout (head 720x720 rgb + depth_linear, wrists 480, cam_rel_poses = camera
extrinsics RELATIVE TO BASE, proprio 61, target_points).

Pipeline per frame:
  1. rebuild the exact serving input (same [...,:3] slice, same client resize_with_pad, same
     prompt) and push it through policy._input_transform -> Observation.from_dict -- i.e. the
     VERIFIED-faithful serving transform chain, not a reimplementation;
  2. prefix forward pass exactly as prefix_probe_grouped.collect() (known-working; prefix_out is
     independent of x_t/t, verified bit-identical earlier);
  3. similarity = cosine(head-image patch outputs [0:256], mean over TASK-TEXT token outputs).
     TASK tokens are located by sentencepiece-encoding the task string and searching the
     tokenized prompt -- fixing the Jul-26 bug where similarity targeted the mean of ALL text
     tokens, 91 of 96 of which are proprio digits;
  4. back-project each 16x16 patch centre through depth + head-camera intrinsics + cam_rel_poses
     into the BASE frame (USD convention: camera looks down -Z, +Y up);
  5. radio ground truth in the same frame for free: obj_base = target_points[left] + ee_left,
     with ee_left = proprio[17:20] (target_points is DEFINED as obj_base - ee_base).

Outputs:
  * the falsifiable stat: mean similarity within 0.35 m of the radio vs elsewhere, + ratio;
  * /root/pcl_overlay.png: rgb | similarity-overlay pairs for 6 frames;
  * /root/pcl_topdown.png: top-down scatter of all back-projected patches coloured by
    similarity, radio starred;
  * /root/pcl_cloud.npz: points + similarities for any later 3D rendering.
"""

from __future__ import annotations

import glob
import json

import numpy as np

HEAD_RGB = "robot_r1::robot_r1:zed_link:Camera:0::rgb"
HEAD_DEPTH = "robot_r1::robot_r1:zed_link:Camera:0::depth_linear"
LEFT_RGB = "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb"
RIGHT_RGB = "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb"
PROPRIO = "robot_r1::proprio"
CAMPOSES = "robot_r1::cam_rel_poses"

G = 16          # SigLIP so400m/14 @ 224 -> 16x16 patches
PATCH = 14
HEAD_RES = 720
NEAR_M = 0.35   # "on the radio" radius for the falsifiable stat


def quat_to_rot(q):
    x, y, z, w = [float(v) for v in q]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def load_intrinsics():
    """fx, fy, cx, cy at 720x720. Calibrated file if present, else square-FOV fallback."""
    try:
        d = json.load(open("/root/zed_intrinsics_calibrated.json"))
        sc = HEAD_RES / float(d.get("width", HEAD_RES))
        return d["fx"] * sc, d["fy"] * sc, d["cx"] * sc, d["cy"] * sc, "calibrated"
    except Exception:
        # OmniGibson default viewer cameras are ~60 deg vertical FOV on square sensors.
        f = (HEAD_RES / 2) / np.tan(np.radians(60) / 2)
        return f, f, HEAD_RES / 2, HEAD_RES / 2, "fov-60-fallback"


def main():
    import dataclasses  # noqa: F401  (kept for parity with sibling probes)

    import jax
    import jax.numpy as jnp
    import sentencepiece as spm

    from openpi.configs.tasks import TASK_REGISTRY
    from openpi.models import pi0 as _pi0
    import openpi.models.model as _model
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import download
    from openpi.training import config as _config
    from openpi_client.image_tools import resize_with_pad

    frames = sorted(glob.glob("/root/pcl_frames/frame_*.npz"))
    print(f"frames captured: {len(frames)}")
    if not frames:
        raise SystemExit("no frames -- did the capture rollout run?")

    task_prompt = TASK_REGISTRY["b1k"]["turning_on_radio"]
    print(f"task prompt: {task_prompt!r}")

    cfg = _config.get_config("pi05_radio_gate")
    policy = _policy_config.create_trained_policy(cfg, "/root/ckpt", default_prompt=task_prompt)
    model = policy._model

    tok_path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model")
    sp = spm.SentencePieceProcessor(model_file=str(tok_path))
    task_ids = sp.encode(task_prompt)
    print(f"task ids ({len(task_ids)}): {task_ids}")

    fx, fy, cx, cy, intr_src = load_intrinsics()
    print(f"intrinsics [{intr_src}]: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

    rng = jax.random.key(0)
    all_pts, all_sims, per_frame = [], [], []

    for fp in frames:
        z = np.load(fp, allow_pickle=True)
        if HEAD_RGB not in z.files or HEAD_DEPTH not in z.files:
            continue
        head = np.asarray(z[HEAD_RGB])[..., :3]
        depth = np.asarray(z[HEAD_DEPTH]).astype(np.float64)
        prop = np.asarray(z[PROPRIO]).astype(np.float32)
        cams = np.asarray(z[CAMPOSES]).astype(np.float64)
        tp = np.asarray(z["target_points"]).astype(np.float32) if "target_points" in z.files else None

        raw = {
            "observation/image_0": resize_with_pad(head, 224, 224),
            "observation/image_1": resize_with_pad(np.asarray(z[LEFT_RGB])[..., :3], 224, 224),
            "observation/image_2": resize_with_pad(np.asarray(z[RIGHT_RGB])[..., :3], 224, 224),
            "observation/state": prop,
            "prompt": task_prompt,
        }
        if tp is not None:
            raw["target_points"] = tp
            raw["target_points_mask"] = np.array([True, True])

        inputs = policy._input_transform(jax.tree.map(lambda x: x, raw))
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        obs = _model.Observation.from_dict(inputs)

        # prefix forward, verbatim from prefix_probe_grouped.collect()
        prefix_tokens, prefix_mask, prefix_ar = model.embed_prefix(obs)
        x_t = jax.random.normal(rng, (1, model.action_horizon, model.action_dim))
        t = jnp.zeros((1,), dtype=jnp.float32)
        suffix_tokens, suffix_mask, suffix_ar, adarms = model.embed_suffix(obs, x_t, t)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar, suffix_ar], axis=0)
        attn = _pi0.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, _), _ = model.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn, positions=positions,
            adarms_cond=[None, adarms],
        )
        po = np.asarray(prefix_out[0], dtype=np.float32)      # (n_prefix, width)

        # locate the TASK tokens (not the proprio digits) in the tokenized prompt
        tok = np.asarray(obs.tokenized_prompt[0]).tolist()
        span = None
        for i in range(len(tok) - len(task_ids) + 1):
            if tok[i:i + len(task_ids)] == task_ids:
                span = (i, i + len(task_ids))
                break
        n_img = 3 * G * G
        if span is None:
            tmask = np.asarray(obs.tokenized_prompt_mask[0]).astype(bool)
            n_txt = int(tmask.sum())
            span = (max(0, n_txt - len(task_ids)), n_txt)
            span_src = "fallback-tail"
        else:
            span_src = "matched"
        text_vec = po[n_img + span[0]: n_img + span[1]].mean(0)

        head_patches = po[0:G * G]
        sims = (head_patches @ text_vec) / (
            np.linalg.norm(head_patches, axis=1) * np.linalg.norm(text_vec) + 1e-8
        )
        sim_grid = sims.reshape(G, G)

        # back-project patch centres through depth into the BASE frame
        cam_pos, cam_quat = cams[0:3], cams[3:7]              # head is the first 7-tuple
        R = quat_to_rot(cam_quat)
        pts, keep = [], []
        scale = HEAD_RES / 224.0
        for r in range(G):
            for c in range(G):
                u = int((c * PATCH + PATCH // 2) * scale)
                v = int((r * PATCH + PATCH // 2) * scale)
                d = float(depth[v, u])
                if not np.isfinite(d) or d <= 0.05 or d > 8.0:
                    keep.append(False)
                    continue
                xc = (u - cx) / fx * d
                yc = (v - cy) / fy * d
                cam_pt = np.array([xc, -yc, -d])              # USD: -Z forward, +Y up
                pts.append(R @ cam_pt + cam_pos)
                keep.append(True)
        pts = np.array(pts) if pts else np.zeros((0, 3))
        keep = np.array(keep)

        radio = None
        if tp is not None:
            radio = tp[0] + prop[17:20]                        # obj = target + ee_left, base frame

        all_pts.append(pts)
        all_sims.append(sims[keep])
        per_frame.append({
            "file": fp, "head": head, "sim_grid": sim_grid, "pts": pts,
            "sims": sims[keep], "radio": radio, "cam_z": float(cam_pos[2]),
            "span_src": span_src,
        })

    P = np.concatenate(all_pts) if all_pts else np.zeros((0, 3))
    S = np.concatenate(all_sims) if all_sims else np.zeros((0,))
    print(f"\nback-projected points: {len(P)}   head-cam z (sanity, expect ~1.2-1.6 m): "
          f"{per_frame[0]['cam_z']:.2f}   token-span: {per_frame[0]['span_src']}")

    radios = [f["radio"] for f in per_frame if f["radio"] is not None]
    if radios and len(P):
        radio_mean = np.mean(radios, axis=0)
        d = np.linalg.norm(P - radio_mean, axis=1)
        near = d < NEAR_M
        print("\n=== FALSIFIABLE STAT ===")
        print(f"radio (base frame, mean over frames): {np.round(radio_mean, 3)}")
        print(f"points within {NEAR_M} m: {int(near.sum())} / {len(P)}")
        if near.any() and (~near).any():
            ratio = S[near].mean() / max(S[~near].mean(), 1e-9)
            print(f"mean sim NEAR radio : {S[near].mean():+.4f}")
            print(f"mean sim ELSEWHERE  : {S[~near].mean():+.4f}")
            print(f"RATIO               : {ratio:.3f}   (>1 = similarity mass sits ON the radio)")
        else:
            print("radio not covered by any back-projected patch in these frames")

    # ---- visuals ------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    show = per_frame[:: max(1, len(per_frame) // 6)][:6]
    fig, axes = plt.subplots(2, len(show), figsize=(3.2 * len(show), 6.6))
    for j, f in enumerate(show):
        axes[0, j].imshow(f["head"]); axes[0, j].axis("off")
        axes[0, j].set_title(f["file"].split("_")[-1].split(".")[0], fontsize=8)
        axes[1, j].imshow(f["head"])
        g = np.kron(f["sim_grid"], np.ones((HEAD_RES // G, HEAD_RES // G)))
        axes[1, j].imshow(g, alpha=0.55, cmap="jet")
        axes[1, j].axis("off")
    axes[0, 0].set_ylabel("rgb"); axes[1, 0].set_ylabel("similarity")
    plt.suptitle("head camera | task-token similarity (16x16 patch grid)")
    plt.tight_layout()
    plt.savefig("/root/pcl_overlay.png", dpi=110)

    if len(P):
        plt.figure(figsize=(7, 7))
        o = np.argsort(S)
        plt.scatter(P[o, 0], P[o, 1], c=S[o], s=7, cmap="jet")
        plt.colorbar(label="task-token similarity")
        if radios:
            plt.scatter([radio_mean[0]], [radio_mean[1]], marker="*", s=420, c="white",
                        edgecolors="black", linewidths=1.5, label="radio (oracle)")
            plt.legend()
        plt.xlabel("x base (m)"); plt.ylabel("y base (m)")
        plt.title("latent point cloud, top-down (base frame)")
        plt.axis("equal"); plt.tight_layout()
        plt.savefig("/root/pcl_topdown.png", dpi=110)

    np.savez_compressed("/root/pcl_cloud.npz", points=P, sims=S,
                        radio=radio_mean if radios else np.zeros(3))
    print("\nwrote /root/pcl_overlay.png /root/pcl_topdown.png /root/pcl_cloud.npz")


if __name__ == "__main__":
    main()

"""Probe harness stage 1: extract per-frame intermediate activations from a pi0.5-class
openpi checkpoint over dataset frames, plus cheap labels available without sim-state decoding.

Runs on the training box inside the openpi venv (uv run python probes/extract_features.py ...).
Dumps: features per selected layer (N, D) + labels (N, K) -> npz shards under --out.

v1 labels (no sim decode needed):
  - action_t (23-D executed action at frame)
  - future_ee_delta (mean of next H actions, arm joints)  [action-relevance target]
  - gripper_cmd (left/right gripper action dims)
  - phase_frac (frame_index / episode_length)             [progress proxy]
Sim-state labels (object 6-DoF pose etc.) are added by decode_state_labels.py (stage 1b).
"""

import argparse
import dataclasses
import numpy as np
import jax
import jax.numpy as jnp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="pi05_b1k_ours_lora")
    ap.add_argument("--ckpt", required=True, help="checkpoint step dir with params/")
    ap.add_argument("--n-frames", type=int, default=8000)
    ap.add_argument("--stride", type=int, default=25, help="sample every Nth frame per episode")
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--out", default="/workspace/probe_data")
    ap.add_argument("--dataset-root", default=None, help="override root (e.g. a labelled slice)")
    ap.add_argument("--episodes-json", default=None, help="JSON list of episode indices")
    args = ap.parse_args()

    import os
    os.makedirs(args.out, exist_ok=True)

    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config

    cfg = _config.get_config(args.config_name)
    policy = _policy_config.create_trained_policy(cfg, args.ckpt)
    model = policy._model  # noqa: SLF001 - probe harness reaches into internals by design

    # Dataset: reuse the training loader path (episode_filter etc.)
    from openpi.training import data_loader as _dl
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    if args.dataset_root:
        import dataclasses as _dc, json as _json
        _eps = _json.load(open(args.episodes_json)) if args.episodes_json else None
        data_cfg = _dc.replace(data_cfg, dataset_root=args.dataset_root,
                               dataset_kwargs={"tolerance_s": 5e-4, **({"episodes": _eps} if _eps else {})})
        print(f"slice mode: {args.dataset_root} ({len(_eps) if _eps else 'all'} eps)", flush=True)
    dataset = _dl.create_b1k_dataset(data_cfg, cfg.model.action_horizon)
    # Training-side transform stack (mirrors create_b1k_dataloader + model transforms):
    # raw items feed labels; transformed items feed the model.
    transformed_ds = _dl.TransformedDataset(
        dataset,
        [
            *data_cfg.repack_transforms.inputs,
            *data_cfg.data_transforms.inputs,
            *data_cfg.model_transforms.inputs,
        ],
    )

    rng = np.random.default_rng(0)
    n = len(dataset)
    idxs = np.arange(0, n, args.stride)
    rng.shuffle(idxs)
    idxs = np.sort(idxs[: args.n_frames])  # random SUBSET, sequential ORDER (video-decode locality)

    feats_by_layer: dict[str, list] = {}
    labels: dict[str, list] = {k: [] for k in ["action_t", "future_ee_delta", "gripper_cmd", "phase_frac", "ep_idx", "frame_idx"]}

    from openpi.models import model as _model_mod
    import einops  # noqa: F401

    def make_attn_mask_local(mask, ar_mask):
        from openpi.models.pi0 import make_attn_mask
        return make_attn_mask(mask, ar_mask)

    def capture_prefix_kv(obs: "_model_mod.Observation") -> dict[str, np.ndarray]:
        """Mirror sample_actions' prefix stage; return per-layer mean-pooled V features.

        The KV cache is what the action expert attends into — the most action-relevant
        per-layer feature available without model surgery.
        """
        obs = _model_mod.preprocess_observation(None, obs, train=False)
        prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs)
        attn_mask = make_attn_mask_local(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=attn_mask, positions=positions)
        # kv_cache = (K, V), each stacked over layers: (L, b, pos, kv_heads, head_dim)
        _, V = kv_cache
        V = np.asarray(V)
        L = V.shape[0]
        V = V.reshape(L, V.shape[1], V.shape[2], -1)          # (L, b, pos, d)
        m = np.asarray(prefix_mask)[None, ..., None]          # (1, b, pos, 1)
        pooled = (V * m).sum(2) / np.maximum(m.sum(2), 1.0)   # (L, b, d)
        return {f"layer{li:02d}": pooled[li].astype(np.float16) for li in range(L)}

    # ---- labels from the arrow table: zero video decode ----
    base = dataset
    while not hasattr(base, "hf_dataset") and hasattr(base, "_dataset"):
        base = base._dataset
    arrow = base.hf_dataset
    print(f"sampling {len(idxs)} frames from {n}; labels via arrow table")
    for i in idxs:
        row = arrow[int(i)]
        a0 = np.asarray(row["action"], dtype=np.float32)
        hi = min(int(i) + args.horizon, len(arrow))
        fut = np.asarray(arrow[int(i): hi]["action"], dtype=np.float32)
        same_ep = np.asarray(arrow[int(i): hi]["episode_index"]) == row["episode_index"]
        fut = fut[same_ep] if same_ep.any() else a0[None]
        labels["action_t"].append(a0)
        labels["future_ee_delta"].append(fut[:, 7:14].mean(0) - a0[7:14])
        labels["gripper_cmd"].append(a0[[14, 22]])
        labels["phase_frac"].append(0.0)  # filled in train_probes from (ep_idx, frame_idx)
        labels["ep_idx"].append(int(row["episode_index"]))
        labels["frame_idx"].append(int(row["frame_index"]))

    # ---- features: single decode per frame, batched forwards ----
    BS = 16
    for b0 in range(0, len(idxs), BS):
        chunk = idxs[b0: b0 + BS]
        items = []
        for i in chunk:
            t = transformed_ds[int(i)]
            items.append({k: v for k, v in t.items() if k != "actions"})
        batch = jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]), *items)
        obs = _model_mod.Observation.from_dict(batch)
        inter = capture_prefix_kv(obs)  # per-layer (b, d)
        for lname, arr in inter.items():
            feats_by_layer.setdefault(lname, []).extend(list(arr))
        if (b0 // BS) % 4 == 0:
            print(f"{b0 + len(chunk)}/{len(idxs)}", flush=True)

    for lname, rows in feats_by_layer.items():
        np.savez_compressed(f"{args.out}/feat_{lname.replace('/', '_')}.npz", X=np.stack(rows))
    np.savez_compressed(f"{args.out}/labels.npz", **{k: np.asarray(v) for k, v in labels.items()})
    print("EXTRACT_DONE", len(idxs), "frames,", len(feats_by_layer), "layers")


if __name__ == "__main__":
    main()

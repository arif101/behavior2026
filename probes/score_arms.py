"""Score a Phase-A checkpoint on a FIXED, seeded frame set, split into commit vs ordinary frames.

Why this exists: the two A/B arms' *training* losses are not comparable. The weighted arm's
batches are ~24% gripper-commit frames, the control's ~3% -- two different exams, not two scores
on one exam. This scores any checkpoint on one common exam so the numbers mean the same thing.

It also tests the hypothesis directly rather than by proxy. The claim behind contact
oversampling is that uniform sampling starves commit frames of gradient budget. The sharp
prediction is therefore:

    weighted arm should beat the control ON COMMIT FRAMES,
    possibly while giving up a little ON ORDINARY FRAMES (the budget came from somewhere).

Reporting both halves separately makes that trade visible; a single average would hide it.

Determinism matters: flow-matching loss samples a random timestep and noise, so the same
checkpoint scored twice with different RNG differs. Both the frame selection and the per-batch
RNG are seeded off --seed ONLY, never off the checkpoint, so every arm sees identical frames
AND identical noise. Comparing arms scored with different seeds is meaningless.

Usage (needs a free GPU; ~14GB, inference only -- no optimizer state):
    python probes/score_arms.py --ckpt <step_dir> --n 2048 --out scores_arm1.json
"""

import argparse
import json

import numpy as np


def _resolve_hf(dataset):
    """Walk the TransformedDataset wrapper chain down to the LeRobotDataset's hf_dataset."""
    node = dataset
    for _ in range(6):
        hf = getattr(node, "hf_dataset", None)
        if hf is not None:
            return hf
        nxt = getattr(node, "_dataset", None) or getattr(node, "dataset", None)
        if nxt is None or nxt is node:
            break
        node = nxt
    raise RuntimeError("could not reach hf_dataset from the transformed dataset")


class _FixedSampler:
    """Yields a fixed, pre-computed index list (so every arm sees the identical frames)."""

    def __init__(self, indices):
        self._idx = list(int(i) for i in indices)

    def __len__(self):
        return len(self._idx)

    def __iter__(self):
        return iter(self._idx)


def _score(model, loader, seed):
    import jax

    losses = []
    for i, (obs, actions) in enumerate(loader):
        # RNG keyed on batch INDEX, not on time or checkpoint -> identical noise across arms.
        rng = jax.random.key(seed + i)
        loss = model.compute_loss(rng, obs, actions, train=False)
        losses.append(float(np.asarray(jax.device_get(loss)).mean()))
    return float(np.mean(losses)) if losses else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="pi05_phaseA_point")
    ap.add_argument("--ckpt", required=True, help="checkpoint step dir containing params/")
    ap.add_argument("--n", type=int, default=2048, help="frames per split (commit / ordinary)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0, help="MUST match across arms")
    ap.add_argument("--out", default="/root/scores.json")
    args = ap.parse_args()

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config
    from openpi.training import data_loader as dl

    cfg = _config.get_config(args.config_name)
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    dataset = dl.transform_dataset(
        dl.create_b1k_dataset(data_cfg, cfg.model.action_horizon), data_cfg
    )

    w = _resolve_hf(dataset).data.column("sample_weight").to_numpy(zero_copy_only=False)
    w = np.asarray(w, dtype=np.float64)
    commit_pool = np.flatnonzero(w > 1.0)
    ordinary_pool = np.flatnonzero(w <= 1.0)
    print(f"pools: commit={len(commit_pool)} ordinary={len(ordinary_pool)}", flush=True)

    # Seeded off --seed alone => identical frame sets for every arm.
    rng = np.random.default_rng(args.seed)
    n = min(args.n, len(commit_pool), len(ordinary_pool))
    splits = {
        "commit": rng.choice(commit_pool, n, replace=False),
        "ordinary": rng.choice(ordinary_pool, n, replace=False),
    }

    policy = _policy_config.create_trained_policy(cfg, args.ckpt)
    model = policy._model  # noqa: SLF001 - scoring harness reaches into internals by design

    results = {"ckpt": args.ckpt, "n_per_split": int(n), "seed": args.seed}
    for name, idx in splits.items():
        loader = dl.DataLoaderImpl(
            data_cfg,
            dl.TorchDataLoader(
                dataset,
                local_batch_size=args.batch_size,
                sampler=_FixedSampler(np.sort(idx)),
                num_batches=max(1, n // args.batch_size),
                num_workers=4,
                seed=args.seed,
            ),
        )
        results[name] = _score(model, loader, args.seed)
        print(f"{name}: {results[name]:.5f}", flush=True)

    # The headline number: negative means the checkpoint is relatively better on contact.
    results["commit_minus_ordinary"] = results["commit"] - results["ordinary"]
    json.dump(results, open(args.out, "w"), indent=1)
    print(json.dumps(results, indent=1), flush=True)


if __name__ == "__main__":
    main()

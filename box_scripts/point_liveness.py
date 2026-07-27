"""Is the AdaLN point-conditioning channel ALIVE in a checkpoint?

Distinct from "is it useful". Usefulness means higher task success, which is unmeasurable while
closed-loop success is 0. Liveness -- does the 3D point change what the policy does at all --
is measurable today, and it is the thing at risk in Phase A: only 2.67% of training frames carry
a real point (120 episodes, the same ones G3 used), and the AdaLN weights restart from zero-init.
If the channel comes back dead, nothing about point conditioning should be read into this run.

Three conditions on the SAME observation with the SAME sampling noise:
    A  real point,       mask=True
    B  displaced point,  mask=True   (+20cm in x -- a wrong but plausible target)
    C  zeros,            mask=False  (the unconditioned path)

Reading the result:
    |A-C| ~ 0                -> DEAD. The point is ignored; zero-init never moved.
    |A-C| > 0, |A-B| ~ 0     -> PRESENT BUT NOT SPATIALLY SELECTIVE. The policy reacts to being
                                given a point at all, but not to WHERE it is -- it learned the
                                mask as a mode switch, not the coordinates. This is the sneaky
                                failure, and averaging A vs C alone would call it a success.
    both > 0                 -> LIVE and spatially selective.

Noise is fixed per sample across all three conditions. Without that, flow-matching sampling
variance alone produces differences and every checkpoint looks "alive".

Needs a free GPU (~14GB, inference only). Uses TRAINING frames, since only they carry real
points -- this is a mechanism test, not a generalization test.
"""

import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="pi05_phaseA_point")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=64, help="labelled frames to test")
    ap.add_argument("--displace", type=float, default=0.20, help="metres to shift the point for B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/root/point_liveness.json")
    ap.add_argument("--dataset-root", default=None,
                    help="override dataset root (e.g. a small labelled slice on the eval box)")
    ap.add_argument("--episodes-json", default=None, help="JSON list of episode indices for that slice")
    args = ap.parse_args()

    import dataclasses

    import jax
    import jax.numpy as jnp
    from openpi.models import model as _model
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config
    from openpi.training import data_loader as dl

    cfg = _config.get_config(args.config_name)
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    if args.dataset_root:
        eps = json.load(open(args.episodes_json)) if args.episodes_json else None
        dc = dataclasses.replace(
            dc, dataset_root=args.dataset_root,
            dataset_kwargs={"tolerance_s": 5e-4, **({"episodes": eps} if eps else {})})
        print(f"slice mode: {args.dataset_root} ({len(eps) if eps else 'all'} episodes)", flush=True)
    dataset = dl.transform_dataset(dl.create_b1k_dataset(dc, cfg.model.action_horizon), dc)

    # find frames that actually carry a real point -- the only ones where A vs B is meaningful
    hf = dataset
    for _ in range(6):
        if hasattr(hf, "hf_dataset"):
            hf = hf.hf_dataset
            break
        hf = getattr(hf, "_dataset", None) or getattr(hf, "dataset", None)
    mask_col = np.asarray(hf.data.column("target_points_mask").to_pylist(), dtype=object)
    labelled = [i for i, m in enumerate(mask_col) if m is not None and any(bool(x) for x in m)]
    print(f"labelled frames available: {len(labelled):,}", flush=True)
    rng_np = np.random.default_rng(args.seed)
    sel = rng_np.choice(labelled, min(args.n, len(labelled)), replace=False)

    policy = _policy_config.create_trained_policy(cfg, args.ckpt)
    model = policy._model  # noqa: SLF001

    loader = dl.TorchDataLoader(
        dataset,
        local_batch_size=1,
        sampler=list(int(i) for i in np.sort(sel)),
        num_batches=len(sel),
        num_workers=2,
        seed=args.seed,
    )

    d_ac, d_ab, mag = [], [], []
    for i, batch in enumerate(loader):
        pts = np.asarray(batch["target_points"])
        msk = np.asarray(batch["target_points_mask"])

        variants = {}
        variants["A"] = (pts.copy(), msk.copy())
        b_pts = pts.copy()
        b_pts[..., 0] += args.displace  # shift the target in x
        variants["B"] = (b_pts, msk.copy())
        variants["C"] = (np.zeros_like(pts), np.zeros_like(msk, dtype=bool))

        # one noise draw, reused across A/B/C so the ONLY difference is the conditioning
        noise = jax.random.normal(
            jax.random.key(args.seed + i), (pts.shape[0], cfg.model.action_horizon, 32)
        )
        acts = {}
        for k, (p, m) in variants.items():
            b = dict(batch)
            b["target_points"] = jnp.asarray(p)
            b["target_points_mask"] = jnp.asarray(m)
            obs = _model.Observation.from_dict(b)
            a = model.sample_actions(jax.random.key(args.seed), obs, noise=noise)
            acts[k] = np.asarray(jax.device_get(a))

        d_ac.append(float(np.linalg.norm(acts["A"] - acts["C"])))
        d_ab.append(float(np.linalg.norm(acts["A"] - acts["B"])))
        mag.append(float(np.linalg.norm(acts["A"])))
        if (i + 1) % 16 == 0:
            print(f"  {i+1}/{len(sel)}", flush=True)

    res = {
        "ckpt": args.ckpt,
        "n": len(d_ac),
        "displace_m": args.displace,
        "mean_action_norm": float(np.mean(mag)),
        "A_vs_C_conditioned_vs_not": float(np.mean(d_ac)),
        "A_vs_B_right_vs_wrong_point": float(np.mean(d_ab)),
        # normalised so the numbers are comparable across checkpoints
        "rel_A_vs_C": float(np.mean(d_ac) / max(1e-9, np.mean(mag))),
        "rel_A_vs_B": float(np.mean(d_ab) / max(1e-9, np.mean(mag))),
    }
    res["verdict"] = (
        "DEAD (point ignored)"
        if res["rel_A_vs_C"] < 0.01
        else "PRESENT BUT NOT SPATIALLY SELECTIVE"
        if res["rel_A_vs_B"] < 0.01
        else "LIVE and spatially selective"
    )
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()

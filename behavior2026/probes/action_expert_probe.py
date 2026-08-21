"""Probe the ACTION EXPERT's hidden states -- the module the 3D point actually reaches.

Why this replaces the prefix probe: pi0.py calls
    self.PaliGemma.llm([prefix_tokens, suffix_tokens], ..., adarms_cond=[None, adarms_cond])
i.e. the point-derived conditioning modulates ONLY the suffix (action expert). It is `None` for the
prefix by construction. So probing prefix KV values for target-relative quantities and concluding
"the injection site is wrong" is circular -- the point was never there to find. Two earlier
prefix-probe readings are also explained away rather than meaningful:
  * gripper_cmd R2 ~0.93 FLAT across all 18 layers -> pi0.5's tokenizer literally puts the
    discretized state INTO THE PROMPT TEXT (`f"Task: {t}, State: {s};\\nAction: "`), so this is text
    passthrough, not a learned representation.
  * future_ee_delta ~0 in the prefix -> the prefix is target-agnostic by design.

WHAT THIS MEASURES, on suffix_out (the action expert's own states):
  1. Is the injected TARGET POINT linearly decodable from the action expert's states? If the point
     is present here but the policy still ignores it behaviourally, the failure is downstream of
     representation (motor/commit), not routing.
  2. Is target-relative FUTURE MOTION decodable here (it is not in the prefix)?
  3. How much do the states MOVE when the point changes (real vs displaced vs mask=False)?
     Fixed noise + fixed timestep, so the only difference is the conditioning.

Reading it:
  point decodable + future motion decodable  -> routing is FINE; look at motor/commit and data
  point decodable + future motion NOT        -> information arrives but is not converted to intent
  point NOT decodable                        -> the adaRMS channel really is too weak/late; the
                                                prefix-token redesign becomes justified
"""

import argparse
import json

import numpy as np


def _ridge_r2(X, Y, alpha=10.0, val_frac=0.2, seed=0):
    """Ridge with a held-out split; returns per-dim R^2 on the validation half."""
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    k = int(n * (1 - val_frac))
    tr, va = idx[:k], idx[k:]
    Xtr, Xva = X[tr], X[va]
    Ytr, Yva = Y[tr], Y[va]
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    A = Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1])
    W = np.linalg.solve(A, Xtr.T @ Ytr)
    pred = Xva @ W
    ss_res = ((Yva - pred) ** 2).sum(0)
    ss_tot = ((Yva - Yva.mean(0, keepdims=True)) ** 2).sum(0) + 1e-9
    return 1.0 - ss_res / ss_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="pi05_phaseA_point")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--displace", type=float, default=0.20)
    ap.add_argument("--timestep", type=float, default=0.5, help="fixed flow timestep")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--episodes-json", default=None)
    ap.add_argument("--out", default="/root/action_expert_probe.json")
    args = ap.parse_args()

    import dataclasses

    import jax
    import jax.numpy as jnp
    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask
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
    dataset = dl.transform_dataset(dl.create_b1k_dataset(dc, cfg.model.action_horizon), dc)

    hf = dataset
    for _ in range(6):
        if hasattr(hf, "hf_dataset"):
            hf = hf.hf_dataset
            break
        hf = getattr(hf, "_dataset", None) or getattr(hf, "dataset", None)
    mcol = np.asarray(hf.data.column("target_points_mask").to_pylist(), dtype=object)
    labelled = [i for i, m in enumerate(mcol) if m is not None and any(bool(x) for x in m)]
    print(f"labelled frames: {len(labelled):,}", flush=True)
    rng = np.random.default_rng(args.seed)
    sel = np.sort(rng.choice(labelled, min(args.n, len(labelled)), replace=False))

    policy = _policy_config.create_trained_policy(cfg, args.ckpt)
    model = policy._model  # noqa: SLF001

    def suffix_states(obs, noise, t):
        """Replicate pi0's forward pass and return the ACTION EXPERT's output states."""
        prefix_tokens, prefix_mask, prefix_ar = model.embed_prefix(obs)
        suffix_tokens, suffix_mask, suffix_ar, adarms = model.embed_suffix(
            obs, noise, jnp.broadcast_to(jnp.asarray(t, jnp.float32), (noise.shape[0],))
        )
        mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar = jnp.concatenate([prefix_ar, suffix_ar], axis=0)
        attn = make_attn_mask(mask, ar)
        positions = jnp.cumsum(mask, axis=1) - 1
        (_, suffix_out), _ = model.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn, positions=positions,
            adarms_cond=[None, adarms],
        )
        return suffix_out

    loader = dl.TorchDataLoader(dataset, local_batch_size=1,
                                sampler=[int(i) for i in sel], num_batches=len(sel),
                                num_workers=2, seed=args.seed)

    feats, pts_y, fut_y, d_ac, d_ab, mag = [], [], [], [], [], []
    for i, batch in enumerate(loader):
        pts = np.asarray(batch["target_points"], dtype=np.float32)
        msk = np.asarray(batch["target_points_mask"])
        acts = np.asarray(batch["actions"], dtype=np.float32)
        noise = jax.random.normal(jax.random.key(args.seed + i),
                                  (pts.shape[0], cfg.model.action_horizon, 32))

        out = {}
        for name, (p, m) in {
            "A": (pts, msk),
            "B": (pts + np.array([args.displace, 0, 0], np.float32), msk),
            "C": (np.zeros_like(pts), np.zeros_like(msk, dtype=bool)),
        }.items():
            b = dict(batch)
            b["target_points"] = jnp.asarray(p)
            b["target_points_mask"] = jnp.asarray(m)
            obs = _model.Observation.from_dict(b)
            s = np.asarray(jax.device_get(suffix_states(obs, noise, args.timestep)))
            out[name] = s

        # mean-pool the action tokens -> one feature vector per sample
        f = out["A"][:, -cfg.model.action_horizon:].mean(axis=1).reshape(-1)
        feats.append(f)
        pts_y.append(pts.reshape(-1))                       # can we read the POINT back out?
        fut_y.append(acts[:, :, :].mean(axis=1).reshape(-1))  # mean future action over the chunk
        d_ac.append(float(np.linalg.norm(out["A"] - out["C"])))
        d_ab.append(float(np.linalg.norm(out["A"] - out["B"])))
        mag.append(float(np.linalg.norm(out["A"])))
        if (i + 1) % 32 == 0:
            print(f"  {i+1}/{len(sel)}", flush=True)

    X = np.stack(feats)
    res = {
        "ckpt": args.ckpt, "n": len(X), "feature_dim": int(X.shape[1]),
        "state_shift_rel_point_vs_none": float(np.mean(d_ac) / max(1e-9, np.mean(mag))),
        "state_shift_rel_right_vs_wrong": float(np.mean(d_ab) / max(1e-9, np.mean(mag))),
    }
    for name, Y in (("target_point", np.stack(pts_y)), ("future_action", np.stack(fut_y))):
        r2 = _ridge_r2(X, Y, seed=args.seed)
        res[f"r2_{name}"] = float(np.mean(r2))
    res["verdict"] = (
        "POINT NOT REPRESENTED in action expert -> adaRMS channel too weak; prefix-token redesign justified"
        if res["r2_target_point"] < 0.1
        else "POINT REPRESENTED but future action weakly decodable -> arrives, not converted to intent"
        if res["r2_future_action"] < 0.1
        else "POINT REPRESENTED and future action decodable -> routing FINE; look downstream (motor/commit/data)"
    )
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()

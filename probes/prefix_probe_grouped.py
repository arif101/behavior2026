"""Does the VLM GROUND the target, or just RECOGNIZE the scene?

The problem with prefix_probe.py
--------------------------------
It reported prefix R2 0.731 / action-expert 0.624 against ~0 permutation floors, and concluded the
prefix encodes the target. Two confounds make that unreadable:

  1. It probed episodes=range(180) -- frames the model TRAINED on.
  2. Its CV folds were strided over frames, so ~6 frames of the SAME episode landed on both the
     ridge's train and test sides. Every frame in an episode shares one target value, so the
     decoder can win by recognizing WHICH SCENE it is looking at and recalling the answer. That is
     a lookup table, not localization.

Confound 2 is not hypothetical: val loss on this checkpoint is 3.05x train (0.0658 vs 0.0215), so
we already know this model memorizes episodes. "Encodes a scene->target lookup for 180 memorized
scenes" predicts exactly the R2 we measured.

The design here
---------------
Fit the decoder on one set of episodes and score it on DISJOINT sets. Grouping is guaranteed by
construction -- no episode can appear on both sides -- so no per-frame episode bookkeeping and no
assumption about loader ordering is needed. Three sets:

  FIT      ep   0-149   model trained on these; decoder is fit here
  SEEN     ep 150-179   model trained on these, decoder did NOT fit here
  UNSEEN   ep 180-199   model NEVER trained on these (the val split)

Reading the result:

  SEEN high, UNSEEN high  -> genuine visual grounding. The target is recoverable from the VLM's
                            representation of scenes the decoder never fit. Prefix injection is
                            unnecessary; my soft-token hypothesis is dead and the generalization
                            gap lives downstream of perception, in the action decoder.
  SEEN high, UNSEEN low   -> grounding works only on trained scenes. The VLM memorized 180
                            scene->target associations. Perception is the bottleneck after all,
                            but the fix is data diversity, not a prefix token.
  BOTH low                -> the earlier 0.731 was pure episode leakage. The VLM is target-blind,
                            and the soft-token argument comes back to life.

Same permutation control as before, granted the same max-over-alpha advantage, so a positive R2
that merely reflects estimator optimism reads as NO SIGNAL.

Usage:
  python prefix_probe_grouped.py --ckpt /path/to/params --config pi05_radio_gate
"""

from __future__ import annotations

import argparse
import dataclasses
import json

import numpy as np


def _episode_loader(cfg, episodes, n_batches):
    """Same config, restricted to a disjoint episode set.

    dataset_kwargs lives on cfg.data.base_config, not on the factory (see val_loss.py).
    """
    import openpi.training.data_loader as _data_loader

    base = cfg.data.base_config
    kwargs = dict(base.dataset_kwargs)
    kwargs["episodes"] = list(episodes)
    cfg2 = dataclasses.replace(
        cfg,
        data=dataclasses.replace(
            cfg.data, base_config=dataclasses.replace(base, dataset_kwargs=kwargs)
        ),
    )
    return _data_loader.create_b1k_data_loader(
        cfg2, shuffle=True, num_batches=n_batches, skip_norm_stats=False
    )


def collect(model, cfg, episodes, n, seed):
    """Mean-pooled prefix activations, last action-expert token, and the target, for one episode set."""
    import jax
    import jax.numpy as jnp
    from openpi.models import pi0 as _pi0

    n_batches = int(np.ceil(n / cfg.batch_size)) * 2 + 4
    loader = _episode_loader(cfg, episodes, n_batches)
    rng = jax.random.key(seed)

    P, S, Y = [], [], []
    for obs, _act in loader:
        prefix_tokens, prefix_mask, prefix_ar = model.embed_prefix(obs)
        bsz = prefix_tokens.shape[0]
        t = jnp.zeros((bsz,), dtype=jnp.float32)
        x_t = jax.random.normal(rng, (bsz, model.action_horizon, model.action_dim))
        suffix_tokens, suffix_mask, suffix_ar, adarms = model.embed_suffix(obs, x_t, t)

        # input_mask bool[B,N] -> axis=1; ar_mask bool[N] unbatched -> axis=0 (pi0.py:287-289).
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar, suffix_ar], axis=0)
        attn = _pi0.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = model.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn, positions=positions,
            adarms_cond=[None, adarms],
        )

        tm = np.asarray(obs.target_points_mask).reshape(bsz, -1)
        keep = tm.all(axis=1)
        if not keep.any():
            continue

        m = prefix_mask[..., None].astype(prefix_out.dtype)
        pooled = np.asarray((prefix_out * m).sum(1) / jnp.maximum(m.sum(1), 1e-6), dtype=np.float32)
        P.append(pooled[keep])
        S.append(np.asarray(suffix_out[:, -1], dtype=np.float32)[keep])
        Y.append(np.asarray(obs.target_points, dtype=np.float32).reshape(bsz, -1)[keep])
        if sum(len(x) for x in P) >= n:
            break

    if not P:
        raise SystemExit(f"no frames survived the target mask for episodes {episodes[0]}..{episodes[-1]}")
    return np.concatenate(P)[:n], np.concatenate(S)[:n], np.concatenate(Y)[:n]


ALPHAS = [1.0, 10.0, 100.0, 1e3, 1e4, 1e5]


def _fit(Xtr, ytr, alpha):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Z = (Xtr - mu) / sd
    ym = ytr.mean(0)
    W = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (ytr - ym))
    return W, mu, sd, ym


def _r2(W, mu, sd, ym, Xte, yte):
    pred = ((Xte - mu) / sd) @ W + ym
    ss_res = ((yte - pred) ** 2).sum()
    ss_tot = ((yte - yte.mean(0)) ** 2).sum()
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


def cross_episode(Xtr, ytr, Xte, yte, rng):
    """Best-over-alpha R2 on a DISJOINT episode set, plus a permutation control with the same
    max-over-alpha advantage."""
    real, ctrl = {}, {}
    yte_perm = yte[rng.permutation(len(yte))]
    for a in ALPHAS:
        W, mu, sd, ym = _fit(Xtr, ytr, a)
        real[a] = _r2(W, mu, sd, ym, Xte, yte)
        ctrl[a] = _r2(W, mu, sd, ym, Xte, yte_perm)
    best = max(real, key=real.get)
    return {
        "r2": round(real[best], 4),
        "alpha": best,
        "r2_shuffled_control": round(max(ctrl.values()), 4),
        "signal_over_control": round(real[best] - max(ctrl.values()), 4),
        "r2_by_alpha": {str(a): round(v, 4) for a, v in real.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="pi05_radio_gate")
    ap.add_argument("--n_fit", type=int, default=768)
    ap.add_argument("--n_test", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="prefix_probe_grouped.json")
    a = ap.parse_args()

    import jax.numpy as jnp
    from openpi.training import config as _config
    import openpi.models.model as _model

    cfg = _config.get_config(a.config)
    model = cfg.model.load(_model.restore_params(a.ckpt, dtype=jnp.bfloat16))

    print("collecting FIT    ep   0-149 (model trained; decoder fit here)", flush=True)
    Pf, Sf, Yf = collect(model, cfg, range(0, 150), a.n_fit, a.seed)
    print(f"  {len(Pf)} frames", flush=True)
    print("collecting SEEN   ep 150-179 (model trained; decoder did NOT fit)", flush=True)
    Ps, Ss, Ys = collect(model, cfg, range(150, 180), a.n_test, a.seed + 1)
    print(f"  {len(Ps)} frames", flush=True)
    print("collecting UNSEEN ep 180-199 (model NEVER trained on these)", flush=True)
    Pu, Su, Yu = collect(model, cfg, range(180, 200), a.n_test, a.seed + 2)
    print(f"  {len(Pu)} frames", flush=True)

    rng = np.random.default_rng(a.seed)
    res = {
        "ckpt": a.ckpt,
        "n_fit": int(len(Pf)), "n_seen": int(len(Ps)), "n_unseen": int(len(Pu)),
        "PREFIX": {
            "SEEN_ep150_179": cross_episode(Pf, Yf, Ps, Ys, rng),
            "UNSEEN_ep180_199": cross_episode(Pf, Yf, Pu, Yu, rng),
        },
        "ACTION_EXPERT": {
            "SEEN_ep150_179": cross_episode(Sf, Yf, Ss, Ys, rng),
            "UNSEEN_ep180_199": cross_episode(Sf, Yf, Su, Yu, rng),
        },
    }

    # Key the verdict off R2, not signal_over_control. Calibration (calib_probe.py) shows the
    # control goes strongly NEGATIVE when the decoder is good -- predicting shuffled targets is
    # worse than predicting the mean -- so signal_over_control can exceed 1.0 and would inflate
    # these thresholds. R2 itself is the honest scale: on low-rank features like real activations
    # a genuine linear relationship reads ~1.0, and unrelated targets read negative.
    pre_u = res["PREFIX"]["UNSEEN_ep180_199"]["r2"]
    pre_s = res["PREFIX"]["SEEN_ep150_179"]["r2"]
    if pre_u > 0.30:
        res["verdict"] = ("GENUINE GROUNDING: target decodes on episodes the model never saw. "
                          "Prefix soft token NOT warranted; the gap is downstream of perception.")
    elif pre_s > 0.30:
        res["verdict"] = ("SCENE MEMORIZATION: decodes on trained episodes only. Perception is the "
                          "bottleneck, but the fix is data diversity, not a prefix token.")
    else:
        res["verdict"] = ("TARGET-BLIND once episode leakage is removed: the earlier 0.731 was "
                          "leakage. Soft token in the KV cache is back on the table.")

    print(json.dumps(res, indent=1))
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()

"""Is the TARGET encoded in the prefix (KV cache), or only in the action expert?

Why this matters
----------------
Point conditioning currently enters ONLY the action expert:

    use_adarms=[False, True]            # pi0.py:80  — AdaLN on the action-expert tower only
    adarms_cond=[None, adarms_cond]     # pi0.py:292 — None for the prefix

The prefix is where images and language are attended together — the semantic processing. If the
target is absent there, the VLM is building a generic scene representation and never learns WHICH
object matters; the point only modulates motor output downstream. "Which object is the goal" is a
scene-understanding question being answered in the wrong tower.

The evidence usually cited against prefix injection (arXiv 2606.27663: AdaLN 77.5 vs "text prompt"
38.5) compared AdaLN against the point rendered as TEXT — numbers the VLM must parse out of
language. That says little about a LEARNED token in the KV cache, which is exactly what SERF uses
for map tokens (40.7 -> 63.5). Encoding and site are different variables; that comparison conflated
them.

What this measures
------------------
Ridge-decode the target position from prefix KV activations, and compare against the same decode
from the action expert (`suffix_out`), which reads R2 0.369 for the point / 0.046 for the future
action.

  prefix R2 HIGH  -> the VLM already encodes the target; action-expert injection is sufficient
  prefix R2 LOW   -> the VLM is target-blind; a learned point token in the KV cache is warranted,
                     and it lands via the same mechanism as SERF's map tokens

Seeded off --seed alone (never off the checkpoint) so every arm sees identical frames AND identical
flow-matching noise — the same discipline as probes/score_arms.py.

Usage:
  python prefix_probe.py --ckpt /path/to/params --dataset b1k_radio --n 256 --out prefix_probe.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def ridge_r2(X: np.ndarray, y: np.ndarray, alpha: float = 1.0, folds: int = 5) -> float:
    """Cross-validated R^2 of a ridge decode. CV matters: at ~1k frames and high feature dim a
    plain fit reports strongly positive R^2 on pure noise (we hit exactly that earlier with
    future_ee_delta, which flipped sign between 1200 and 3000 frames)."""
    n = len(X)
    idx = np.arange(n)
    scores = []
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        Xtr, Xte = X[tr], X[te]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        ym = y[tr].mean(0)
        W = np.linalg.solve(Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1]), Xtr.T @ (y[tr] - ym))
        pred = Xte @ W + ym
        ss_res = ((y[te] - pred) ** 2).sum()
        ss_tot = ((y[te] - y[te].mean(0)) ** 2).sum()
        scores.append(1.0 - ss_res / max(ss_tot, 1e-12))
    return float(np.mean(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="pi05_phaseA_point")
    ap.add_argument("--n", type=int, default=256, help="frames to probe")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="prefix_probe.json")
    a = ap.parse_args()

    import jax
    import jax.numpy as jnp

    from openpi.training import config as _config
    from openpi.policies import policy_config
    from openpi.shared import download

    cfg = _config.get_config(a.config)
    ckpt = download.maybe_download(a.ckpt)
    policy = policy_config.create_trained_policy(cfg, ckpt)
    model = policy._model

    loader = _config.create_data_loader(cfg, num_workers=0, shuffle=True, seed=a.seed)
    rng = jax.random.key(a.seed)

    P, S, Y = [], [], []
    for batch in loader:
        obs, _ = batch
        # Embed prefix and suffix exactly as sample_actions does, so the activations are the ones
        # the policy actually conditions on.
        prefix_tokens, prefix_mask, prefix_ar = model.embed_prefix(obs)
        t = jnp.zeros((prefix_tokens.shape[0],), dtype=jnp.float32)
        x_t = jax.random.normal(rng, (prefix_tokens.shape[0], model.action_horizon, model.action_dim))
        suffix_tokens, suffix_mask, suffix_ar, adarms = model.embed_suffix(obs, x_t, t)

        from openpi.models import model as _m
        attn = _m.make_attn_mask(jnp.concatenate([prefix_mask, suffix_mask], axis=1),
                                 jnp.concatenate([prefix_ar, suffix_ar], axis=1))
        positions = jnp.cumsum(jnp.concatenate([prefix_mask, suffix_mask], axis=1), axis=1) - 1
        (prefix_out, suffix_out), _ = model.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn, positions=positions,
            adarms_cond=[None, adarms])

        # Mean-pool over valid prefix tokens; last suffix token for the action side.
        m = prefix_mask[..., None].astype(prefix_out.dtype)
        P.append(np.asarray((prefix_out * m).sum(1) / jnp.maximum(m.sum(1), 1e-6)))
        S.append(np.asarray(suffix_out[:, -1]))
        Y.append(np.asarray(obs.target_points).reshape(len(P[-1]), -1))
        if sum(len(x) for x in P) >= a.n:
            break

    P, S, Y = np.concatenate(P)[: a.n], np.concatenate(S)[: a.n], np.concatenate(Y)[: a.n]
    res = {
        "ckpt": a.ckpt, "n": int(len(P)),
        "r2_target_from_PREFIX": ridge_r2(P, Y),
        "r2_target_from_ACTION_EXPERT": ridge_r2(S, Y),
    }
    gap = res["r2_target_from_ACTION_EXPERT"] - res["r2_target_from_PREFIX"]
    res["verdict"] = (
        "PREFIX TARGET-BLIND -> learned point token in KV cache warranted" if gap > 0.15
        else "prefix encodes the target -> action-expert injection sufficient"
    )
    print(json.dumps(res, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

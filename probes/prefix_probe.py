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
language. That says little about a LEARNED soft token in the KV cache, which is exactly what SERF
uses for map tokens (40.7 -> 63.5). Encoding and site are different variables; that comparison
conflated them.

What this measures
------------------
Ridge-decode the target position from prefix activations, and compare against the same decode from
the action expert (`suffix_out`), which is the tower the point is actually injected into.

  prefix R2 HIGH  -> the VLM already infers the target from pixels+language; AdaLN is sufficient
  prefix R2 LOW   -> the VLM is target-blind; a learned point token in the KV cache is warranted,
                     and it lands via the same mechanism as SERF's map tokens

Methodology notes (learned the hard way)
----------------------------------------
* d >> n. Prefix width is ~2048 and we probe ~1k frames, so a ridge decode reports strongly
  positive R2 on PURE NOISE at weak regularization. We hit exactly that with future_ee_delta, whose
  R2 flipped sign between 1200 and 3000 frames. So this probe (a) cross-validates, (b) sweeps alpha,
  and (c) runs a PERMUTATION CONTROL — the same decode against shuffled targets. The control gets
  the identical max-over-alpha selection advantage, so it is a fair noise floor. Any arm that does
  not clear its control is reported as NO SIGNAL regardless of raw R2.
* Frames come from create_b1k_data_loader, NOT create_data_loader: the latter routes to
  create_torch_dataset(), which builds LeRobotDatasetMetadata(repo_id) with no root, ignores
  dataset_root, and falls through to the Hub as a misleading 401.
* The loader seeds off config.seed, so every checkpoint sees identical frames.
* Only frames with target_points_mask True on both arms are used — masked-out frames carry a zero
  target that is trivially predictable and would inflate both arms.

Usage:
  python prefix_probe.py --ckpt /path/to/checkpoint/params --config pi05_radio_gate --n 1024
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def _ridge_r2_at(X: np.ndarray, y: np.ndarray, alpha: float, folds: int = 5) -> float:
    """Cross-validated R^2 of a ridge decode at one alpha."""
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


ALPHAS = [1.0, 10.0, 100.0, 1e3, 1e4, 1e5]


def probe(X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> dict:
    """Best-over-alpha R2, plus a permutation control granted the same selection advantage."""
    real = {a: _ridge_r2_at(X, y, a) for a in ALPHAS}
    y_perm = y[rng.permutation(len(y))]
    ctrl = {a: _ridge_r2_at(X, y_perm, a) for a in ALPHAS}
    best_a = max(real, key=real.get)
    return {
        "r2": real[best_a],
        "alpha": best_a,
        "r2_shuffled_control": max(ctrl.values()),
        "r2_by_alpha": {str(a): round(v, 4) for a, v in real.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to the checkpoint's params/ dir")
    ap.add_argument("--config", default="pi05_radio_gate")
    ap.add_argument("--n", type=int, default=1024, help="frames to probe")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="prefix_probe.json")
    a = ap.parse_args()

    import jax
    import jax.numpy as jnp

    from openpi.training import config as _config
    import openpi.training.data_loader as _data_loader
    import openpi.models.model as _model
    from openpi.models import pi0 as _pi0  # make_attn_mask is defined here, NOT in model.py

    cfg = _config.get_config(a.config)
    model = cfg.model.load(_model.restore_params(a.ckpt, dtype=jnp.bfloat16))

    # num_batches is generous: masked-out frames get dropped, so ask for more than n/batch.
    n_batches = int(np.ceil(a.n / cfg.batch_size)) * 2 + 4
    loader = _data_loader.create_b1k_data_loader(
        cfg, shuffle=True, num_batches=n_batches, skip_norm_stats=False
    )
    rng = jax.random.key(a.seed)

    P, S, Y = [], [], []
    for obs, _act in loader:
        prefix_tokens, prefix_mask, prefix_ar = model.embed_prefix(obs)
        bsz = prefix_tokens.shape[0]
        t = jnp.zeros((bsz,), dtype=jnp.float32)
        x_t = jax.random.normal(rng, (bsz, model.action_horizon, model.action_dim))
        suffix_tokens, suffix_mask, suffix_ar, adarms = model.embed_suffix(obs, x_t, t)

        # input_mask is bool[B, N] -> axis=1; ar_mask is bool[N] (UNBATCHED) -> axis=0.
        # Mirrors Pi0.__call__ exactly (pi0.py:287-289); concatenating ar on axis=1 raises
        # "axis 1 is out of bounds for array of dimension 1".
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar, suffix_ar], axis=0)
        attn = _pi0.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = model.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn, positions=positions,
            adarms_cond=[None, adarms],
        )

        # Keep only frames whose target is actually present on both arms.
        tm = np.asarray(obs.target_points_mask).reshape(bsz, -1)
        keep = tm.all(axis=1)
        if not keep.any():
            continue

        m = prefix_mask[..., None].astype(prefix_out.dtype)
        pooled = np.asarray((prefix_out * m).sum(1) / jnp.maximum(m.sum(1), 1e-6), dtype=np.float32)
        P.append(pooled[keep])
        S.append(np.asarray(suffix_out[:, -1], dtype=np.float32)[keep])
        Y.append(np.asarray(obs.target_points, dtype=np.float32).reshape(bsz, -1)[keep])
        if sum(len(x) for x in P) >= a.n:
            break

    if not P:
        raise SystemExit("no frames survived the target mask — check point coverage")

    P, S, Y = np.concatenate(P)[: a.n], np.concatenate(S)[: a.n], np.concatenate(Y)[: a.n]
    nprng = np.random.default_rng(a.seed)

    pre, act = probe(P, Y, nprng), probe(S, Y, nprng)
    # Signal only counts if it clears its own shuffled-target floor by a clear margin.
    pre_sig = pre["r2"] - pre["r2_shuffled_control"]
    act_sig = act["r2"] - act["r2_shuffled_control"]

    res = {
        "ckpt": a.ckpt,
        "n_frames": int(len(P)),
        "prefix_dim": int(P.shape[1]),
        "PREFIX": pre,
        "ACTION_EXPERT": act,
        "prefix_signal_over_control": round(pre_sig, 4),
        "action_expert_signal_over_control": round(act_sig, 4),
    }
    if pre_sig < 0.05:
        res["verdict"] = (
            "PREFIX TARGET-BLIND (does not clear its shuffled control) -> "
            "learned point token in KV cache warranted"
        )
    elif act_sig - pre_sig > 0.15:
        res["verdict"] = (
            "PREFIX WEAK relative to action expert -> soft token in KV cache likely to help"
        )
    else:
        res["verdict"] = "prefix encodes the target -> action-expert injection sufficient"

    print(json.dumps(res, indent=1))
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()

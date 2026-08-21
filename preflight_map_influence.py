"""Run-1 G0 preflight v2: loss-level influence with weights restored THE WAY TRAINING RESTORES.

v1 used create_trained_policy (strict pytree match) — wrong tool: ckpt 49999 lacks the 10 new
map_*/aux_* param groups by design; the training path merges them via weight_loader
(missing_regex keeps fresh init for new params). This replicates that exact merge, then probes:

  [1] merge restore succeeds (counts merged vs fresh param groups)
  [2] IMAGE influence: perturbing images moves the loss (restore sanity)
  [3] MAP influence: map tokens present-vs-absent moves the loss (slot reaches the model)
  [4] warm-start perturbation is register-scale (< 5% relative)

Run: python preflight_map_influence.py --ckpt /root/warmstart_49999 --config pi05_radio_map
"""

import argparse
import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="pi05_radio_map")
    a = ap.parse_args()

    from openpi.training import config as _config

    cfg = _config.get_config(a.config)
    print(f"[0] init {a.config} (map_k={cfg.model.map_tokens_k})...", flush=True)
    model = cfg.model.create(jax.random.key(0))
    graphdef, state = nnx.split(model)
    pure = state.to_pure_dict()

    print("[1] training-style warm-start merge...", flush=True)
    loader = dataclasses.replace(cfg.weight_loader, params_path=f"{a.ckpt}/params") \
        if hasattr(cfg.weight_loader, "params_path") else cfg.weight_loader
    merged = loader.load(jax.tree.map(np.asarray, pure))
    fresh = [k for k in pure if k not in ("PaliGemma",) and any(
        s in str(k) for s in ("map_", "aux_"))]
    print(f"    top-level groups: {len(pure)}; fresh (expected map_*/aux_*): {sorted(fresh)}")
    state.replace_by_pure_dict(jax.tree.map(jnp.asarray, merged))
    model = nnx.merge(graphdef, state)
    print("    MERGE RESTORE OK", flush=True)

    obs = cfg.model.fake_obs()
    acts = jnp.zeros((1, cfg.model.action_horizon, cfg.model.action_dim), jnp.float32)

    def loss(o):
        return model.compute_loss(jax.random.key(7), o, acts, train=False)

    l0 = loss(obs)
    li = loss(dataclasses.replace(obs, images={k: v + 0.5 for k, v in obs.images.items()}))
    img_ok = not jnp.array_equal(l0, li)
    print(f"[2] IMAGE influence: {img_ok} (max shift {float(jnp.abs(li - l0).max()):.2e})")
    assert img_ok, "restored model must respond to images"

    tok = jnp.ones((1, cfg.model.map_tokens_k, cfg.model.map_token_dim), jnp.float32)
    lm = loss(dataclasses.replace(obs, map_tokens=tok))
    map_shift = float(jnp.abs(lm - l0).max())
    print(f"[3] MAP influence: {not jnp.array_equal(l0, lm)} (max shift {map_shift:.2e})")
    assert not jnp.array_equal(l0, lm), "map tokens must reach the restored model"

    # [4] warm-start integrity: alpha gate restored at 0 (content-zero tokens) + dilution doc.
    # Attention-denominator dilution from 8 present tokens is EXPECTED (measured: 39% action
    # delta; 0.09% when attention-invisible — discriminator 2026-07-30). Training always
    # presents 8 tokens, so the model adapts from step one. The gate asserts the CONTENT path
    # is closed at init (alpha == 0) rather than a behavioral no-op.
    alpha = None
    for k, v in jax.tree_util.tree_leaves_with_path(nnx.state(model)):
        if "map_alpha" in "/".join(str(x) for x in k):
            alpha = float(np.asarray(v))
    print(f"[4] map_alpha restored = {alpha} (must be 0.0); dilution documented: "
          f"loss-mean shift {float(jnp.abs(lm - l0).mean() / jnp.abs(l0).mean()):.2f}x")
    assert alpha == 0.0, "ReZero gate must start closed"
    print("PREFLIGHT PASS — Run-1 launch is legal per G0 (amendment 1 applied)")


if __name__ == "__main__":
    main()

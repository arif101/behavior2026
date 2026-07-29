"""Run-1 warm-start preflight: loss-level map-token influence with RESTORED weights.

WHY THIS EXISTS: ad-hoc created models cannot test influence — nnx_bridge lazy_init leaves
transformer-block kernels all-zero (measured 9/20 llm leaves), so the blocks output nothing and
NO prefix content (images, text, map tokens) moves the loss. Discovered 2026-07-29 while gating
PREP(d). With a restored checkpoint all params are real and influence is measurable.

RUN ON THE TRAINING BOX before launching Run 1 (adapt CKPT/CONFIG paths):

    python preflight_map_influence.py --ckpt /path/to/49999 --config pi05_radio_map

Checks, in order:
  1. restore succeeds with the K=8 config (new params init fresh, everything else from ckpt)
  2. IMAGE influence: perturbing images moves the loss (sanity that restore worked)
  3. MAP influence: map tokens present-vs-absent moves the loss (the slot reaches the model)
  4. warm-start perturbation: |loss(with tokens @ zero-init) - loss(without)| is SMALL
     (register-only effect) — the number to record in the run log
ALL FOUR must pass before the Run-1 launch is legal per FOVEATED_MEMORY_SPEC_v1 G0.
"""

import argparse
import dataclasses

import jax
import jax.numpy as jnp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="pi05_radio_map")
    a = ap.parse_args()

    from openpi.training import config as _config
    from openpi.policies import policy_config

    cfg = _config.get_config(a.config)
    model = policy_config.create_trained_policy(cfg, a.ckpt)._model  # noqa: SLF001

    mcfg = cfg.model
    obs = mcfg.fake_obs()
    acts = jnp.zeros((1, mcfg.action_horizon, mcfg.action_dim), jnp.float32)

    def loss(o):
        return model.compute_loss(jax.random.key(7), o, acts, train=False)

    l0 = loss(obs)
    li = loss(dataclasses.replace(obs, images={k: v + 0.5 for k, v in obs.images.items()}))
    img_ok = not jnp.array_equal(l0, li)
    print(f"[2] IMAGE influence: {img_ok}  (max shift {float(jnp.abs(li - l0).max()):.2e})")
    assert img_ok, "restored model must respond to images — restore is broken"

    tok = jnp.ones((1, mcfg.map_tokens_k, mcfg.map_token_dim), jnp.float32)
    lm = loss(dataclasses.replace(obs, map_tokens=tok))
    map_shift = float(jnp.abs(lm - l0).max())
    print(f"[3] MAP influence: {not jnp.array_equal(l0, lm)}  (max shift {map_shift:.2e})")
    assert not jnp.array_equal(l0, lm), "map tokens must reach the restored model"

    rel = map_shift / max(float(jnp.abs(l0).max()), 1e-9)
    print(f"[4] warm-start perturbation (record in run log): relative {rel:.2e}")
    assert rel < 0.05, f"zero-init perturbation should be register-scale, got {rel:.2e}"
    print("PREFLIGHT PASS — Run 1 launch is legal per G0")


if __name__ == "__main__":
    main()

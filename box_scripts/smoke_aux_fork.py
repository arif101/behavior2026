"""Fork-side smoke for the Run-1 aux losses (dummy variants, JAX CPU, imports the FORK).

Checks:
  1. K=0: no aux/map params (bit-parity intact after the aux patch)
  2. K=8: aux heads exist; compute_loss FINITE with map_tokens + target_points supplied
  3. blind-stream masking: zeroing T0 changes the loss vs full tokens (e_map gated off)
  4. aux actually contributes: scaling aux head weights changes the loss (params are live
     in an ad-hoc model because the aux heads sit OUTSIDE the zero-param bridged llm —
     they read prefix_out and add to the loss directly)
"""

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import pi0_config


def make(k):
    cfg = pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy",
                               pi05=True, map_tokens_k=k)
    return cfg, cfg.create(jax.random.key(0))


def paths(m):
    return ["/".join(str(k) for k, in [(str(p),) for p in kp]) if False else
            "/".join(str(x) for x in kp)
            for kp, _ in jax.tree_util.tree_leaves_with_path(nnx.state(m))]


cfg0, m0 = make(0)
p0 = paths(m0)
assert not any("aux_" in p or "map_" in p for p in p0), "K=0 must have no aux/map params"
print("PASS k0 no aux params")

cfg, m = make(8)
p8 = paths(m)
assert any("aux_map_head_in" in p for p in p8) and any("aux_img_head_out" in p for p in p8)
print("PASS k8 aux params exist")

obs = cfg.fake_obs()
tok = jnp.ones((1, 8, 72), jnp.float32)
tp = jnp.ones((1, 2, 3), jnp.float32) * 0.5
kw = {"map_tokens": tok, "target_points": tp,
      "stage": jnp.array([2], jnp.int32),
      "aux_pixels": jnp.array([[0.5, 0.5, 1.0, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]], jnp.float32)}
if hasattr(obs, "target_points_mask"):
    kw["target_points_mask"] = jnp.ones((1, 2), bool)
obs_full = dataclasses.replace(obs, **kw)
acts = jnp.zeros((1, cfg.action_horizon, cfg.action_dim), jnp.float32)

l_full = m.compute_loss(jax.random.key(7), obs_full, acts, train=False)
assert bool(jnp.isfinite(l_full).all()), "loss must be finite with aux active"
print("PASS k8 aux loss finite:", float(l_full.mean()))

blind = tok.at[:, 0, :].set(0.0)
l_blind = m.compute_loss(jax.random.key(7), dataclasses.replace(obs_full, map_tokens=blind), acts,
                         train=False)
assert not jnp.array_equal(l_full, l_blind), "blind gating must change the aux contribution"
print("PASS blind-stream gating changes loss:", float(l_full.mean()) - float(l_blind.mean()))

l_nostage = m.compute_loss(jax.random.key(7),
                           dataclasses.replace(obs_full, stage=None, aux_pixels=None), acts,
                           train=False)
assert not jnp.array_equal(l_full, l_nostage), "stage/heat aux must contribute when labels present"
print("PASS stage+heat aux contribute:", float(l_full.mean()) - float(l_nostage.mean()))

m.aux_img_head_out.kernel.value = m.aux_img_head_out.kernel.value + 3.0
l_scaled = m.compute_loss(jax.random.key(7), obs_full, acts, train=False)
assert not jnp.array_equal(l_full, l_scaled), "aux head weights must influence the loss"
print("PASS aux pathway live (weight surgery shifts loss)")
print("ALL AUX SMOKE PASS")

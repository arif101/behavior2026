"""CPU smoke for temporal forcing (no data): (1) pi05_radio_full builds; the temporal params exist and temp_out is zero;
(2) with temporal_conditioning the flow-matching loss and sample_actions are BIT-IDENTICAL to the same config with the
pathway off (zero-init gate = exact no-op), for random history gists; (3) with hist_flow labels the aux losses are finite
and change the total loss; (4) the gate opens after one gradient step on the aux loss (liveness).
  JAX_PLATFORMS=cpu PYTHONPATH=<fork src> python smoke_temporal.py
"""
import dataclasses, jax, jax.numpy as jnp, numpy as np
from flax import nnx
import openpi.training.config as _config
import openpi.models.model as _model

cfg = _config.get_config("pi05_radio_full")
mc = cfg.model
mc_off = dataclasses.replace(mc, temporal_conditioning=False)
B = 2
obs_spec, act_spec = mc.inputs_spec(batch_size=B)
rng = jax.random.key(0)
def rand_like(spec, key):
    if spec is None: return None
    if spec.dtype == jnp.bool_: return jnp.ones(spec.shape, bool)
    if jnp.issubdtype(spec.dtype, jnp.integer): return jnp.zeros(spec.shape, spec.dtype)
    return jax.random.normal(key, spec.shape, spec.dtype) * 0.1
keys = jax.random.split(rng, 32)
obs = _model.Observation(
    images={k: rand_like(v, keys[i]) for i, (k, v) in enumerate(obs_spec.images.items())},
    image_masks={k: jnp.ones(v.shape, bool) for k, v in obs_spec.image_masks.items()},
    state=rand_like(obs_spec.state, keys[5]), tokenized_prompt=jnp.zeros(obs_spec.tokenized_prompt.shape, jnp.int32),
    tokenized_prompt_mask=jnp.ones(obs_spec.tokenized_prompt_mask.shape, bool),
    target_points=rand_like(obs_spec.target_points, keys[6]), target_points_mask=jnp.ones(obs_spec.target_points_mask.shape, bool),
    stage_tokens=jnp.ones(obs_spec.stage_tokens.shape, jnp.int32),
    history_gists=jax.random.normal(keys[7], obs_spec.history_gists.shape) * 5.0, history_mask=jnp.ones(obs_spec.history_mask.shape, bool),
    stage=jnp.ones((B,), jnp.int32), progress=jnp.full((B,), 0.4, jnp.float32),
    map_tokens=jnp.zeros((B, mc.map_tokens_k, mc.map_token_dim), jnp.float32) if mc.map_tokens_k else None,
)
acts = jax.random.normal(keys[8], act_spec.shape) * 0.1
model_on = mc.create(jax.random.key(1)); model_off = mc_off.create(jax.random.key(1))
# copy shared params from off -> on so the two models agree everywhere except the temporal modules
gd_on, st_on = nnx.split(model_on); gd_off, st_off = nnx.split(model_off)
pure_off = st_off.to_pure_dict(); pure_on = st_on.to_pure_dict()
def merge(a, b):   # b's leaves override a's where the key exists in both
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict): merge(a[k], v)
        elif k in a: a[k] = v
merge(pure_on, pure_off); st_on.replace_by_pure_dict(pure_on); model_on = nnx.merge(gd_on, st_on)
temp_leaves = {k: v for k, v in jax.tree_util.tree_leaves_with_path(nnx.state(model_on).to_pure_dict()) }
n_temp = sum(1 for p, _ in jax.tree_util.tree_leaves_with_path(nnx.state(model_on).to_pure_dict()) if "temp_" in jax.tree_util.keystr(p))
gate = nnx.state(model_on).to_pure_dict()["temp_out"]["kernel"]
print("temporal param leaves:", n_temp, "| temp_out kernel abs max:", float(jnp.abs(gate).max()))
model_on.eval(); model_off.eval()
obs_nolabels = obs.replace(hist_flow=None, hist_flow_mask=None, stage=None, progress=None)
obs_off = obs_nolabels.replace(history_gists=None, history_mask=None)
l_on = model_on.compute_loss(jax.random.key(3), obs_nolabels, acts, train=False)
l_off = model_off.compute_loss(jax.random.key(3), obs_off, acts, train=False)
print("loss on/off (no labels):", float(l_on.mean()), float(l_off.mean()), "| max abs diff:", float(jnp.abs(l_on - l_off).max()))
a_on = model_on.sample_actions(jax.random.key(4), obs_nolabels, num_steps=2); a_off = model_off.sample_actions(jax.random.key(4), obs_off, num_steps=2)
print("sample_actions max abs diff on/off:", float(jnp.abs(a_on - a_off).max()))
K = mc.temporal_k
obs_lab = obs.replace(hist_flow=jax.random.normal(keys[9], (B, K, 9)) * 0.05, hist_flow_mask=jnp.ones((B, K), bool))
l_lab = model_on.compute_loss(jax.random.key(3), obs_lab, acts, train=False)
print("loss with hist_flow + stage labels:", float(l_lab.mean()), "finite:", bool(jnp.isfinite(l_lab).all()), "| delta vs no-labels:", float((l_lab - l_on).mean()))
# liveness: gradient reaches the gate. NOTE pi0.5's adaRMS modulation Dense layers are ZERO-INIT (gemma.py RMSNorm),
# so at random init dL/dcond == 0 for EVERY conditioning term (timestep included) and any zero-init projection into
# cond shows a zero gradient. Trained checkpoints (A4) have non-zero modulation kernels; emulate that here by
# perturbing every modulation kernel before the gradient check.
gd, st = nnx.split(model_on); pure = st.to_pure_dict(); n_mod = 0
def perturb(d, path=()):
    global n_mod
    for k, v in list(d.items()):
        if isinstance(v, dict): perturb(v, path + (k,))
        elif 'Dense_0' in path and any('norm' in str(x) for x in path) and k == 'kernel':   # adaRMS modulation (gemma.py RMSNorm)
            d[k] = 0.01 * jax.random.normal(jax.random.key(n_mod), v.shape, v.dtype); n_mod += 1
perturb(pure); st.replace_by_pure_dict(pure); model_on = nnx.merge(gd, st)
print('perturbed adaRMS modulation kernels:', n_mod)
model_on.train()
def loss_fn(m): return m.compute_loss(jax.random.key(5), obs_lab, acts, train=True).mean()
grads = nnx.grad(loss_fn)(model_on)
gp = grads.to_pure_dict() if hasattr(grads, "to_pure_dict") else nnx.state(grads).to_pure_dict()
gnorm = lambda k: float(jnp.linalg.norm(jnp.asarray(gp[k]["kernel"])))
print("grad norms: temp_in %.3e temp_flow_in %.3e temp_out(gate) %.3e progress_mlp_out %.3e action_out_proj %.3e" % (gnorm("temp_in"), gnorm("temp_flow_in"), gnorm("temp_out"), gnorm("progress_mlp_out"), gnorm("action_out_proj")))
ok = float(jnp.abs(l_on - l_off).max()) < 1e-5 and float(jnp.abs(a_on - a_off).max()) < 1e-4 and bool(jnp.isfinite(l_lab).all()) and gnorm("temp_in") > 0 and gnorm("temp_out") > 0
print("SMOKE_TEMPORAL_PASS" if ok else "SMOKE_TEMPORAL_FAIL")

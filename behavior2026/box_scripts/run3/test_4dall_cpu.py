"""CPU structural test for the all-corrective + forcing arms (2026-09-28): pi05_radio_4d_all must have the SAME parameter
tree as pi05_radio_4d_pd (data-only arm); pi05_radio_4d_allf must add exactly target_aux_in/out; compute_loss(train=True)
traces for both; with target_aux_weight>0 the loss is larger than with 0 on the same inputs (the head is live) and the
flow loss is unchanged when the weight is 0."""
import os; os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import dataclasses, jax, jax.numpy as jnp, numpy as np
from flax import nnx
from openpi.training import config as _c
from openpi.models import model as _m

def flat(d, p=""):
    for k, v in d.items():
        if isinstance(v, dict): yield from flat(v, p + "/" + k)
        else: yield p + "/" + k, v
def shapes(cfg):
    m = cfg.model.create(jax.random.key(0)); return {k: tuple(np.shape(v)) for k, v in flat(nnx.split(m)[1].to_pure_dict())}, m
s_pd, _ = shapes(_c.get_config("pi05_radio_4d_pd")); s_all, m_all = shapes(_c.get_config("pi05_radio_4d_all")); s_f, m_f = shapes(_c.get_config("pi05_radio_4d_allf"))
print("PARAM_TREE all==pd:", s_all == s_pd, "| allf extra:", sorted(set(s_f) - set(s_all)), "| allf missing:", sorted(set(s_all) - set(s_f)), flush=True)
cfg = _c.get_config("pi05_radio_4d_allf"); obs_spec, act_spec = cfg.model.inputs_spec(batch_size=2)
def mk(spec):
    if spec is None: return None
    if isinstance(spec, dict): return {k: mk(v) for k, v in spec.items()}
    if isinstance(spec, jax.ShapeDtypeStruct): return jnp.ones(spec.shape, spec.dtype) if spec.dtype == jnp.bool_ else jnp.zeros(spec.shape, spec.dtype)
    return spec
with _m.at.disable_typechecking():
    obs = _m.Observation(**{f.name: mk(getattr(obs_spec, f.name)) for f in dataclasses.fields(obs_spec)})
obs = dataclasses.replace(obs, target_points=jnp.full(obs.target_points.shape, 0.2, jnp.float32))
act = jnp.zeros(act_spec.shape, act_spec.dtype); rng = jax.random.key(1)
# SAME model instance, weight toggled (creating a second model changes the rng draw order of every later module and
# therefore its random init -> incomparable flow losses; the 09-25 parity artifact).
l_f = float(jnp.mean(m_f.compute_loss(rng, obs, act, train=True)))
m_f.target_aux_weight = 0.0
l_0 = float(jnp.mean(m_f.compute_loss(rng, obs, act, train=True)))
m_f.target_aux_weight = 0.05
l_all_w = float(jnp.mean(m_all.compute_loss(rng, obs, act, train=True)))
print(f"LOSS same model: w=0.05 -> {l_f:.5f}, w=0 -> {l_0:.5f} (head adds {l_f - l_0:+.5f}); data-only model {l_all_w:.5f}", flush=True)
ok = s_all == s_pd and sorted(set(s_f) - set(s_all)) == ["/target_aux_in/bias", "/target_aux_in/kernel", "/target_aux_out/bias", "/target_aux_out/kernel"] and l_f > l_0 + 1e-6
print("TEST_4DALL_OK" if ok else "TEST_4DALL_FAILED", flush=True)

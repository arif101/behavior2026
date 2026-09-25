"""CPU structural test of the pointer-dropout arm (no GPU, no data): (1) pi05_radio_4d_pd creates the SAME parameter
tree as pi05_radio_4d (no new params -> the FULL warm start + missing_regex apply unchanged); (2) compute_loss(train=True)
and sample_actions trace (jax.eval_shape) on spec-shaped inputs, i.e. the dropout / noise / anchor-follow code paths are
shape-correct; (3) the anchor-follow rule: mask True -> anchor2 == EE_R + pointer_R; mask False -> anchor2 == EE_R."""
import os; os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import dataclasses, jax, jax.numpy as jnp, numpy as np
from flax import nnx
from openpi.training import config as _c
from openpi.models import model as _m

def flat(d, p=""):
    for k, v in d.items():
        if isinstance(v, dict): yield from flat(v, p + "/" + k)
        else: yield p + "/" + k, v

cfg_pd = _c.get_config("pi05_radio_4d_pd"); cfg_4d = _c.get_config("pi05_radio_4d")
mc = cfg_pd.model; print("flags:", mc.pointer_drop_p, mc.pointer_serve_noise_std, mc.pointer_anchor_follow, "| 4d:", cfg_4d.model.pointer_drop_p, cfg_4d.model.pointer_anchor_follow, flush=True)
model = mc.create(jax.random.key(0))
shapes_pd = {k: tuple(np.shape(v)) for k, v in flat(nnx.split(model)[1].to_pure_dict())}
m4 = cfg_4d.model.create(jax.random.key(0)); shapes_4d = {k: tuple(np.shape(v)) for k, v in flat(nnx.split(m4)[1].to_pure_dict())}; del m4
print("PARAM_TREE", "identical" if shapes_pd == shapes_4d else f"DIFFERS only_pd={sorted(set(shapes_pd)-set(shapes_4d))[:5]} only_4d={sorted(set(shapes_4d)-set(shapes_pd))[:5]}", flush=True)
obs_spec, act_spec = mc.inputs_spec(batch_size=2)
def mk(spec):
    if spec is None: return None
    if isinstance(spec, dict): return {k: mk(v) for k, v in spec.items()}
    if isinstance(spec, jax.ShapeDtypeStruct): return jnp.ones(spec.shape, spec.dtype) if spec.dtype == jnp.bool_ else jnp.zeros(spec.shape, spec.dtype)
    return spec
with _m.at.disable_typechecking() if hasattr(_m, "at") else __import__("contextlib").nullcontext():
    obs = _m.Observation(**{f.name: mk(getattr(obs_spec, f.name)) for f in dataclasses.fields(obs_spec)})
act = jnp.zeros(act_spec.shape, act_spec.dtype)
# anchor-follow rule on concrete values (no tracing): EE_R = anchors[:,0]; pointer_R = target_points[:,1]
a = jnp.array([[[0.5, 0.1, 0.9], [0.5, -0.1, 0.9], [0.0, 0.0, 0.0]]] * 2, jnp.float32)
tp = jnp.array([[[0.1, 0.2, 0.3], [0.05, 0.06, 0.07]]] * 2, jnp.float32); tpm = jnp.array([[True, True], [False, False]])
o2 = dataclasses.replace(obs, anchors=a, target_points=tp, target_points_mask=tpm); o3 = model._follow_anchor(o2)
exp0 = a[0, 0] + tp[0, 1]; ok = bool(jnp.allclose(o3.anchors[0, 2], exp0) and jnp.allclose(o3.anchors[1, 2], a[1, 0]))
print("ANCHOR_FOLLOW", "OK" if ok else "WRONG", np.asarray(o3.anchors[:, 2]).round(3).tolist(), flush=True)
ls = jax.eval_shape(lambda: model.compute_loss(jax.random.key(1), obs, act, train=True)); print("LOSS_TRACE_OK", ls.shape, ls.dtype, flush=True)
ss = jax.eval_shape(lambda: model.sample_actions(jax.random.key(2), obs, num_steps=2)); print("SAMPLE_TRACE_OK", ss.shape, flush=True)
print("TEST_4DPD_OK", flush=True)

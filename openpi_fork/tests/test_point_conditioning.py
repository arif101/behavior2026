"""CPU tests for AdaLN 3D-point conditioning of the pi05 action expert.

Run (from repo root, no GPU needed):

    PYTHONPATH=src:packages/openpi-client/src JAX_PLATFORMS=cpu \
        .venv/bin/python -m pytest tests/test_point_conditioning.py -v

or as a plain script:

    PYTHONPATH=src:packages/openpi-client/src JAX_PLATFORMS=cpu \
        .venv/bin/python tests/test_point_conditioning.py

Covers:
  (a) forward-pass shapes with/without point_conditioning
  (b) zero-init equivalence: point_conditioning=True at init is (bit-)identical to False
  (c) gradients flow to the point-MLP / null-token / stage params when enabled
  (d) the validity-mask path (invalid arms use the learned null token, points are ignored)
  (e) B1KInputs packing (absent key -> zeros + invalid mask sentinel)
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.configs.robots.base_config import ObservationConfig
from openpi.configs.robots.base_config import RobotConfig
from openpi.configs.robots.base_config import StateActionConfig
import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
from openpi.policies.b1k_policy import B1KInputs

ACTION_DIM = 8
ACTION_HORIZON = 4


def make_config(**kwargs) -> pi0_config.Pi0Config:
    return pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=ACTION_DIM,
        action_horizon=ACTION_HORIZON,
        max_token_len=16,
        pi05=True,
        dtype="float32",
        **kwargs,
    )


def make_model(config: pi0_config.Pi0Config):
    return config.create(jax.random.key(0))


def _flat_params(model) -> dict:
    flat = nnx.state(model, nnx.Param).flat_state()
    items = flat.items() if hasattr(flat, "items") else flat
    return {tuple(str(p) for p in path): vs for path, vs in items}


def _obs_with_points(obs, points, mask, stages=None):
    return obs.replace(
        target_points=jnp.asarray(points, dtype=jnp.float32),
        target_points_mask=jnp.asarray(mask, dtype=jnp.bool_),
        stage_tokens=None if stages is None else jnp.asarray(stages, dtype=jnp.int32),
    )


def _adarms_cond(model, obs, batch_size=1):
    """Returns the adaRMS conditioning vector produced by embed_suffix (time + point/stage extras)."""
    obs = _model.preprocess_observation(None, obs, train=False)
    noisy_actions = jnp.zeros((batch_size, ACTION_HORIZON, ACTION_DIM), dtype=jnp.float32)
    timestep = jnp.full((batch_size,), 0.5, dtype=jnp.float32)
    _, _, _, adarms_cond = model.embed_suffix(obs, noisy_actions, timestep)
    return adarms_cond


# ---------------------------------------------------------------- (a) shapes


def test_forward_shapes():
    cfg_off = make_config()
    cfg_on = make_config(point_conditioning=True, stage_conditioning=True)
    model_off = make_model(cfg_off)
    model_on = make_model(cfg_on)

    obs = cfg_off.fake_obs(batch_size=1)
    noise = jax.random.normal(jax.random.key(2), (1, ACTION_HORIZON, ACTION_DIM))

    acts_off = model_off.sample_actions(jax.random.key(1), obs, num_steps=2, noise=noise)
    assert acts_off.shape == (1, ACTION_HORIZON, ACTION_DIM)

    # Conditioned model, no points provided (sentinel path).
    acts_on = model_on.sample_actions(jax.random.key(1), obs, num_steps=2, noise=noise)
    assert acts_on.shape == (1, ACTION_HORIZON, ACTION_DIM)

    # Conditioned model, points provided.
    obs_pts = _obs_with_points(
        obs, [[[0.1, -0.2, 0.3], [0.0, 0.5, -0.4]]], [[True, True]], stages=[[3, 7]]
    )
    acts_on_pts = model_on.sample_actions(jax.random.key(1), obs_pts, num_steps=2, noise=noise)
    assert acts_on_pts.shape == (1, ACTION_HORIZON, ACTION_DIM)

    # Conditioning vector has action-expert width.
    width = 64  # gemma "dummy" width
    assert _adarms_cond(model_on, obs_pts).shape == (1, width)
    assert _adarms_cond(model_off, obs).shape == (1, width)
    print("PASS test_forward_shapes")


# ------------------------------------------------- (b) zero-init equivalence


def test_stock_params_identical():
    """All stock parameters must be identical between conditioned/unconditioned models (same seed):
    the new modules are created after the stock ones, so the rng stream for stock params is untouched."""
    model_off = make_model(make_config())
    model_on = make_model(make_config(point_conditioning=True, stage_conditioning=True))
    p_off = _flat_params(model_off)
    p_on = _flat_params(model_on)
    extra = set(p_on) - set(p_off)
    assert all(any(k in "/".join(path) for k in ("point_", "stage_")) for path in extra), extra
    for path, vs in p_off.items():
        assert jnp.array_equal(vs.value, p_on[path].value), f"stock param differs: {path}"
    # Zero-init contract for the identity start.
    assert jnp.all(p_on[("point_mlp_out", "kernel")].value == 0)
    assert jnp.all(p_on[("point_mlp_out", "bias")].value == 0)
    assert jnp.all(p_on[("point_null_embed",)].value == 0)
    assert jnp.all(p_on[("stage_proj", "kernel")].value == 0)
    assert jnp.all(p_on[("stage_proj", "bias")].value == 0)
    print("PASS test_stock_params_identical")


def test_zero_init_equivalence():
    cfg_off = make_config()
    cfg_on = make_config(point_conditioning=True, stage_conditioning=True)
    model_off = make_model(cfg_off)
    model_on = make_model(cfg_on)

    obs = cfg_off.fake_obs(batch_size=1)
    obs_pts = _obs_with_points(
        obs, [[[0.1, -0.2, 0.3], [0.0, 0.5, -0.4]]], [[True, False]], stages=[[3, 7]]
    )
    noise = jax.random.normal(jax.random.key(2), (1, ACTION_HORIZON, ACTION_DIM))
    actions = cfg_off.fake_act(batch_size=1)

    acts_off = model_off.sample_actions(jax.random.key(1), obs, num_steps=2, noise=noise)
    acts_on_none = model_on.sample_actions(jax.random.key(1), obs, num_steps=2, noise=noise)
    acts_on_pts = model_on.sample_actions(jax.random.key(1), obs_pts, num_steps=2, noise=noise)

    d_none = float(jnp.max(jnp.abs(acts_off - acts_on_none)))
    d_pts = float(jnp.max(jnp.abs(acts_off - acts_on_pts)))
    print(f"sample_actions max|diff|: no-points={d_none:.3e}, with-points={d_pts:.3e}")
    assert d_none == 0.0, f"zero-init equivalence broken (no points): {d_none}"
    assert d_pts == 0.0, f"zero-init equivalence broken (with points): {d_pts}"

    # compute_loss equivalence, eval and train mode (train also exercises the point-noise branch, which
    # must not perturb the output at zero-init and must not disturb the stock rng stream).
    for train in (False, True):
        loss_off = model_off.compute_loss(jax.random.key(3), obs, actions, train=train)
        loss_on = model_on.compute_loss(jax.random.key(3), obs_pts, actions, train=train)
        d_loss = float(jnp.max(jnp.abs(loss_off - loss_on)))
        print(f"compute_loss(train={train}) max|diff|: {d_loss:.3e}")
        assert d_loss == 0.0, f"zero-init loss equivalence broken (train={train}): {d_loss}"
    print("PASS test_zero_init_equivalence")


# ------------------------------------------------------- (c) gradient flow


def _perturb_adarms_dense(model, scale=0.05):
    """Simulates a warm-started checkpoint: the adaRMS modulation Dense layers in gemma are zero-init in
    this fork, so at a *fresh* init no gradient flows into the conditioning vector at all (true for the
    time MLP as well). Training always warm-starts from pi05_base where these are trained, so we perturb
    them to nonzero before checking gradient flow."""
    state = nnx.state(model)
    flat = state.flat_state()
    items = flat.items() if hasattr(flat, "items") else flat
    rng = jax.random.key(7)
    n = 0
    for path, vs in items:
        if any("Dense_0" in str(p) for p in path) and str(path[-1]) == "kernel":
            rng, sub = jax.random.split(rng)
            vs.value = scale * jax.random.normal(sub, vs.value.shape, vs.value.dtype)
            n += 1
    assert n > 0, "no adaRMS Dense kernels found to perturb"
    nnx.update(model, state)


def test_gradient_flow():
    cfg = make_config(point_conditioning=True, stage_conditioning=True)
    model = make_model(cfg)
    _perturb_adarms_dense(model)
    # Also move the point-MLP output layer off zero so gradient can reach point_mlp_in.
    model.point_mlp_out.kernel.value = 0.05 * jax.random.normal(
        jax.random.key(8), model.point_mlp_out.kernel.value.shape
    )

    obs = cfg.fake_obs(batch_size=1)
    actions = cfg.fake_act(batch_size=1)

    def grads_for(obs_in):
        def loss_fn(m):
            return jnp.mean(m.compute_loss(jax.random.key(3), obs_in, actions, train=False))

        return _flat_params_from_state(nnx.grad(loss_fn)(model))

    def _flat_params_from_state(state):
        flat = state.flat_state()
        items = flat.items() if hasattr(flat, "items") else flat
        return {tuple(str(p) for p in path): vs for path, vs in items}

    def norm(g, *names):
        return float(
            sum(jnp.sum(jnp.abs(vs.value)) for path, vs in g.items() if any(n in "/".join(path) for n in names))
        )

    # Valid points: gradient must flow to the point MLP; the null token is unused (zero grad).
    obs_valid = _obs_with_points(obs, [[[0.1, -0.2, 0.3], [0.0, 0.5, -0.4]]], [[True, True]], stages=[[3, 7]])
    g = grads_for(obs_valid)
    g_in, g_out = norm(g, "point_mlp_in"), norm(g, "point_mlp_out")
    g_null, g_stage = norm(g, "point_null_embed"), norm(g, "stage_proj")
    print(f"valid-mask grads: point_mlp_in={g_in:.3e} point_mlp_out={g_out:.3e} null={g_null:.3e} stage_proj={g_stage:.3e}")
    assert g_in > 0, "no gradient to point_mlp_in with valid points"
    assert g_out > 0, "no gradient to point_mlp_out with valid points"
    assert g_null == 0.0, "null token received gradient despite valid mask"
    assert g_stage > 0, "no gradient to stage projection"

    # Invalid points: gradient must flow to the null token instead; the point MLP is unused.
    obs_invalid = _obs_with_points(obs, [[[0.1, -0.2, 0.3], [0.0, 0.5, -0.4]]], [[False, False]], stages=[[3, 7]])
    g = grads_for(obs_invalid)
    g_in, g_out, g_null = norm(g, "point_mlp_in"), norm(g, "point_mlp_out"), norm(g, "point_null_embed")
    print(f"invalid-mask grads: point_mlp_in={g_in:.3e} point_mlp_out={g_out:.3e} null={g_null:.3e}")
    assert g_null > 0, "no gradient to null token with invalid mask"
    assert g_in == 0.0, "point_mlp_in received gradient despite invalid mask"
    assert g_out == 0.0, "point_mlp_out received gradient despite invalid mask"
    print("PASS test_gradient_flow")


# ------------------------------------------------------ (d) invalid-mask path


def test_invalid_mask_path():
    cfg = make_config(point_conditioning=True)
    model = make_model(cfg)
    # Move the point pathway off zero-init so behavior differences are observable.
    model.point_mlp_out.kernel.value = 0.1 * jax.random.normal(
        jax.random.key(9), model.point_mlp_out.kernel.value.shape
    )
    model.point_null_embed.value = 0.1 * jax.random.normal(jax.random.key(10), model.point_null_embed.value.shape)

    obs = cfg.fake_obs(batch_size=1)
    p1 = [[[0.1, -0.2, 0.3], [0.0, 0.5, -0.4]]]
    p2 = [[[-0.3, 0.1, 0.2], [0.4, -0.1, 0.6]]]

    c_valid_p1 = _adarms_cond(model, _obs_with_points(obs, p1, [[True, True]]))
    c_valid_p2 = _adarms_cond(model, _obs_with_points(obs, p2, [[True, True]]))
    c_invalid_p1 = _adarms_cond(model, _obs_with_points(obs, p1, [[False, False]]))
    c_invalid_p2 = _adarms_cond(model, _obs_with_points(obs, p2, [[False, False]]))
    # No points at all (note: fake_obs on a point-conditioned config includes spec points, so strip them).
    c_none = _adarms_cond(model, obs.replace(target_points=None, target_points_mask=None))
    c_sentinel = _adarms_cond(model, _obs_with_points(obs, np.zeros((1, 2, 3)), [[False, False]]))

    # Valid points influence the conditioning vector, and different points differ.
    assert float(jnp.max(jnp.abs(c_valid_p1 - c_valid_p2))) > 0, "points ignored despite valid mask"
    # Invalid mask: the point VALUES must not matter (null token used).
    assert jnp.array_equal(c_invalid_p1, c_invalid_p2), "point values leaked through invalid mask"
    # Invalid differs from valid (null token vs encoded point).
    assert float(jnp.max(jnp.abs(c_valid_p1 - c_invalid_p1))) > 0
    # Absent points == explicit zeros + invalid mask (the serve-side sentinel contract).
    assert jnp.array_equal(c_none, c_sentinel), "absent-points fallback != zeros+invalid sentinel"
    # Per-arm masking: arm 0 valid / arm 1 invalid differs from both all-valid and all-invalid.
    c_mixed = _adarms_cond(model, _obs_with_points(obs, p1, [[True, False]]))
    assert float(jnp.max(jnp.abs(c_mixed - c_valid_p1))) > 0
    assert float(jnp.max(jnp.abs(c_mixed - c_invalid_p1))) > 0
    print("PASS test_invalid_mask_path")


# ------------------------------------------------------ (e) input transform


def _make_robot_config() -> RobotConfig:
    return RobotConfig(
        name="test",
        robot_type="test",
        observations={
            "image_0": ObservationConfig(
                name="image_0", obs_key="test::image_0", dataset_key="obs.image", resolution=[224, 224]
            )
        },
        action_key="action",
        action_dim=4,
        action=[StateActionConfig(name="joints", indices=[0, 1, 2, 3])],
        proprio=[StateActionConfig(name="joints", indices=[0, 1, 2, 3])],
    )


def test_b1k_inputs_packing():
    robot_config = _make_robot_config()
    base_data = {
        "observation/state": np.random.rand(4).astype(np.float32),
        "observation/image_0": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }

    # Flag off: no new keys (control arms stay exactly stock).
    tf_off = B1KInputs(model_type=_model.ModelType.PI05, robot_config=robot_config)
    out = tf_off(dict(base_data))
    assert "target_points" not in out
    assert "target_points_mask" not in out
    assert "stage_tokens" not in out

    tf_on = B1KInputs(
        model_type=_model.ModelType.PI05, robot_config=robot_config, point_conditioning=True, stage_conditioning=True
    )

    # Absent key -> zeros + all-invalid mask sentinel.
    out = tf_on(dict(base_data))
    assert out["target_points"].shape == (2, 3)
    assert out["target_points"].dtype == np.float32
    assert np.all(out["target_points"] == 0)
    assert out["target_points_mask"].shape == (2,)
    assert out["target_points_mask"].dtype == np.bool_
    assert not out["target_points_mask"].any()
    assert out["stage_tokens"].shape == (2,)
    assert out["stage_tokens"].dtype == np.int32
    assert np.all(out["stage_tokens"] == 0)

    # Present points, no mask -> assumed valid.
    pts = np.array([[0.1, -0.2, 0.3], [0.0, 0.5, -0.4]], dtype=np.float64)
    out = tf_on({**base_data, "target_points": pts})
    assert out["target_points"].dtype == np.float32
    np.testing.assert_allclose(out["target_points"], pts.astype(np.float32))
    assert out["target_points_mask"].all()

    # Present points + explicit per-arm mask + stage tokens -> passthrough.
    out = tf_on(
        {**base_data, "target_points": pts, "target_points_mask": [True, False], "stage_tokens": [3, 7]}
    )
    assert out["target_points_mask"].tolist() == [True, False]
    assert out["stage_tokens"].tolist() == [3, 7]
    print("PASS test_b1k_inputs_packing")


if __name__ == "__main__":
    test_forward_shapes()
    test_stock_params_identical()
    test_zero_init_equivalence()
    test_gradient_flow()
    test_invalid_mask_path()
    test_b1k_inputs_packing()
    print("\nALL TESTS PASSED")

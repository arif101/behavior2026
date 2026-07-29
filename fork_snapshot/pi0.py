import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        # FOVEATED MEMORY: zero-init out-proj => warm-start tokens EQUAL the 0.02
        # registers exactly; map_tokens_k == 0 creates no params (bit-parity).
        self.map_k = getattr(config, "map_tokens_k", 0)
        if self.map_k > 0:
            self.map_proj_in = nnx.Linear(config.map_token_dim, 256, rngs=rngs)
            self.map_proj_out = nnx.Linear(
                256, paligemma_config.width, kernel_init=nnx.initializers.zeros_init(), rngs=rngs
            )
            self.map_registers = nnx.Param(
                nnx.initializers.normal(0.02)(rngs.params(), (self.map_k, paligemma_config.width))
            )
            # Run-1 aux decode heads (training-only; see patch_aux_losses.py)
            self.aux_map_head_in = nnx.Linear(paligemma_config.width, 64, rngs=rngs)
            self.aux_map_head_out = nnx.Linear(64, 6, rngs=rngs)
            self.aux_img_head_in = nnx.Linear(paligemma_config.width, 64, rngs=rngs)
            self.aux_img_head_out = nnx.Linear(64, 6, rngs=rngs)
            self.aux_stage_in = nnx.Linear(paligemma_config.width, 64, rngs=rngs)
            self.aux_stage_out = nnx.Linear(64, 4, rngs=rngs)
            self.aux_heat = nnx.Linear(paligemma_config.width, 1, rngs=rngs)

        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # AdaLN 3D-point conditioning (arXiv 2606.27663): per-arm target points are embedded and ADDED to the
        # adaRMS conditioning vector that already carries the flow-matching timestep. All new modules are created
        # AFTER the stock modules above so that the rng stream (and thus the init) of every stock parameter is
        # identical to the unconditioned model.
        self.point_conditioning = config.point_conditioning
        self.stage_conditioning = config.stage_conditioning
        self.point_noise_std = config.point_noise_std
        if config.point_conditioning:
            width = action_expert_config.width
            # Sinusoidal pre-encoding dim per coordinate (same mechanism as the timestep embedding).
            self.point_posemb_dim = width // 4
            # Per-arm MLP: fourier(3 coords) -> width/2 -> width/2; shared across arms, arm identity is
            # preserved by concatenation order. The output layer is ZERO-INIT so that training starts exactly
            # at the unconditioned model (identity-init trick).
            self.point_mlp_in = nnx.Linear(3 * self.point_posemb_dim, width // 2, rngs=rngs)
            self.point_mlp_out = nnx.Linear(
                width // 2, width // 2, kernel_init=nnx.initializers.zeros_init(), rngs=rngs
            )
            # Learned per-arm null token used when a target point is invalid ("no target"). Zero-init, so the
            # invalid path also starts exactly at the unconditioned model.
            self.point_null_embed = nnx.Param(jnp.zeros((pi0_config.NUM_POINT_ARMS, width // 2)))
        if config.stage_conditioning:
            width = action_expert_config.width
            # Reserved interface for per-arm discrete stage tokens: embedding table + zero-init projection,
            # injected additively into the same adaRMS conditioning vector.
            self.stage_embed = nnx.Embed(pi0_config.STAGE_VOCAB_SIZE, width // 4, rngs=rngs)
            self.stage_proj = nnx.Linear(
                pi0_config.NUM_POINT_ARMS * (width // 4),
                width,
                kernel_init=nnx.initializers.zeros_init(),
                rngs=rngs,
            )

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]

        # FOVEATED MEMORY: K map tokens after text; ar_mask=False (full prefix
        # attention — perception CAN see the map), tail position = RoPE-safe.
        if getattr(self, "map_k", 0) > 0 and obs.map_tokens is not None:
            _mt = nnx.swish(self.map_proj_in(obs.map_tokens))
            _mt = self.map_proj_out(_mt) + self.map_registers
            tokens.append(_mt)
            input_mask.append(jnp.ones(_mt.shape[:2], dtype=jnp.bool_))
            ar_mask += [False] * _mt.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    def _embed_cond_extras(self, obs: _model.Observation, batch_size: int) -> at.Float[at.Array, "b emb"] | None:
        """Embeds target points (and reserved stage tokens) into an additive term for the adaRMS conditioning
        vector. Returns None when neither point nor stage conditioning is enabled. At init (zero-init output
        layers + zero null tokens) this term is exactly zero, so the model starts as the unconditioned model."""
        n_arms = pi0_config.NUM_POINT_ARMS
        parts = []
        if self.point_conditioning:
            points, mask = obs.target_points, obs.target_points_mask
            if points is None:
                # No points provided: behave as "no target" (zeros + invalid mask -> learned null token).
                points = jnp.zeros((batch_size, n_arms, 3), dtype=jnp.float32)
                mask = jnp.zeros((batch_size, n_arms), dtype=jnp.bool_)
            # Zero out invalid points before encoding so garbage/NaN sentinels can never leak (incl. gradients).
            points = jnp.where(mask[..., None], points, 0.0)
            # Sinusoidal pre-encoding of each coordinate, same mechanism (and periods) as the timestep
            # embedding. Values are meters (gripper-relative), typically well within [-4, 4].
            pos_emb = posemb_sincos(points.reshape(-1), self.point_posemb_dim, min_period=4e-3, max_period=4.0)
            pos_emb = pos_emb.reshape(batch_size * n_arms, 3 * self.point_posemb_dim)
            point_emb = self.point_mlp_in(pos_emb)
            point_emb = nnx.swish(point_emb)
            point_emb = self.point_mlp_out(point_emb)  # zero-init: exactly 0 at start of training
            point_emb = point_emb.reshape(batch_size, n_arms, -1)
            null_emb = jnp.broadcast_to(self.point_null_embed.value[None], point_emb.shape).astype(point_emb.dtype)
            point_emb = jnp.where(mask[..., None], point_emb, null_emb)
            parts.append(point_emb.reshape(batch_size, -1))  # concat arms -> [b, width]
        if self.stage_conditioning:
            stage_tokens = obs.stage_tokens
            if stage_tokens is None:
                stage_tokens = jnp.zeros((batch_size, n_arms), dtype=jnp.int32)
            stage_emb = self.stage_embed(stage_tokens).reshape(batch_size, -1)
            parts.append(self.stage_proj(stage_emb))  # zero-init: exactly 0 at start of training
        if not parts:
            return None
        return sum(parts[1:], start=parts[0])

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
            # AdaLN point/stage conditioning: additive into the SAME adaRMS conditioning vector, so the
            # per-layer gamma/beta/gate derive from it exactly as they already do for the timestep.
            cond_extras = self._embed_cond_extras(obs, batch_size=noisy_actions.shape[0])
            if cond_extras is not None:
                adarms_cond = adarms_cond + cond_extras
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        if self.point_conditioning and train and self.point_noise_std > 0 and observation.target_points is not None:
            # Anti-brittleness: train-time Gaussian noise on target points. Derived via fold_in so the
            # point_conditioning=False rng stream (preprocess/noise/time) stays bit-identical to stock.
            point_rng = jax.random.fold_in(rng, 2606)
            noisy_points = observation.target_points + self.point_noise_std * jax.random.normal(
                point_rng, observation.target_points.shape
            )
            observation = observation.replace(target_points=noisy_points)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)

        # FOVEATED MEMORY aux decodes (patch_aux_losses.py): anti-sink + visual grounding.
        tp_raw = getattr(observation, "target_points", None)
        if self.map_k > 0 and observation.map_tokens is not None and tp_raw is not None:
            b = tp_raw.shape[0]
            tp = tp_raw.reshape(b, -1)[:, :6].astype(jnp.float32)
            tpm = getattr(observation, "target_points_mask", None)
            w = (jnp.repeat(tpm.reshape(b, -1)[:, :2].astype(jnp.float32), 3, axis=-1)
                 if tpm is not None else jnp.ones((b, 6), jnp.float32))
            map_out = prefix_out[:, -self.map_k :, :].mean(axis=1)
            img_out = prefix_out[:, :256, :].mean(axis=1)  # camera 0 = base_0_rgb (head)
            pred_map = self.aux_map_head_out(nnx.swish(self.aux_map_head_in(map_out)))
            pred_img = self.aux_img_head_out(nnx.swish(self.aux_img_head_in(img_out)))
            not_blind = (jnp.abs(observation.map_tokens[:, 0, :]).sum(-1) > 0).astype(jnp.float32)
            e_map = (jnp.abs(pred_map.astype(jnp.float32) - tp) * w).mean(-1) * not_blind
            e_img = (jnp.abs(pred_img.astype(jnp.float32) - tp) * w).mean(-1)
            loss = loss + (0.1 * (e_map + e_img)).astype(loss.dtype)[:, None]
            # aux-v2 (patch_aux_v2.py): stage CE + per-camera patch-heatmap CE
            if observation.stage is not None:
                sl = self.aux_stage_out(nnx.swish(self.aux_stage_in(
                    prefix_out[:, : 3 * 256, :].mean(axis=1)))).astype(jnp.float32)
                lse = jax.nn.log_softmax(sl, axis=-1)
                st = jnp.clip(observation.stage.reshape(-1), 0, 3)
                e_stage = -jnp.take_along_axis(lse, st[:, None], axis=-1)[:, 0]
                loss = loss + (0.05 * e_stage).astype(loss.dtype)[:, None]
            if observation.aux_pixels is not None:
                ap = observation.aux_pixels.astype(jnp.float32)
                e_heat = 0.0
                n_vis = 0.0
                for ci in range(3):  # camera order: base_0(head), left wrist, right wrist
                    logits = self.aux_heat(prefix_out[:, ci * 256 : (ci + 1) * 256, :])
                    logits = logits[..., 0].astype(jnp.float32)
                    u, v, vis = ap[:, 3 * ci], ap[:, 3 * ci + 1], ap[:, 3 * ci + 2]
                    pu = jnp.clip((u * 16).astype(jnp.int32), 0, 15)
                    pv = jnp.clip((v * 16).astype(jnp.int32), 0, 15)
                    tgt = pv * 16 + pu
                    lsm = jax.nn.log_softmax(logits, axis=-1)
                    ce = -jnp.take_along_axis(lsm, tgt[:, None], axis=-1)[:, 0]
                    e_heat = e_heat + ce * vis
                    n_vis = n_vis + vis
                e_heat = e_heat / jnp.maximum(n_vis, 1.0)
                loss = loss + (0.1 * e_heat).astype(loss.dtype)[:, None]
        return loss

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

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
def _sincos3d(xyz: at.Float[at.Array, "b n 3"], n_freqs: int) -> at.Float[at.Array, "b n f"]:
    """Sinusoidal features of base-frame points: wavelengths 0.05 m .. ~4 m (geometric), per axis sin+cos."""
    freqs = 2.0 * jnp.pi / (0.05 * (80.0 ** (jnp.arange(n_freqs, dtype=jnp.float32) / max(n_freqs - 1, 1))))
    ang = xyz[..., :, None] * freqs[None, None, None, :]                                  # [b, n, 3, F]
    return jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1).reshape(*xyz.shape[:-1], -1)


def _sincos4d(xyzt: at.Float[at.Array, "b n 4"], n_freqs: int) -> at.Float[at.Array, "b n f"]:
    """Sinusoidal features of (x, y, z) in meters and t in seconds (t scaled so 8.5 s ~ 4 m)."""
    scaled = jnp.concatenate([xyzt[..., :3], xyzt[..., 3:4] * (4.0 / 8.5)], axis=-1)
    freqs = 2.0 * jnp.pi / (0.05 * (80.0 ** (jnp.arange(n_freqs, dtype=jnp.float32) / max(n_freqs - 1, 1))))
    ang = scaled[..., :, None] * freqs[None, None, None, :]
    return jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1).reshape(*xyzt.shape[:-1], -1)


def _geo_kernel(key_xyz, key_valid, anchors, sigma, n_prefix_q, n_suffix_q):
    """[b, S, 3] key positions (+validity) and [b, A, 3] anchors -> geo [b, T, S, A] for T = n_prefix_q + n_suffix_q
    queries: exp(-|p_s - a|^2 / sigma^2) for the suffix queries, 0 for prefix queries and invalid keys."""
    d2 = jnp.sum((key_xyz[:, :, None, :] - anchors[:, None, :, :]) ** 2, axis=-1)          # [b, S, A]
    k = jnp.exp(-d2 / (sigma * sigma)) * key_valid.astype(jnp.float32)[..., None]           # [b, S, A]
    suffix_rows = jnp.broadcast_to(k[:, None], (k.shape[0], n_suffix_q, k.shape[1], k.shape[2]))
    if n_prefix_q > 0:
        prefix_rows = jnp.zeros((k.shape[0], n_prefix_q, k.shape[1], k.shape[2]), jnp.float32)
        return jnp.concatenate([prefix_rows, suffix_rows], axis=1)
    return suffix_rows


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
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False],
                      use_geo=(int(getattr(config, "geo_anchors", 2)) if getattr(config, "geo_attention", False) else 0),
                      use_key_bias=bool(getattr(config, "hist_tokens", False)),
                      use_mixed_layers=bool(getattr(config, "mixed_layer_attention", False)))
        self.mixed_layer_attention = bool(getattr(config, "mixed_layer_attention", False))
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
        # 4D-ATTENDABLE PERCEPTION (ARCH_4D_ATTENTION_SPEC): A1 zero-init 3D positional encoding on patch tokens,
        # A3 geometric attention kernels (gains live inside gemma.Attention as `geo_gain`, zero-init).
        self.pe3d = bool(getattr(config, "pe3d", False))
        self.geo_attention = bool(getattr(config, "geo_attention", False))
        self.geo_sigma = float(getattr(config, "geo_sigma", 0.15))
        self.pe3d_freqs = int(getattr(config, "pe3d_freqs", 16))
        if self.pe3d:
            self.pe3d_in = nnx.Linear(3 * 2 * self.pe3d_freqs, 256, rngs=rngs)
            self.pe3d_out = nnx.Linear(256, paligemma_config.width, kernel_init=nnx.initializers.zeros,
                                       bias_init=nnx.initializers.zeros, rngs=rngs)
        # A2 4D HISTORY TOKENS: content projection (identity when hist_dim == width: the stored tokens already live in
        # the LLM embedding space), zero-init 4D PE, zero-init registers; near-invisible at init via hist_vis_bias.
        self.hist_tokens = bool(getattr(config, "hist_tokens", False))
        self.hist_k = int(getattr(config, "hist_k", 8)); self.hist_cells = int(getattr(config, "hist_cells", 16))
        self.n_hist = self.hist_k * self.hist_cells if self.hist_tokens else 0
        self.hist_vis_bias = float(getattr(config, "hist_vis_bias", -10.0))
        self.hist_ground_weight = float(getattr(config, "hist_ground_weight", 0.05))
        if self.hist_tokens:
            _hd = int(getattr(config, "hist_dim", 2048)); _w = paligemma_config.width
            _kinit = (lambda key, shape, dtype=jnp.float32: jnp.eye(shape[0], shape[1], dtype=dtype)) if _hd == _w else nnx.initializers.lecun_normal()
            self.hist_in = nnx.Linear(_hd, _w, kernel_init=_kinit, bias_init=nnx.initializers.zeros, rngs=rngs)
            self.hist_pe_in = nnx.Linear(4 * 2 * self.pe3d_freqs, 256, rngs=rngs)
            self.hist_pe_out = nnx.Linear(256, _w, kernel_init=nnx.initializers.zeros, bias_init=nnx.initializers.zeros, rngs=rngs)
            self.hist_registers = nnx.Param(jnp.zeros((self.hist_cells, _w), jnp.float32))
            self.hist_ground_in = nnx.Linear(_w, 128, rngs=rngs)
            self.hist_ground_out = nnx.Linear(128, 3, rngs=rngs)
        self.anti_shortcut = getattr(config, "anti_shortcut", False)
        if self.map_k > 0:
            self.map_proj_in = nnx.Linear(config.map_token_dim, 256, rngs=rngs)
            self.map_proj_out = nnx.Linear(
                256, paligemma_config.width, kernel_init=nnx.initializers.zeros_init(), rngs=rngs
            )
            self.map_registers = nnx.Param(
                nnx.initializers.normal(0.02)(rngs.params(), (self.map_k, paligemma_config.width))
            )
            # ReZero gate: tokens start EXACTLY zero (RMSNorm scales any nonzero token to full
            # magnitude — measured 20x warm-start perturbation without this). Gradient flows
            # through alpha first; the pathway unlocks as training finds it useful.
            self.map_alpha = nnx.Param(jnp.zeros(()))
            # RECONSTRUCTION ANTI-SINK (run1b fix): decode the map tokens OWN raw input from
            # the LLM outputs at their positions. With alpha=0 those outputs carry ZERO input
            # information (content-zero + position-transparent, proven), so this loss sits at
            # chance and its gradient DEMANDS the gate open. Map-exclusive by construction —
            # the aux_map/target decode was satisfiable from image context (measured alpha
            # stall at ~3e-4 through 13.4k steps); exact voxel values are not.
            self.map_recon_in = nnx.Linear(paligemma_config.width, 128, rngs=rngs)
            self.map_recon_out = nnx.Linear(128, config.map_token_dim, rngs=rngs)
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
        # Continuous progress conditioning: sinusoidal scalar -> zero-init MLP -> adaRMS (mirrors the
        # point/stage terms; zero-init output => exactly the unconditioned model at start of training).
        self.progress_conditioning = getattr(config, "progress_conditioning", False)
        if self.progress_conditioning:
            width = action_expert_config.width
            self.progress_posemb_dim = width // 4
            self.progress_mlp_in = nnx.Linear(self.progress_posemb_dim, width // 2, rngs=rngs)
            self.progress_mlp_out = nnx.Linear(
                width // 2, width, kernel_init=nnx.initializers.zeros_init(), rngs=rngs
            )

        # STAGE/PROGRESS prediction head (patch_stage_head.py): 2-layer MLP (hidden 256,
        # gelu) over mean-pooled action-expert suffix features. Created AFTER every
        # stock/optional module above so the rng stream (and thus the init) of all
        # existing params is unchanged; stage_head=False creates no params at all, so
        # restoring run1b checkpoints is bit-identical. Warm-starting WITH the head
        # needs missing_regex '.*stage_head.*' in the CheckpointWeightLoader.
        self.stage_head = getattr(config, "stage_head", False)
        if self.stage_head:
            self.stage_classes = config.stage_classes
            self.stage_loss_weight = config.stage_loss_weight
            self.progress_loss_weight = config.progress_loss_weight
            self.stage_head_in = nnx.Linear(action_expert_config.width, 256, rngs=rngs)
            self.stage_head_out = nnx.Linear(256, config.stage_classes + 1, rngs=rngs)

        # MAP-GEOMETRY AdaLN conditioning (patch_map_adaln.py): zero-init MLP over the
        # geometry channels of the map tokens, additive into adarms_cond. Created AFTER
        # every stock/optional module above so the rng stream (and thus the init) of all
        # existing params is unchanged; map_geo_conditioning=False creates no params, so
        # restoring run1b checkpoints is bit-identical. Warm-starting WITH the route
        # needs missing_regex '.*map_geo.*' in the CheckpointWeightLoader.
        self.map_geo_conditioning = getattr(config, "map_geo_conditioning", False)
        if self.map_geo_conditioning:
            _mg_width = action_expert_config.width
            self.map_geo_mlp_in = nnx.Linear(6 * config.map_token_dim, _mg_width // 2, rngs=rngs)
            self.map_geo_mlp_out = nnx.Linear(
                _mg_width // 2, _mg_width, kernel_init=nnx.initializers.zeros_init(), rngs=rngs
            )

        # GT-DEPTH aux head (patch_depth_aux.py): tiny MLP shared across all prefix
        # image tokens -> per-patch log1p(depth). Created AFTER every stock/optional
        # module above so the rng stream (and thus the init) of all existing params is
        # unchanged; depth_aux=False creates no params at all, so restoring run1b
        # checkpoints is bit-identical. Warm-starting WITH the head needs missing_regex
        # '.*depth_aux.*' in the CheckpointWeightLoader.
        self.depth_aux = getattr(config, "depth_aux", False)
        if self.depth_aux:
            self.depth_aux_weight = config.depth_aux_weight
            self.depth_aux_in = nnx.Linear(paligemma_config.width, 64, rngs=rngs)
            self.depth_aux_out = nnx.Linear(64, 1, rngs=rngs)

        # MODALITY DROPOUT (patch_modality_dropout.py): no params — train-only input
        # regularizer; checkpoint restore untouched at any flag value.
        self.modality_dropout_p = getattr(config, "modality_dropout_p", 0.0)

        # TEMPORAL FORCING (pi0_config.temporal_conditioning; research/TEMPORAL_4D_SWEEP §4). Created AFTER every
        # stock/optional module so the rng stream of all existing params is unchanged; False => no params.
        # Route: history gists -> temporal queries -> mean -> temp_out (ZERO-INIT = the gate) -> adaRMS cond.
        # Warm-starting needs missing_regex ".*temp_.*". Liveness readout: ||temp_out.kernel|| over checkpoints.
        self.temporal_conditioning = getattr(config, "temporal_conditioning", False)
        if self.temporal_conditioning:
            _tw = action_expert_config.width
            self.temporal_k = config.temporal_k
            self.temporal_nq = config.temporal_queries
            self.temporal_flow_weight = config.temporal_flow_weight
            self.temporal_stage_weight = config.temporal_stage_weight
            self.temp_in = nnx.Linear(config.temporal_gist_dim, _tw, rngs=rngs)
            self.temp_slot_embed = nnx.Param(nnx.initializers.normal(0.02)(rngs.params(), (config.temporal_k + 1, _tw)))
            self.temp_queries = nnx.Param(nnx.initializers.normal(0.02)(rngs.params(), (config.temporal_queries, _tw)))
            self.temp_kv_norm = nnx.RMSNorm(_tw, rngs=rngs)
            self.temp_attn_0 = nnx.MultiHeadAttention(num_heads=config.temporal_heads, in_features=_tw, decode=False, rngs=rngs)
            self.temp_attn_1 = nnx.MultiHeadAttention(num_heads=config.temporal_heads, in_features=_tw, decode=False, rngs=rngs)
            self.temp_ln_0 = nnx.RMSNorm(_tw, rngs=rngs)
            self.temp_ln_1 = nnx.RMSNorm(_tw, rngs=rngs)
            self.temp_mlp_0_in = nnx.Linear(_tw, 2 * _tw, rngs=rngs)
            self.temp_mlp_0_out = nnx.Linear(2 * _tw, _tw, rngs=rngs)
            self.temp_mlp_1_in = nnx.Linear(_tw, 2 * _tw, rngs=rngs)
            self.temp_mlp_1_out = nnx.Linear(2 * _tw, _tw, rngs=rngs)
            self.temp_feat_norm = nnx.RMSNorm(_tw, rngs=rngs)   # unit-RMS pooled feature -> the gate's gradient scales with dL/dcond, not |feat|
            self.temp_out = nnx.Linear(_tw, _tw, kernel_init=nnx.initializers.zeros_init(), rngs=rngs)  # THE GATE: exactly 0 at init
            # pre-gate supervision heads (training-only): change targets over the K offsets + stage/progress readout
            self.temp_flow_in = nnx.Linear(_tw, 256, rngs=rngs)
            self.temp_flow_out = nnx.Linear(256, config.temporal_k * 9, rngs=rngs)
            self.temp_stage_in = nnx.Linear(_tw, 128, rngs=rngs)
            self.temp_stage_out = nnx.Linear(128, getattr(config, "stage_classes", 4) + 1, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"], at.Bool[at.Array, "b s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for _ci, name in enumerate(obs.images):
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            if self.pe3d and obs.patch_xyz is not None:
                # A1: zero-init 3D positional encoding of the patch's base-frame point (invalid patches -> 0 encoding)
                _xyz = obs.patch_xyz[:, _ci].astype(jnp.float32)                       # [b, 256, 3]
                _val = obs.patch_valid[:, _ci].astype(jnp.float32)[..., None]          # [b, 256, 1]
                _pe = self.pe3d_out(nnx.swish(self.pe3d_in(_sincos3d(_xyz, self.pe3d_freqs)))) * _val
                image_tokens = image_tokens + _pe.astype(image_tokens.dtype)

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
            _mt = self.map_alpha * (self.map_proj_out(_mt) + self.map_registers)
            tokens.append(_mt)
            import os as _os
            _viz = _os.environ.get("MAP_TOKENS_INVISIBLE", "0") != "1"
            input_mask.append(jnp.full(_mt.shape[:2], _viz, dtype=jnp.bool_))
            ar_mask += [False] * _mt.shape[1]
            _n_map = _mt.shape[1]
        else:
            _n_map = 0
        _n_hist = 0
        if self.hist_tokens and obs.history_tokens is not None:
            _b = obs.history_tokens.shape[0]
            _ht = obs.history_tokens.astype(jnp.float32).reshape(_b, self.n_hist, -1)
            _hx = obs.history_xyz.astype(jnp.float32).reshape(_b, self.n_hist, 3)
            _hdt = jnp.repeat(obs.history_dt.astype(jnp.float32), self.hist_cells, axis=1)[..., None]      # [b, n_hist, 1]
            _pe = self.hist_pe_out(nnx.swish(self.hist_pe_in(_sincos4d(jnp.concatenate([_hx, _hdt], -1), self.pe3d_freqs))))
            _reg = jnp.tile(self.hist_registers.value[None], (1, self.hist_k, 1))
            _hv = obs.history_valid.reshape(_b, self.n_hist)
            _tok = (self.hist_in(_ht) + _pe + _reg) * _hv.astype(jnp.float32)[..., None]
            tokens.append(_tok.astype(tokens[0].dtype))
            input_mask.append(_hv)
            ar_mask += [False] * self.n_hist
            _n_hist = self.n_hist
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        # POSITION-TRANSPARENT map tokens: they contribute 0 to the RoPE position cumsum, so
        # suffix (and nothing else) keeps IDENTICAL positions with or without them. Measured:
        # naive appending shifted every suffix position by K -> 3x warm-start loss perturbation.
        pos_weight = input_mask
        if _n_map + _n_hist:
            pos_weight = input_mask.at[:, -(_n_map + _n_hist):].set(False)
        return tokens, input_mask, ar_mask, pos_weight

    def _map_out(self, prefix_out):
        """Map-token outputs (the K tokens before the history tail)."""
        if self.n_hist:
            return prefix_out[:, -(self.map_k + self.n_hist) : -self.n_hist, :]
        return prefix_out[:, -self.map_k :, :]

    def _hist_attn_mask(self, attn_mask, n_prefix):
        """History tokens attend ONLY among themselves (so the temporal-grounding aux cannot copy the current frame)."""
        if not self.n_hist:
            return attn_mask
        rows = jnp.arange(attn_mask.shape[1]); cols = jnp.arange(attn_mask.shape[2])
        is_hist_row = (rows >= n_prefix - self.n_hist) & (rows < n_prefix)
        is_hist_col = (cols >= n_prefix - self.n_hist) & (cols < n_prefix)
        block = is_hist_row[:, None] & (~is_hist_col)[None, :]
        return jnp.where(block[None], False, attn_mask)

    def _hist_key_bias(self, batch, n_prefix, n_suffix):
        if not self.n_hist:
            return None
        cols = jnp.arange(n_prefix + n_suffix)
        is_hist = (cols >= n_prefix - self.n_hist) & (cols < n_prefix)
        return jnp.broadcast_to((is_hist.astype(jnp.float32) * self.hist_vis_bias)[None], (batch, n_prefix + n_suffix))

    def prefix_key_geometry(self, obs: _model.Observation, n_prefix: int):
        """A3: base-frame 3D position + validity for every prefix key, aligned with embed_prefix's layout
        (images in obs.images order, 256 patches each; text/map/other tokens -> invalid)."""
        b = obs.patch_xyz.shape[0]
        xyz = obs.patch_xyz.astype(jnp.float32).reshape(b, -1, 3)
        val = obs.patch_valid.reshape(b, -1)
        n_img = xyz.shape[1]
        n_tail = self.n_hist if (self.n_hist and obs.history_xyz is not None) else 0
        pad = max(n_prefix - n_img - n_tail, 0)
        parts_x = [xyz, jnp.zeros((b, pad, 3), jnp.float32)]; parts_v = [val, jnp.zeros((b, pad), bool)]
        if n_tail:
            parts_x.append(obs.history_xyz.astype(jnp.float32).reshape(b, n_tail, 3)); parts_v.append(obs.history_valid.reshape(b, n_tail))
        xyz = jnp.concatenate(parts_x, axis=1)[:, :n_prefix]
        val = jnp.concatenate(parts_v, axis=1)[:, :n_prefix]
        return xyz, val

    def compute_head_gist(self, image: at.Float[at.Array, "b h w c"]) -> at.Float[at.Array, "b d"]:
        """TEMPORAL FORCING serve helper: [-1, 1] image at model resolution -> gist = mean of the 256 projected
        SigLIP tokens (precompute_gists.py contract). The tower is frozen in the temporal arm, so this equals the
        precomputed training gists."""
        toks, _ = self.PaliGemma.img(image, train=False)
        return toks.astype(jnp.float32).mean(axis=1)

    def _temporal_summary(self, obs: _model.Observation, batch_size: int):
        """History gists [b, K+1, D] (oldest ... current) + validity -> pooled query feature [b, W] (pre-gate) and
        the gated adaRMS term [b, W] (zero at init). Missing history => zero gists, only the current slot attended."""
        K = self.temporal_k; W = self.temp_queries.value.shape[-1]; Q = self.temporal_nq
        g, m = obs.history_gists, obs.history_mask
        if g is None:
            g = jnp.zeros((batch_size, K + 1, self.temp_in.in_features), jnp.float32)
            m = jnp.zeros((batch_size, K + 1), dtype=jnp.bool_)
        g = g.astype(jnp.float32); m = m.astype(jnp.bool_)
        kv = self.temp_in(g) + self.temp_slot_embed.value[None].astype(jnp.float32)
        kv = jnp.where(m[..., None], kv, 0.0)
        kv = self.temp_kv_norm(kv)
        m_att = m.at[:, -1].set(True)                       # the current slot is always attendable (no NaN softmax)
        mask = m_att[:, None, None, :]                       # [b, 1(heads), 1(q), K+1]
        x = jnp.broadcast_to(self.temp_queries.value[None].astype(jnp.float32), (batch_size, Q, W))
        for attn, ln, mlp_in, mlp_out in ((self.temp_attn_0, self.temp_ln_0, self.temp_mlp_0_in, self.temp_mlp_0_out),
                                          (self.temp_attn_1, self.temp_ln_1, self.temp_mlp_1_in, self.temp_mlp_1_out)):
            x = x + attn(ln(x), kv, kv, mask=mask, deterministic=True)
            x = x + mlp_out(jax.nn.gelu(mlp_in(x)))
        feat = self.temp_feat_norm(x.mean(axis=1))
        return feat, self.temp_out(feat)

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
        # MAP-GEOMETRY AdaLN term (patch_map_adaln.py). Geometry rows [T1,T3,T4,T5,T6,T7]
        # with the target-derived fields of T5/T7 ([28:32], blind_view convention) zeroed;
        # identical for full and blind token streams by construction. tanh(x/4) bounds the
        # mixed feature scales (map_recon convention). Zero-init output => exactly 0 at
        # start of training; all-zero map tokens (the null/no-map convention) produce a
        # learned constant 'blank map' embedding, consistent with the prefix null token.
        if getattr(self, "map_geo_conditioning", False) and obs.map_tokens is not None:
            import numpy as _mg_np
            _mg = obs.map_tokens.astype(jnp.float32)[:, (1, 3, 4, 5, 6, 7), :]
            _mg_mask = _mg_np.ones((6, obs.map_tokens.shape[-1]), _mg_np.float32)
            _mg_mask[3, 28:32] = 0.0
            _mg_mask[5, 28:32] = 0.0
            _mg = _mg * jnp.asarray(_mg_mask)[None]
            _mg = jnp.tanh(_mg / 4.0).reshape(batch_size, -1)
            _mg = self.map_geo_mlp_in(_mg)
            _mg = nnx.swish(_mg)
            parts.append(self.map_geo_mlp_out(_mg))  # zero-init: exactly 0 at start of training
        if getattr(self, "progress_conditioning", False):
            progress = obs.progress
            if progress is None:
                progress = jnp.zeros((batch_size,), dtype=jnp.float32)
            prog_emb = posemb_sincos(progress.reshape(-1).astype(jnp.float32),
                                     self.progress_posemb_dim, min_period=4e-3, max_period=4.0)
            prog_emb = prog_emb.reshape(batch_size, self.progress_posemb_dim)
            prog_emb = nnx.swish(self.progress_mlp_in(prog_emb))
            parts.append(self.progress_mlp_out(prog_emb))  # zero-init: exactly 0 at start of training
        if getattr(self, "temporal_conditioning", False):
            _tf, _tg = self._temporal_summary(obs, batch_size)
            parts.append(_tg)  # zero-init gate: exactly 0 at start of training
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
        preprocess_rng, noise_rng, time_rng, _as_drop, _as_wl, _as_wr = jax.random.split(rng, 6)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        # ANTI-SHORTCUT (patch_antishortcut.py): train-only input-channel dropout.
        if self.anti_shortcut and train:
            import dataclasses as _dc
            _b = observation.state.shape[0]
            if getattr(observation, "target_points", None) is not None:
                _keep = jax.random.bernoulli(_as_drop, 0.6, (_b,))
                _tp = observation.target_points * _keep[:, None, None].astype(observation.target_points.dtype)
                _tpm = observation.target_points_mask
                if _tpm is not None:
                    _tpm = jnp.logical_and(_tpm, _keep[:, None])
                observation = _dc.replace(observation, target_points=_tp, target_points_mask=_tpm)
            _masks = dict(observation.image_masks)
            for _r, _nm in ((_as_wl, "left_wrist_0_rgb"), (_as_wr, "right_wrist_0_rgb")):
                if _nm in _masks:
                    _kc = jax.random.bernoulli(_r, 0.8, _masks[_nm].shape)
                    _masks[_nm] = jnp.logical_and(_masks[_nm], _kc)
            observation = _dc.replace(observation, image_masks=_masks)

        # MODALITY DROPOUT (patch_modality_dropout.py): independent per-stream drop,
        # train-only. Runs BEFORE point noise (noise on a dropped point is discarded by
        # the mask) and BEFORE the loss blocks that read image_masks (GT-depth aux).
        # fold_in(8206) keeps every stock rng stream bit-identical (point-noise
        # precedent). Enable exactly ONE of anti_shortcut / modality_dropout_p.
        if self.modality_dropout_p > 0 and train:
            import dataclasses as _mdc
            _mdp = self.modality_dropout_p
            _mdb = observation.state.shape[0]
            _r_pt, _r_map, _r_wl, _r_wr = jax.random.split(jax.random.fold_in(rng, 8206), 4)
            if observation.target_points_mask is not None:
                _mdkeep = jax.random.bernoulli(_r_pt, 1.0 - _mdp, (_mdb,))
                observation = _mdc.replace(
                    observation,
                    target_points_mask=jnp.logical_and(
                        observation.target_points_mask, _mdkeep[:, None]),
                )
            if observation.map_tokens is not None:
                _mdkeepm = jax.random.bernoulli(_r_map, 1.0 - _mdp, (_mdb,))
                observation = _mdc.replace(
                    observation,
                    map_tokens=observation.map_tokens
                    * _mdkeepm[:, None, None].astype(observation.map_tokens.dtype),
                )
            _mdmasks = dict(observation.image_masks)
            for _mdr, _mdnm in ((_r_wl, "left_wrist_0_rgb"), (_r_wr, "right_wrist_0_rgb")):
                if _mdnm in _mdmasks:
                    _mdkc = jax.random.bernoulli(_mdr, 1.0 - _mdp, _mdmasks[_mdnm].shape)
                    _mdmasks[_mdnm] = jnp.logical_and(_mdmasks[_mdnm], _mdkc)
            observation = _mdc.replace(observation, image_masks=_mdmasks)

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
        prefix_tokens, prefix_mask, prefix_ar_mask, prefix_posw = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        _np_, _ns_ = prefix_tokens.shape[1], suffix_tokens.shape[1]
        _hist_on = bool(self.n_hist and observation.history_tokens is not None)
        if _hist_on:
            attn_mask = self._hist_attn_mask(attn_mask, _np_)
        positions = jnp.cumsum(jnp.concatenate([prefix_posw, suffix_mask], axis=1), axis=1) - 1
        _geo = None
        if self.geo_attention and observation.patch_xyz is not None and observation.anchors is not None:
            _kx, _kv = self.prefix_key_geometry(observation, _np_)
            _kx = jnp.concatenate([_kx, jnp.zeros((_kx.shape[0], _ns_, 3), jnp.float32)], axis=1)   # suffix keys: no position
            _kv = jnp.concatenate([_kv, jnp.zeros((_kv.shape[0], _ns_), bool)], axis=1)
            _geo = _geo_kernel(_kx, _kv, observation.anchors.astype(jnp.float32), self.geo_sigma, _np_, _ns_)
        _kb = self._hist_key_bias(prefix_tokens.shape[0], _np_, _ns_) if _hist_on else None
        if getattr(self, "mixed_layer_attention", False):
            # A4 two-pass training: prefix pass (its own mask/positions) -> cached K/V of every layer -> suffix pass
            # attending to a learned per-layer blend of them. Equivalent to the joint pass at identity init.
            _pm = attn_mask[:, :_np_, :_np_]; _pp = positions[:, :_np_]
            _kb_p = self._hist_key_bias(prefix_tokens.shape[0], _np_, 0) if _hist_on else None
            (prefix_out, _), _kvc = self.PaliGemma.llm([prefix_tokens, None], mask=_pm, positions=_pp, key_bias=_kb_p)
            _sm = attn_mask[:, _np_:, :]; _sp = positions[:, _np_:]
            _geo_s = None if _geo is None else _geo[:, _np_:, :, :]
            (_, suffix_out), _ = self.PaliGemma.llm([None, suffix_tokens], mask=_sm, positions=_sp, kv_cache=_kvc,
                                                    adarms_cond=[None, adarms_cond], geo=_geo_s, key_bias=_kb, mixed_layers=True)
        else:
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond], geo=_geo,
                key_bias=_kb,
            )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        if _hist_on and observation.rail_now is not None and self.hist_ground_weight > 0:
            # A2 temporal grounding: the current rail position (base frame, /0.3 m units) from the HISTORY outputs only
            # (history rows cannot attend to the current frame), masked-mean over valid history tokens.
            _ho = prefix_out[:, -self.n_hist :, :].astype(jnp.float32)
            _hm = observation.history_valid.reshape(_ho.shape[0], self.n_hist).astype(jnp.float32)[..., None]
            _pool = (_ho * _hm).sum(1) / jnp.maximum(_hm.sum(1), 1.0)
            _pred = self.hist_ground_out(nnx.swish(self.hist_ground_in(_pool)))
            _tgt = observation.rail_now.astype(jnp.float32) / 0.3
            _any = (observation.history_valid.reshape(_ho.shape[0], -1).any(-1)).astype(jnp.float32)
            _e = jnp.mean(jnp.square(_pred - _tgt), axis=-1) * _any
            loss = loss + (self.hist_ground_weight * _e).astype(loss.dtype)[:, None]

        # FOVEATED MEMORY aux decodes (patch_aux_losses.py): anti-sink + visual grounding.
        tp_raw = getattr(observation, "target_points", None)
        if self.map_k > 0 and observation.map_tokens is not None and tp_raw is not None:
            b = tp_raw.shape[0]
            tp = tp_raw.reshape(b, -1)[:, :6].astype(jnp.float32)
            tpm = getattr(observation, "target_points_mask", None)
            w = (jnp.repeat(tpm.reshape(b, -1)[:, :2].astype(jnp.float32), 3, axis=-1)
                 if tpm is not None else jnp.ones((b, 6), jnp.float32))
            map_out = self._map_out(prefix_out).mean(axis=1)
            img_out = prefix_out[:, :256, :].mean(axis=1)  # camera 0 = base_0_rgb (head)
            pred_map = self.aux_map_head_out(nnx.swish(self.aux_map_head_in(map_out)))
            pred_img = self.aux_img_head_out(nnx.swish(self.aux_img_head_in(img_out)))
            not_blind = (jnp.abs(observation.map_tokens[:, 0, :]).sum(-1) > 0).astype(jnp.float32)
            e_map = (jnp.abs(pred_map.astype(jnp.float32) - tp) * w).mean(-1) * not_blind
            e_img = (jnp.abs(pred_img.astype(jnp.float32) - tp) * w).mean(-1)
            # reconstruction anti-sink: predict tanh(x/4) of the raw 72-d inputs from the
            # outputs at the map positions (bounded target balances mixed feature scales)
            recon = self.map_recon_out(nnx.swish(self.map_recon_in(
                self._map_out(prefix_out))))
            tgt_n = jnp.tanh(observation.map_tokens.astype(jnp.float32) / 4.0)
            e_recon = jnp.abs(recon.astype(jnp.float32) - tgt_n).mean((-1, -2))
            loss = loss + (0.1 * (e_map + e_img) + 0.1 * e_recon).astype(loss.dtype)[:, None]
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
        # STAGE/PROGRESS head (patch_stage_head.py): pool the action-expert suffix
        # features that feed action_out_proj (mean over the action tokens), one MLP
        # trunk -> stage_classes logits + 1 progress channel. Losses are per-sample
        # scalars broadcast over the horizon dim exactly like the aux losses above;
        # each term is skipped when its label is absent from the batch (serve /
        # unlabeled data), so the head is loss-inert without labels.
        if self.stage_head:
            _sh_feat = suffix_out[:, -self.action_horizon :, :].mean(axis=1)
            _sh = self.stage_head_out(jax.nn.gelu(self.stage_head_in(_sh_feat))).astype(jnp.float32)
            if observation.stage is not None:
                _sh_lsm = jax.nn.log_softmax(_sh[:, : self.stage_classes], axis=-1)
                _sh_st = jnp.clip(observation.stage.reshape(-1).astype(jnp.int32), 0, self.stage_classes - 1)
                _e_stage = -jnp.take_along_axis(_sh_lsm, _sh_st[:, None], axis=-1)[:, 0]
                loss = loss + (self.stage_loss_weight * _e_stage).astype(loss.dtype)[:, None]
            if observation.progress is not None:
                _sh_prog = jax.nn.sigmoid(_sh[:, self.stage_classes])
                _sh_tgt = jnp.clip(observation.progress.reshape(-1).astype(jnp.float32), 0.0, 1.0)
                _e_prog = jnp.square(_sh_prog - _sh_tgt)
                loss = loss + (self.progress_loss_weight * _e_prog).astype(loss.dtype)[:, None]
        # GT-DEPTH aux (patch_depth_aux.py): masked L1 on log1p(metric depth) over the
        # 3x256 prefix image tokens (camera order = image embedding order). Weights:
        # validity (gt > 0.01 m) AND the per-camera image mask (a dropped/absent camera
        # contributes nothing — composes with modality dropout, which mutates
        # image_masks BEFORE this block runs). Per-sample scalar broadcast over the
        # horizon dim like every aux loss; loss-inert when the batch has no labels.
        if self.depth_aux and observation.gt_depth is not None:
            _gd = observation.gt_depth.astype(jnp.float32).reshape(-1, 3, 256)
            _dpred = self.depth_aux_out(jax.nn.gelu(self.depth_aux_in(
                prefix_out[:, : 3 * 256, :])))[..., 0].astype(jnp.float32).reshape(-1, 3, 256)
            _dvalid = (_gd > 0.01).astype(jnp.float32)
            _dcmask = jnp.stack(
                [observation.image_masks[_c].astype(jnp.float32)
                 for _c in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")], axis=1)
            _dw = _dvalid * _dcmask[:, :, None]
            _derr = jnp.abs(_dpred - jnp.log1p(_gd)) * _dw
            _e_depth = _derr.sum((-1, -2)) / jnp.maximum(_dw.sum((-1, -2)), 1.0)
            loss = loss + (self.depth_aux_weight * _e_depth).astype(loss.dtype)[:, None]
        # TEMPORAL FORCING pre-gate supervision (training-only heads): masked MSE on current-minus-past
        # [EE_L, EE_R, button] displacements (0.1 m units) over the K offsets + stage CE / progress MSE readout.
        # The gradient lands on the temporal queries BEFORE the zero-init gate, so the channel cannot stay dead.
        if getattr(self, "temporal_conditioning", False) and observation.hist_flow is not None:
            _tf, _ = self._temporal_summary(observation, batch_shape[0])
            _pred = self.temp_flow_out(jax.nn.gelu(self.temp_flow_in(_tf))).astype(jnp.float32).reshape(-1, self.temporal_k, 9)
            _tgt = observation.hist_flow.astype(jnp.float32) / 0.3   # 0.3 m units: EE flow <= ~0.5, radio flow <= ~1.2 (0.1 m units gave loss ~5 and grad-norm ~40 at init, swamping the clipped policy gradient)
            _fm = observation.hist_flow_mask.astype(jnp.float32) if observation.hist_flow_mask is not None else jnp.ones(_pred.shape[:2], jnp.float32)
            _e_flow = (jnp.square(_pred - _tgt).mean(-1) * _fm).sum(-1) / jnp.maximum(_fm.sum(-1), 1.0)
            loss = loss + (self.temporal_flow_weight * _e_flow).astype(loss.dtype)[:, None]
            if observation.stage is not None:
                _ts = self.temp_stage_out(jax.nn.gelu(self.temp_stage_in(_tf))).astype(jnp.float32)
                _nc = _ts.shape[-1] - 1
                _tlab = jnp.clip(observation.stage.reshape(-1).astype(jnp.int32), 0, _nc - 1)
                _e_ts = -jnp.take_along_axis(jax.nn.log_softmax(_ts[:, :_nc]), _tlab[:, None], axis=-1)[:, 0]
                if observation.progress is not None:
                    _e_ts = _e_ts + jnp.square(jax.nn.sigmoid(_ts[:, -1]) - jnp.clip(observation.progress.reshape(-1).astype(jnp.float32), 0.0, 1.0))
                loss = loss + (self.temporal_stage_weight * _e_ts).astype(loss.dtype)[:, None]
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
        prefix_tokens, prefix_mask, prefix_ar_mask, prefix_posw = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        _hist_on = bool(self.n_hist and observation.history_tokens is not None)
        if _hist_on:
            prefix_attn_mask = self._hist_attn_mask(prefix_attn_mask, prefix_tokens.shape[1])
        positions = jnp.cumsum(prefix_posw, axis=1) - 1
        _kb_p = self._hist_key_bias(prefix_tokens.shape[0], prefix_tokens.shape[1], 0) if _hist_on else None
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions, key_bias=_kb_p)

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
            positions = jnp.sum(prefix_posw, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            _geo = None
            if self.geo_attention and observation.patch_xyz is not None and observation.anchors is not None:
                # keys = [prefix (cached) ; suffix]; queries = suffix only -> [b, suffix_len, prefix+suffix, A]
                _np_, _ns_ = prefix_tokens.shape[1], suffix_tokens.shape[1]
                _kx, _kv = self.prefix_key_geometry(observation, _np_)
                _kx = jnp.concatenate([_kx, jnp.zeros((_kx.shape[0], _ns_, 3), jnp.float32)], axis=1)
                _kv = jnp.concatenate([_kv, jnp.zeros((_kv.shape[0], _ns_), bool)], axis=1)
                _geo = _geo_kernel(_kx, _kv, observation.anchors.astype(jnp.float32), self.geo_sigma, 0, _ns_)
            _kb_s = self._hist_key_bias(suffix_tokens.shape[0], prefix_tokens.shape[1], suffix_tokens.shape[1]) if _hist_on else None
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
                geo=_geo,
                key_bias=_kb_s,
                mixed_layers=getattr(self, "mixed_layer_attention", False),
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

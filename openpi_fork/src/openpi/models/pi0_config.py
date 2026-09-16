import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0

# Number of arms for point/stage conditioning (bimanual: left, right).
NUM_POINT_ARMS = 2
# Vocabulary size of the reserved stage-token embedding table.
STAGE_VOCAB_SIZE = 32


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    # FOVEATED MEMORY: K map soft tokens appended to the prefix after text (0 = off;
    # the K=0 path creates no params and is bit-identical to baseline).
    map_tokens_k: int = 0
    map_token_dim: int = 72
    # Anti-shortcut input-channel dropout in compute_loss (train-only): P(drop point)=0.4,
    # P(drop each wrist cam)=0.2. See patch_antishortcut.py.
    anti_shortcut: bool = False

    # AdaLN 3D-point conditioning (arXiv 2606.27663 interface). When enabled, per-arm gripper-relative 3D
    # target points are embedded (sinusoidal + MLP, zero-init output layer) and added to the SAME adaRMS
    # conditioning vector that already carries the flow-matching timestep. Requires pi05=True. With the
    # default False, the model is exactly the stock pi05 model (no extra params, no behavior change).
    point_conditioning: bool = False
    # Train-time Gaussian noise (meters, std) added to valid target points; anti-brittleness. Only active
    # when point_conditioning=True and train=True.
    point_noise_std: float = 0.02
    # Reserved: per-arm discrete stage tokens injected additively into the same adaRMS conditioning vector
    # (embedding table STAGE_VOCAB_SIZE x width/4 + zero-init projection). Requires pi05=True.
    stage_conditioning: bool = False
    # Continuous episode-progress [0,1] conditioning: sinusoidal-encoded scalar -> zero-init MLP,
    # added to the same adaRMS conditioning vector. Complements the discrete stage token with fine
    # within-stage timing. Requires pi05=True. Default False = stock model, no extra params.
    progress_conditioning: bool = False

    # STAGE/PROGRESS prediction head (patch_stage_head.py): 2-layer MLP (hidden 256,
    # gelu) over mean-pooled action-expert suffix features -> stage_classes logits +
    # 1 sigmoid progress scalar, trained with CE / MSE aux losses. False => no params
    # created, init and checkpoint restore bit-identical to baseline (run1b safe).
    stage_head: bool = False
    stage_classes: int = 4
    stage_loss_weight: float = 0.05
    progress_loss_weight: float = 0.02

    # MAP-GEOMETRY AdaLN conditioning (patch_map_adaln.py): geometry channels of the
    # (8,72) map tokens (rows T1/T3/T4/T6 + T5/T7 with target fields zeroed; T0/T2
    # excluded) -> tanh(x/4) -> MLP (zero-init output) -> additive term on the SAME
    # adaRMS conditioning vector as the timestep + target points. False => no params
    # created, init and checkpoint restore bit-identical to baseline (run1b safe).
    map_geo_conditioning: bool = False
    # GT-DEPTH auxiliary head (patch_depth_aux.py): per-patch log1p(depth) L1 from the
    # prefix image-token features (16x16 per camera), masked on validity + image mask.
    # Depth is a LABEL only (policy input stays RGB-only). False => no params created,
    # init and checkpoint restore bit-identical to baseline (run1b safe).
    depth_aux: bool = False
    depth_aux_weight: float = 0.05
    # MODALITY DROPOUT (patch_modality_dropout.py): train-only independent per-stream
    # drop prob (target_points -> null mask, map_tokens -> zeros/null, each wrist cam
    # -> image_mask False; head cam never dropped). Generalizes anti_shortcut — enable
    # exactly ONE of the two. 0.0 => byte-identical behavior to baseline.
    modality_dropout_p: float = 0.0
    # TEMPORAL FORCING (research/TEMPORAL_4D_SWEEP_2026_09_06 §4; 2026-09-16): K past HEAD-camera gists at
    # chunk-boundary stride (precomputed by the FROZEN SigLIP tower of the warm start, precompute_gists.py; the
    # tower is frozen in the temporal arm so serve-time gists match) -> temporal_queries learnable queries through
    # 2 cross-attention blocks -> mean -> ZERO-INIT projection -> the adaRMS conditioning vector (action expert
    # only, never the VLM prefix). Pre-gate change-prediction head (EE/object flow over the K offsets) + stage/
    # progress readout supervise the pathway (training-only heads). False => no params, bit-identical restore.
    temporal_conditioning: bool = False
    temporal_k: int = 8
    temporal_stride: int = 32
    temporal_gist_dim: int = 2048
    temporal_queries: int = 4
    temporal_heads: int = 8
    temporal_flow_weight: float = 0.2   # with 0.3 m target units the aux term starts ~0.1 (GPU smoke 2026-09-16: 1.0 @ 0.1 m units = loss 5, grad-norm 40)
    temporal_stage_weight: float = 0.2
    pytorch_compile_mode: str | None = "max-autotune"

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if (self.point_conditioning or self.stage_conditioning) and not self.pi05:
            raise ValueError(
                "point_conditioning/stage_conditioning require pi05=True (they extend the adaRMS pathway)."
            )
        if self.temporal_conditioning and not self.pi05:
            raise ValueError("temporal_conditioning requires pi05=True (adaRMS pathway).")
        if self.map_geo_conditioning and (not self.pi05 or self.map_tokens_k != 8):
            raise ValueError(
                "map_geo_conditioning requires pi05=True and map_tokens_k == 8 (patch_map_adaln.py: "
                "the geometry row/field selection is written against the frozen (8,72) token contract)."
            )
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                target_points=(
                    jax.ShapeDtypeStruct([batch_size, NUM_POINT_ARMS, 3], jnp.float32)
                    if self.point_conditioning
                    else None
                ),
                target_points_mask=(
                    jax.ShapeDtypeStruct([batch_size, NUM_POINT_ARMS], bool) if self.point_conditioning else None
                ),
                stage_tokens=(
                    jax.ShapeDtypeStruct([batch_size, NUM_POINT_ARMS], jnp.int32) if self.stage_conditioning else None
                ),
                history_gists=(
                    jax.ShapeDtypeStruct([batch_size, self.temporal_k + 1, self.temporal_gist_dim], jnp.float32)
                    if self.temporal_conditioning else None
                ),
                history_mask=(
                    jax.ShapeDtypeStruct([batch_size, self.temporal_k + 1], bool) if self.temporal_conditioning else None
                ),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)

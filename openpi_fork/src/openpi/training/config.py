"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, List, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

from openpi.configs import ROBOT_REGISTRY
import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.b1k_policy as b1k_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.shared.nnx_utils as nnx_utils  # patch_run2_config: freeze_filter
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.lerobot_compat as _lerobot_compat
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


# ---------------------------------------------------------------------------
# G3 point-conditioning gate (BEHAVIOR-2026): descriptive-language prompts.
#
# The G3 "lang" arms replace the raw dataset task names (e.g. "turning_on_radio") with natural
# instructions via transforms.RemapPrompt. Keys must match the LeRobot dataset task strings
# exactly; unmapped prompts pass through unchanged.
# ---------------------------------------------------------------------------
G3_DESCRIPTIVE_PROMPTS: dict[str, str] = {
    "turning_on_radio": "Go to the radio sitting on the table and press its power button to switch it on.",
    "picking_up_trash": "Collect the empty soda cans and drop each one into the trash can.",
    "attach_a_camera_to_a_tripod": (
        "Pick up the digital camera and attach it to the mounting plate on top of the camera tripod."
    ),
    "thawing_frozen_food": (
        "Take the frozen chicken and bread out of the fridge and set them on a plate next to the microwave to thaw."
    ),
}

# PLACEHOLDER — set to the real gate-task dataset location at H100 time (the merged 4-task
# LeRobot dataset produced by scripts/b1k/add_target_points.py, which carries the
# target_points/target_points_mask columns required by pi05_g3_lang_point).
G3_GATE_DATASET_ROOT = "/data/b1k_g3_gate"


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created. Should be list of str if using MultiLeRobotDataset.
    repo_id: str | List[str] | None = None
    # TEMPORAL FORCING: per-key history frame offsets (ascending, e.g. [-256, ..., -32, 0]); the b1k dataset adds
    # them as delta_timestamps (offset / fps). None => no history keys requested.
    history_frame_offsets: dict[str, list[int]] | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset | _lerobot_compat.LeRobotDataset] = ()

    # ============== Behavior Dataset Params ==================
    # Dataset class to use for loading the behavior-style lerobot dataset
    data_cls: Any = _lerobot_compat.LeRobotDataset
    # Path to local copy of the dataset
    # Note that this includes repo_id if LeRobotDataset and not include repo_id if MultiLeRobotDataset
    dataset_root: str | None = None
    # Extra kwargs to pass into the dataset constructor, if using a custom dataset class that requires additional arguments.
    dataset_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str | List[str] = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(
        self, assets_dir: epath.Path, asset_id: str | List[str] | None
    ) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        if isinstance(asset_id, list):
            # Only load the first asset_id assuming that the datasets are similar
            asset_id = asset_id[0]
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotB1KDataConfig(DataConfigFactory):
    robot_config_name: str = tyro.MISSING
    extra_delta_transform: bool = True
    action_sequence_keys: Sequence[str] = ("action",)

    # Dataset keys for AdaLN point/stage conditioning (only read when the model config enables
    # point_conditioning/stage_conditioning). The label pipeline stores per-frame per-arm values under these keys.
    target_points_key: str = "target_points"
    target_points_mask_key: str = "target_points_mask"
    stage_tokens_key: str | None = "stage_tokens"   # None -> no repack; B1KInputs derives per-arm tokens from the stage column
    # 4D-attendable perception: dataset column with the per-frame camera poses (3 x 7, base frame; FK precompute).
    cam_pose_key: str | None = None
    # v2 labels (RELABEL_V2.md, 2026-09-16): the press/full-stack configs point these at the crisp columns
    # (stage_key="stage_v2", progress_key="progress", target_points_key="target_points_v2"). Defaults keep every
    # Run-2/Run-3 arm byte-identical. progress_key=None -> no repack (RepackTransform KeyErrors on absent columns)
    # and B1KInputs falls back to (stage + 0.5) / stage_classes.
    stage_key: str = "stage"
    progress_key: str | None = None

    # Optional prompt rewrite table (raw task name -> descriptive instruction). Applied via
    # transforms.RemapPrompt as the first data transform, i.e. both at training time (after
    # prompt_from_task injects the raw dataset task name) and at serve time (on the websocket
    # prompt), so train/eval prompts stay consistent by construction. Unmapped prompts pass through.
    prompt_remap: dict[str, str] | None = None

    def _build_delta_mappings(self, robot_config) -> list[tuple[list[int], list[int]]]:
        state_slices = []
        state_offset = 0
        for proprio_config in robot_config.proprio:
            dim = 1 if proprio_config.is_eef else len(proprio_config.indices)
            state_slices.append(list(range(state_offset, state_offset + dim)))
            state_offset += dim

        mappings = []
        state_slice_index = 0
        for action_config in robot_config.action:
            if action_config.is_eef or not action_config.needs_delta_comp:
                continue
            action_dim = len(action_config.indices)
            while state_slice_index < len(state_slices) and len(state_slices[state_slice_index]) != action_dim:
                state_slice_index += 1
            if state_slice_index >= len(state_slices):
                raise ValueError(
                    f"Could not find a state slice for delta action group {action_config.name!r} "
                    f"with dim {action_dim} in robot config {robot_config.robot_type!r}."
                )
            mappings.append((action_config.indices, state_slices[state_slice_index]))
            state_slice_index += 1
        return mappings

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        robot_config = ROBOT_REGISTRY[self.robot_config_name]

        # Build repack mapping dynamically based on available observations
        repack_mapping = {}
        for i in range(3):  # Support up to 3 cameras (image_0, image_1, image_2)
            image_key = f"image_{i}"
            if image_key in robot_config.observations:
                repack_mapping[f"observation/{image_key}"] = robot_config.observations[image_key].dataset_key

        # Add non-image observations
        repack_mapping.update(
            {
                "observation/state": "observation.state",
                "actions": robot_config.action_key,
                "prompt": "prompt",
            }
        )

        # AdaLN point/stage conditioning: repack the label-pipeline keys only when the model consumes them,
        # so the point_conditioning=False path is byte-identical to stock.
        point_conditioning = getattr(model_config, "point_conditioning", False)
        stage_conditioning = getattr(model_config, "stage_conditioning", False)
        if point_conditioning:
            repack_mapping["target_points"] = self.target_points_key
            repack_mapping["target_points_mask"] = self.target_points_mask_key
        if stage_conditioning and self.stage_tokens_key is not None:
            repack_mapping["stage_tokens"] = self.stage_tokens_key
        if stage_conditioning and self.stage_tokens_key is None:
            repack_mapping["stage"] = self.stage_key          # tokens derived from the stage column in B1KInputs
        map_tokens_k = getattr(model_config, "map_tokens_k", 0)
        if map_tokens_k > 0:
            repack_mapping["map_tokens_full"] = "map_tokens_full"
            repack_mapping["map_tokens_blind"] = "map_tokens_blind"
            repack_mapping["stage"] = self.stage_key
            repack_mapping["aux_pixels"] = "aux_pixels"
        # STAGE/PROGRESS head (patch_stage_head.py): repack the parquet 'stage' column
        # whenever the model's stage_head consumes it (idempotent with the map path
        # above). NO 'progress' repack: RepackTransform KeyErrors on absent columns and
        # the parquet has no 'progress' column; B1KInputs derives the coarse fallback.
        stage_head = getattr(model_config, "stage_head", False)
        if stage_head:
            repack_mapping["stage"] = self.stage_key
        if self.progress_key is not None:
            repack_mapping["progress"] = self.progress_key   # v2 label clock (explicit key wins in B1KInputs)

        # GT-depth aux (patch_depth_aux.py): repack the precomputed gt_depth_ds column
        # only when the model consumes it. !! RepackTransform KeyErrors on absent
        # columns — every dataset in a depth_aux=True training mix must have been run
        # through add_depth_aux_labels.py first.
        if getattr(model_config, "depth_aux", False):
            repack_mapping["gt_depth"] = "gt_depth_ds"
        _geo = bool(getattr(model_config, "pe3d", False) or getattr(model_config, "geo_attention", False))
        if _geo and self.cam_pose_key:
            repack_mapping["cam_pose"] = self.cam_pose_key      # FK precompute column (3 cams x 7, base frame)
            if "gt_depth" not in repack_mapping:
                repack_mapping["gt_depth"] = "gt_depth_ds"
        temporal_conditioning = getattr(model_config, "temporal_conditioning", False)
        history_offsets = None
        _hist_tokens = bool(getattr(model_config, "hist_tokens", False))
        if _hist_tokens:
            _K, _S = model_config.hist_k, model_config.hist_stride
            _offs = [-(_K - i) * _S for i in range(_K)] + [0]
            history_offsets = {k: _offs for k in ("hist_tok", "hist_cellxyz", "hist_cellvalid", "odom_xyyaw")}
            for _hk in ("hist_tok", "hist_tok_is_pad", "hist_cellxyz", "hist_cellvalid", "odom_xyyaw"):
                repack_mapping[_hk] = _hk
            if "target_points_v2" not in repack_mapping and self.target_points_key:
                repack_mapping["target_points_v2"] = self.target_points_key
        if temporal_conditioning:
            _K, _S = model_config.temporal_k, model_config.temporal_stride
            _offs = [-(_K - i) * _S for i in range(_K)] + [0]
            history_offsets = dict(history_offsets or {}); history_offsets.update({"gist_head": _offs, "hist_geo": _offs})
            for _hk in ("gist_head", "gist_head_is_pad", "hist_geo", "hist_geo_is_pad"):
                repack_mapping[_hk] = _hk
        repack_transform = _transforms.Group(inputs=[_transforms.RepackTransform(repack_mapping)])

        # Prepare data for policy training
        # Convert images to uint8 numpy arrays, add masks
        input_transforms: list[_transforms.DataTransformFn] = []
        if self.prompt_remap:
            input_transforms.append(_transforms.RemapPrompt(self.prompt_remap))
        input_transforms.append(
            b1k_policy.B1KInputs(
                robot_config=robot_config,
                model_type=model_config.model_type,
                point_conditioning=point_conditioning,
                stage_conditioning=stage_conditioning,
                map_tokens_k=map_tokens_k,
                depth_aux=getattr(model_config, "depth_aux", False),
                stage_head=stage_head,
                stage_classes=getattr(model_config, "stage_classes", 4),
                temporal_conditioning=temporal_conditioning,
                temporal_k=getattr(model_config, "temporal_k", 8),
                geo_inputs=_geo,
                hist_tokens=_hist_tokens,
                hist_k=getattr(model_config, "hist_k", 8),
                hist_cells=getattr(model_config, "hist_cells", 16),
                hist_dim=getattr(model_config, "hist_dim", 2048),
                hist_stride=getattr(model_config, "hist_stride", 32),
            )
        )
        if float(getattr(model_config, "proprio_noise_std", 0.0)) > 0:
            input_transforms.append(_transforms.ProprioNoise(
                std=float(model_config.proprio_noise_std), heavy_p=float(getattr(model_config, "proprio_heavy_p", 0.0)),
                heavy_std=float(getattr(model_config, "proprio_heavy_std", 0.3)), start=3))
        data_transforms = _transforms.Group(
            inputs=input_transforms,
            outputs=[b1k_policy.B1KOutputs(action_dim=robot_config.action_dim)],
        )

        # extra delta transform.
        if self.extra_delta_transform:
            delta_mappings = self._build_delta_mappings(robot_config)

            data_transforms = data_transforms.push(
                inputs=[_transforms.MappedDeltaActions(delta_mappings)],
                outputs=[_transforms.MappedAbsoluteActions(delta_mappings)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=(robot_config.action_key,),
            use_quantile_norm=False,
            history_frame_offsets=history_offsets,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.

    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "B1K"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 8
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    # ============== B1K Specific Params ==================
    # How often (in steps) to log validation metrics. If None, no validation will be performed.
    val_log_interval: int | None = None
    # Validation batch size (optional, defaults to batch_size if not set)
    val_batch_size: int | None = None
    # Number of validation batches to average for validation loss
    val_num_batches: int = 10
    # Optionally, repo_id for validation set (if different from train)
    val_repo_id: str | None = None
    val_episodes_index: List[int] | None = None

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.

# ---- RUN 3 (patch_run3_configs.py): pre-registered DATA arms -------------------------------------
# Identical model / optimizer / schedule / seed / batch / steps; all warm-started from the
# Run-2 final params (arif101/b26-run2-params @49999, every module present => missing_regex
# admits nothing but lora). Norm stats = Run-2's, copied per arm (NOT recomputed per mix, so
# input normalisation is identical across arms and matches the init checkpoint).
# Launch env: B1K_STAGE_OVERSAMPLE=8 for every arm (Run-2 setting);
#             B1K_SAMPLE_WEIGHT_COL=sample_weight for a1..a5, UNSET for a0.
_RUN3_STEPS = 15000
_RUN3_WARMUP = 500
_RUN3_SAVE = 2500
_RUN3_WARMSTART = "/root/warmstart_run3/params"
_RUN3_ARMS = {
    "a0": "/root/b1k_radio_mix_a1",  # map only, NO poison down-weight (control / floor)
    "a1": "/root/b1k_radio_mix_a1",  # map only + poison down-weight (sample_weight col)
    "a2": "/root/b1k_radio_mix_a2",  # a1 + b1k_radio_factory   (honest grasp+transport)
    "a3": "/root/b1k_radio_mix_a3",  # a1 + b1k_radio_episodes  (honest grasp + learned press)
    "a4": "/root/b1k_radio_mix_a4",  # a1 + factory + episodes  (full mix)
    "a5": "/root/b1k_radio_mix_a5",  # a4 + b1k_radio_approach  (data lands later)
}


def _run3_cfg(arm: str, dataset_root: str) -> "TrainConfig":
    return TrainConfig(
        name=f"pi05_radio_run3_{arm}",
        model=pi0_config.Pi0Config(
            action_horizon=32,
            pi05=True,
            point_conditioning=True,
            point_noise_std=0.02,
            map_tokens_k=8,
            anti_shortcut=False,
            modality_dropout_p=0.2,
            map_geo_conditioning=True,
            depth_aux=True,
            stage_head=True,
        ),
        data=LeRobotB1KDataConfig(
            repo_id="b1k_radio",
            base_config=DataConfig(
                data_cls=_lerobot_compat.LeRobotDataset,
                dataset_root=dataset_root,
                prompt_from_task=True,
                dataset_kwargs={"tolerance_s": 5e-4},
            ),
            robot_config_name="b1k/R1Pro",
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(_RUN3_WARMSTART, missing_regex=".*lora.*"),
        freeze_filter=nnx_utils.PathRegex(".*map_(proj|registers|alpha|recon).*"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=_RUN3_WARMUP, peak_lr=2.5e-5, decay_steps=_RUN3_STEPS, decay_lr=2.5e-6),
        batch_size=32,
        num_train_steps=_RUN3_STEPS,
        save_interval=_RUN3_SAVE,
        log_interval=100,
        num_workers=8,
        exp_name=f"radio_run3_{arm}",
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir="./outputs/checkpoints",
    )


_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #  ================= behavior configs =================
    TrainConfig(
        name="pi05_b1k",
        model=pi0_config.Pi0Config(action_horizon=32, pi05=True),
        data=LeRobotB1KDataConfig(
            repo_id="turning_on_radio",
            base_config=DataConfig(
                data_cls=_lerobot_compat.LeRobotDataset,
                dataset_root="/viscam/u/shiyuc/openpi/2026-challenge-demos/b1k/turning_on_radio",
                prompt_from_task=True,
                dataset_kwargs={"tolerance_s": 5e-4},
            ),
            robot_config_name="b1k/R1Pro",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        save_interval=10_000,
        num_train_steps=50_000,
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir="./outputs/checkpoints",
    ),
    # ----------------------------------------------------------------------
    # G3 point-conditioning gate (BEHAVIOR-2026). See DESIGN.md.
    #
    # G3 has FOUR experimental arms but only THREE training runs:
    #   arm 1 "taskid"              -> pi05_g3_taskid      (raw task names, no points)
    #   arm 2 "lang"                -> pi05_g3_lang        (descriptive prompts, no points)
    #   arm 3 "lang+point, ORACLE"  -> pi05_g3_lang_point
    #   arm 4 "lang+point, HONEST"  -> pi05_g3_lang_point  (SAME checkpoint as arm 3)
    # Arms 3 and 4 differ only at EVAL time: both use the pi05_g3_lang_point checkpoint, and
    # the eval client supplies target_points over the websocket serve contract either from
    # privileged sim state (oracle, arm 3) or from the honest grounding head (arm 4). Training
    # is identical for both because the dataset's target_points are oracle labels either way,
    # so a fourth training config would be a bit-identical duplicate of the third.
    #
    # Common choices: warm-start pi05_base; full fine-tune (LoRA OFF — G3 decision);
    # batch size / lr / optimizer copied from pi05_b1k (library defaults: batch 32, AdamW +
    # cosine decay); 30k steps, checkpoint every 5k. All three arms train on the SAME merged
    # gate-task dataset ("b1k_g3_gate": turning_on_radio, picking_up_trash,
    # attach_a_camera_to_a_tripod, thawing_frozen_food) — dataset_root is a placeholder to be
    # filled at H100 time. The dataset carries target_points/target_points_mask columns
    # (added by scripts/b1k/add_target_points.py); the control arms simply never read them
    # (their repack mapping omits the keys, keeping them byte-identical to stock pi05_b1k
    # training).
    # ----------------------------------------------------------------------
    TrainConfig(
        # G3 arm 1: raw task names as prompts ("turning_on_radio", ...), no point conditioning.
        name="pi05_g3_taskid",
        model=pi0_config.Pi0Config(action_horizon=32, pi05=True),
        data=LeRobotB1KDataConfig(
            repo_id="b1k_g3_gate",
            base_config=DataConfig(
                data_cls=_lerobot_compat.LeRobotDataset,
                dataset_root=G3_GATE_DATASET_ROOT,  # PLACEHOLDER — fill at H100 time.
                prompt_from_task=True,
                dataset_kwargs={"tolerance_s": 5e-4},
            ),
            robot_config_name="b1k/R1Pro",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        save_interval=5_000,
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir="./outputs/checkpoints",
    ),
    TrainConfig(
        # G3 arm 2: descriptive natural-language prompts (G3_DESCRIPTIVE_PROMPTS), no points.
        # Isolates "better language" from "3D points" so arm 3's lift is attributable.
        name="pi05_g3_lang",
        model=pi0_config.Pi0Config(action_horizon=32, pi05=True),
        data=LeRobotB1KDataConfig(
            repo_id="b1k_g3_gate",
            base_config=DataConfig(
                data_cls=_lerobot_compat.LeRobotDataset,
                dataset_root=G3_GATE_DATASET_ROOT,  # PLACEHOLDER — fill at H100 time.
                prompt_from_task=True,
                dataset_kwargs={"tolerance_s": 5e-4},
            ),
            robot_config_name="b1k/R1Pro",
            prompt_remap=G3_DESCRIPTIVE_PROMPTS,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        save_interval=5_000,
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir="./outputs/checkpoints",
    ),
    TrainConfig(
        # GATE RUN: can this architecture reach nonzero q on ONE task?
        # turning_on_radio, 200 episodes, ~100% point coverage, ABSOLUTE joint actions.
        # Comet measured plain pi0.5 at 0.30-0.60 on this exact task, so nonzero is reachable.
        name="pi05_radio_gate",
        model=pi0_config.Pi0Config(
            action_horizon=32,
            pi05=True,
            point_conditioning=True,
            point_noise_std=0.02,
        ),
        data=LeRobotB1KDataConfig(
            repo_id="b1k_radio",
            base_config=DataConfig(
                data_cls=_lerobot_compat.LeRobotDataset,
                dataset_root="/root/b1k_radio2",
                prompt_from_task=True,
                dataset_kwargs={"tolerance_s": 5e-4},
            ),
            robot_config_name="b1k/R1Pro",
            # ABSOLUTE joint targets. Delta measured 0.00 vs absolute 0.30 on this task, PI's own
            # pi05_libero uses absolute, and our traces show ~65% command delivery under delta.
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params",
            missing_regex=".*lora.*|.*point_.*",  # fresh zero-init AdaLN point params
        ),
        batch_size=32,
        num_train_steps=50_000,
        save_interval=3_360,   # ~1 epoch at 430k frames / (32 x 4 GPUs)
        log_interval=100,
        num_workers=8,
        exp_name="radio_gate",
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir="./outputs/checkpoints",
    ),
    TrainConfig(
        # GATE RUN: can this architecture reach nonzero q on ONE task?
        # turning_on_radio, 200 episodes, ~100% point coverage, ABSOLUTE joint actions.
        # Comet measured plain pi0.5 at 0.30-0.60 on this exact task, so nonzero is reachable.
        name="pi05_radio_map",
        model=pi0_config.Pi0Config(
            action_horizon=32,
            pi05=True,
            point_conditioning=True,
            point_noise_std=0.02,
            map_tokens_k=8,
            anti_shortcut=True,
        ),
        data=LeRobotB1KDataConfig(
            repo_id="b1k_radio",
            base_config=DataConfig(
                data_cls=_lerobot_compat.LeRobotDataset,
                dataset_root="/root/b1k_radio_map",
                prompt_from_task=True,
                dataset_kwargs={"tolerance_s": 5e-4},
            ),
            robot_config_name="b1k/R1Pro",
            # ABSOLUTE joint targets. Delta measured 0.00 vs absolute 0.30 on this task, PI's own
            # pi05_libero uses absolute, and our traces show ~65% command delivery under delta.
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/root/warmstart_run1b/params",
            missing_regex=".*lora.*|.*map_.*|.*aux_.*",  # fresh init: map slot + aux heads
        ),
        batch_size=32,
        num_train_steps=50_000,
        save_interval=3_360,   # ~1 epoch at 430k frames / (32 x 4 GPUs)
        log_interval=100,
        num_workers=8,
        exp_name="radio_map",
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir="./outputs/checkpoints",
    ),
    TrainConfig(
        # RUN 2 (patch_run2_config.py): the measured-recipe run. Mix = b1k_radio_map
        # + splice-corrective (initiation neighborhood, 0.03 m) + RaC-recovery bank
        # (approach-stall recovery on policy-visited states, all far-field >0.35 m —
        # measured 2026-08-07), merged schema-exact by assemble_run2_mix.py, all three
        # carrying gt_depth_ds. New heads/routes per RUN2_PATCHES_BUILD.md. Prefix
        # map-token route FROZEN (G1: consumed-but-unpaid at 90% visibility); the
        # map GEOMETRY->AdaLN route is the Run-2 map venue. Oversampling: set
        # B1K_STAGE_OVERSAMPLE=8 in the launch env (patch_stage_oversample.py) —
        # source-level corrective weighting is baked at assembly. PRE-REGISTER the
        # eval protocol BEFORE looking at results (G1 early-separation lesson).
        name="pi05_radio_run2",
        model=pi0_config.Pi0Config(
            action_horizon=32,
            pi05=True,
            point_conditioning=True,
            point_noise_std=0.02,
            map_tokens_k=8,
            anti_shortcut=False,        # superseded by modality_dropout_p; enable ONE
            modality_dropout_p=0.2,
            map_geo_conditioning=True,
            depth_aux=True,
            stage_head=True,
        ),
        data=LeRobotB1KDataConfig(
            repo_id="b1k_radio",
            base_config=DataConfig(
                data_cls=_lerobot_compat.LeRobotDataset,
                dataset_root="/root/b1k_radio_run2mix",
                prompt_from_task=True,
                dataset_kwargs={"tolerance_s": 5e-4},
            ),
            robot_config_name="b1k/R1Pro",
            # ABSOLUTE joint targets (run-1 measurement: delta 0.00 vs absolute 0.30).
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/root/warmstart_run2/params",   # = run1b final params, restored on the trainer
            missing_regex=".*lora.*|.*stage_head.*|.*map_geo.*|.*depth_aux.*",
        ),
        # Freeze ONLY the prefix map-token route (params restore from run1b, then
        # never update): input/output projections, registers, ReZero alpha, recon
        # anti-sink. Aux decode heads stay trainable exactly as run1b.
        freeze_filter=nnx_utils.PathRegex(".*map_(proj|registers|alpha|recon).*"),
        batch_size=32,
        num_train_steps=50_000,
        save_interval=3_650,   # ~1 epoch at ~467k mix frames / (32 x 4 GPUs)
        log_interval=100,
        num_workers=8,
        exp_name="radio_run2",
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir="./outputs/checkpoints",
    ),
    *[_run3_cfg(_a, _r) for _a, _r in _RUN3_ARMS.items()],  # patch_run3_configs
    TrainConfig(
        # PRESS FIX (2026-09-12): A4 recipe (map + factory + episodes + poison-downweight) with
        # stage_conditioning ENABLED — the one flag that was OFF for every Run-3 arm. Feeds the
        # action expert the already-labeled task stage (the 'stage' column, now packed to
        # stage_tokens via the b1k_policy derivation) so it can separate grasp- from press-behavior
        # and execute the toggle. Warm-start from A4 (keeps the grasp); the new zero-init
        # stage_embed/stage_proj graft on as a no-op and learn. Controlled test: only stage_conditioning
        # differs from A4. See behavior2026/PRESS_FIX_SPEC.md. NOTE for training box: confirm the A4
        # mix at dataset_root and the A4 params at the weight_loader path are present (reassemble/pull).
        name="pi05_radio_press",
        model=pi0_config.Pi0Config(
            action_horizon=32,
            pi05=True,
            point_conditioning=True,
            point_noise_std=0.02,
            stage_conditioning=True,   # <-- THE FIX (was False/default for all Run-3 arms)
            progress_conditioning=True,   # continuous [0,1] progress into adaRMS (code ready)
            map_tokens_k=8,
            anti_shortcut=False,
            modality_dropout_p=0.2,
            map_geo_conditioning=True,
            depth_aux=True,
            stage_head=True,
        ),
        data=LeRobotB1KDataConfig(
            repo_id="b1k_radio",
            base_config=DataConfig(
                data_cls=_lerobot_compat.LeRobotDataset,
                dataset_root="/root/b1k_radio_mix_a5_v2",   # v2 labels (relabel_v2.py) on the A4 recipe + approach_v2
                prompt_from_task=True,
                dataset_kwargs={"tolerance_s": 5e-4},
            ),
            robot_config_name="b1k/R1Pro",
            extra_delta_transform=False,
            # v2 labels (RELABEL_V2.md): crisp 4-class stage, progress clock, stage-indexed target (rail before lift,
            # button after). Every source of the mix must carry these columns (relabel_v2.py).
            stage_key="stage_v2",
            progress_key="progress",
            target_points_key="target_points_v2",
            stage_tokens_key=None,   # no such column: B1KInputs broadcasts stage_v2 to both arms
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/root/ckpt_a4/a4/params",   # warm-start from A4 (keeps the grasp); confirm path on training box
            missing_regex=".*lora.*|.*stage_head.*|.*map_geo.*|.*depth_aux.*|.*stage_embed.*|.*stage_proj.*|.*progress_.*",  # progress_mlp_* are fresh (zero-init smoke 2026-09-12)
        ),
        freeze_filter=nnx_utils.PathRegex(".*map_(proj|registers|alpha|recon).*"),
        batch_size=32,
        num_train_steps=10_000,   # warm-start fine-tune from A4; tune as needed
        save_interval=2_500,
        log_interval=100,
        num_workers=8,
        exp_name="radio_press",
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir="./outputs/checkpoints",
    ),
    TrainConfig(
        # FULL STACK = press fix + temporal forcing (2026-09-16). Base comment: A4 recipe (map + factory + episodes + poison-downweight) with
        # stage_conditioning ENABLED — the one flag that was OFF for every Run-3 arm. Feeds the
        # action expert the already-labeled task stage (the 'stage' column, now packed to
        # stage_tokens via the b1k_policy derivation) so it can separate grasp- from press-behavior
        # and execute the toggle. Warm-start from A4 (keeps the grasp); the new zero-init
        # stage_embed/stage_proj graft on as a no-op and learn. Controlled test: only stage_conditioning
        # differs from A4. See behavior2026/PRESS_FIX_SPEC.md. NOTE for training box: confirm the A4
        # mix at dataset_root and the A4 params at the weight_loader path are present (reassemble/pull).
        name="pi05_radio_full",
        model=pi0_config.Pi0Config(
            action_horizon=32,
            pi05=True,
            point_conditioning=True,
            point_noise_std=0.02,
            stage_conditioning=True,   # <-- THE FIX (was False/default for all Run-3 arms)
            progress_conditioning=True,   # continuous [0,1] progress into adaRMS (code ready)
            map_tokens_k=8,
            anti_shortcut=False,
            modality_dropout_p=0.2,
            map_geo_conditioning=True,
            depth_aux=True,
            stage_head=True,
            temporal_conditioning=True,   # FULL STACK (2026-09-16): + temporal forcing (K=8 gists @ stride 32, zero-init gate)
        ),
        data=LeRobotB1KDataConfig(
            repo_id="b1k_radio",
            base_config=DataConfig(
                data_cls=_lerobot_compat.LeRobotDataset,
                dataset_root="/root/b1k_radio_mix_full",   # v2 labels + gist_head/hist_geo (precompute_gists.py)
                prompt_from_task=True,
                dataset_kwargs={"tolerance_s": 5e-4},
            ),
            robot_config_name="b1k/R1Pro",
            extra_delta_transform=False,
            # v2 labels (RELABEL_V2.md): crisp 4-class stage, progress clock, stage-indexed target (rail before lift,
            # button after). Every source of the mix must carry these columns (relabel_v2.py).
            stage_key="stage_v2",
            progress_key="progress",
            target_points_key="target_points_v2",
            stage_tokens_key=None,   # no such column: B1KInputs broadcasts stage_v2 to both arms
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/root/ckpt_full_init/params",   # symlink -> A4 (default) or S1 params: chosen by the S1 eval (2026-09-16)
            missing_regex=".*lora.*|.*stage_head.*|.*map_geo.*|.*depth_aux.*|.*stage_embed.*|.*stage_proj.*|.*progress_.*|.*temp_.*",  # progress_mlp_* are fresh (zero-init smoke 2026-09-12)
        ),
        # the SigLIP tower is FROZEN so the serve-time gists (computed by the model's own tower) equal the precomputed
        # ones from the A4 tower (precompute_gists.py). Deviation from the A4 recipe, logged in RUN3_CONFIG.md.
        freeze_filter=nnx.Any(nnx_utils.PathRegex(".*map_(proj|registers|alpha|recon).*"), nnx_utils.PathRegex(".*PaliGemma.*img.*")),
        batch_size=32,
        num_train_steps=10_000,   # warm-start fine-tune from A4; tune as needed
        save_interval=2_500,
        log_interval=100,
        num_workers=8,
        exp_name="radio_full",
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir="./outputs/checkpoints",
    ),
    TrainConfig(
        # FULL STACK = press fix + temporal forcing (2026-09-16). Base comment: A4 recipe (map + factory + episodes + poison-downweight) with
        # stage_conditioning ENABLED — the one flag that was OFF for every Run-3 arm. Feeds the
        # action expert the already-labeled task stage (the 'stage' column, now packed to
        # stage_tokens via the b1k_policy derivation) so it can separate grasp- from press-behavior
        # and execute the toggle. Warm-start from A4 (keeps the grasp); the new zero-init
        # stage_embed/stage_proj graft on as a no-op and learn. Controlled test: only stage_conditioning
        # differs from A4. See behavior2026/PRESS_FIX_SPEC.md. NOTE for training box: confirm the A4
        # mix at dataset_root and the A4 params at the weight_loader path are present (reassemble/pull).
        name="pi05_radio_geo",
        model=pi0_config.Pi0Config(
            action_horizon=32,
            pi05=True,
            point_conditioning=True,
            point_noise_std=0.02,
            stage_conditioning=True,   # <-- THE FIX (was False/default for all Run-3 arms)
            progress_conditioning=True,   # continuous [0,1] progress into adaRMS (code ready)
            map_tokens_k=8,
            anti_shortcut=False,
            modality_dropout_p=0.2,
            map_geo_conditioning=True,
            depth_aux=True,
            stage_head=True,
            temporal_conditioning=True,   # FULL STACK (2026-09-16): + temporal forcing (K=8 gists @ stride 32, zero-init gate)
            # 4D-ATTENDABLE PERCEPTION (ARCH_4D_ATTENTION_SPEC, 2026-09-22): zero-init 3D PE on patch tokens (A1),
            # zero-init geometric attention gains between the expert's queries and 3D-positioned keys (A3),
            # proprio noise as the copycat remedy (B2). Bit-parity with pi05_radio_full at step 0.
            pe3d=True,
            geo_attention=True,
            geo_sigma=0.15,
            geo_anchors=2,
            proprio_noise_std=0.03,
            proprio_heavy_p=0.2,
        ),
        data=LeRobotB1KDataConfig(
            repo_id="b1k_radio",
            base_config=DataConfig(
                data_cls=_lerobot_compat.LeRobotDataset,
                dataset_root="/root/b1k_radio_mix_full",   # v2 labels + gist_head/hist_geo (precompute_gists.py)
                prompt_from_task=True,
                dataset_kwargs={"tolerance_s": 5e-4},
            ),
            robot_config_name="b1k/R1Pro",
            extra_delta_transform=False,
            # v2 labels (RELABEL_V2.md): crisp 4-class stage, progress clock, stage-indexed target (rail before lift,
            # button after). Every source of the mix must carry these columns (relabel_v2.py).
            stage_key="stage_v2",
            progress_key="progress",
            target_points_key="target_points_v2",
            stage_tokens_key=None,   # no such column: B1KInputs broadcasts stage_v2 to both arms
            cam_pose_key="cam_pose",   # FK precompute column (fk_cam_poses.py); absent -> 3D path inert
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/root/ckpt_full_init/params",   # symlink -> A4 (default) or S1 params: chosen by the S1 eval (2026-09-16)
            missing_regex=".*lora.*|.*stage_head.*|.*map_geo.*|.*depth_aux.*|.*stage_embed.*|.*stage_proj.*|.*progress_.*|.*temp_.*|.*pe3d_.*|.*geo_gain.*|.*geo3_.*",  # progress_mlp_* are fresh (zero-init smoke 2026-09-12)
        ),
        # the SigLIP tower is FROZEN so the serve-time gists (computed by the model's own tower) equal the precomputed
        # ones from the A4 tower (precompute_gists.py). Deviation from the A4 recipe, logged in RUN3_CONFIG.md.
        freeze_filter=nnx.Any(nnx_utils.PathRegex(".*map_(proj|registers|alpha|recon).*"), nnx_utils.PathRegex(".*PaliGemma.*img.*")),
        batch_size=32,
        num_train_steps=10_000,   # warm-start fine-tune from A4; tune as needed
        save_interval=2_500,
        log_interval=100,
        num_workers=8,
        exp_name="radio_geo",
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir="./outputs/checkpoints",
    ),
    TrainConfig(
        # FULL STACK = press fix + temporal forcing (2026-09-16). Base comment: A4 recipe (map + factory + episodes + poison-downweight) with
        # stage_conditioning ENABLED — the one flag that was OFF for every Run-3 arm. Feeds the
        # action expert the already-labeled task stage (the 'stage' column, now packed to
        # stage_tokens via the b1k_policy derivation) so it can separate grasp- from press-behavior
        # and execute the toggle. Warm-start from A4 (keeps the grasp); the new zero-init
        # stage_embed/stage_proj graft on as a no-op and learn. Controlled test: only stage_conditioning
        # differs from A4. See behavior2026/PRESS_FIX_SPEC.md. NOTE for training box: confirm the A4
        # mix at dataset_root and the A4 params at the weight_loader path are present (reassemble/pull).
        name="pi05_radio_4d",
        model=pi0_config.Pi0Config(
            action_horizon=32,
            pi05=True,
            point_conditioning=True,
            point_noise_std=0.02,
            stage_conditioning=True,   # <-- THE FIX (was False/default for all Run-3 arms)
            progress_conditioning=True,   # continuous [0,1] progress into adaRMS (code ready)
            map_tokens_k=8,
            anti_shortcut=False,
            modality_dropout_p=0.2,
            map_geo_conditioning=True,
            depth_aux=True,
            stage_head=True,
            temporal_conditioning=True,   # FULL STACK (2026-09-16): + temporal forcing (K=8 gists @ stride 32, zero-init gate)
            # 4D-ATTENDABLE PERCEPTION (ARCH_4D_ATTENTION_SPEC, 2026-09-22): zero-init 3D PE on patch tokens (A1),
            # zero-init geometric attention gains between the expert's queries and 3D-positioned keys (A3),
            # proprio noise as the copycat remedy (B2). Bit-parity with pi05_radio_full at step 0.
            pe3d=True,
            geo_attention=True,
            geo_sigma=0.15,
            geo_anchors=2,
            proprio_noise_std=0.03,
            proprio_heavy_p=0.2,
            # A2: 4D history tokens in the prefix (K=8 past head frames x 16 cells @ stride 32), near-invisible at init
            hist_tokens=True,
            hist_k=8,
            hist_cells=16,
            hist_dim=2048,
            hist_stride=32,
            hist_vis_bias=-10.0,
            hist_ground_weight=0.05,
        ),
        data=LeRobotB1KDataConfig(
            repo_id="b1k_radio",
            base_config=DataConfig(
                data_cls=_lerobot_compat.LeRobotDataset,
                dataset_root="/root/b1k_radio_mix_full",   # v2 labels + gist_head/hist_geo (precompute_gists.py)
                prompt_from_task=True,
                dataset_kwargs={"tolerance_s": 5e-4},
            ),
            robot_config_name="b1k/R1Pro",
            extra_delta_transform=False,
            # v2 labels (RELABEL_V2.md): crisp 4-class stage, progress clock, stage-indexed target (rail before lift,
            # button after). Every source of the mix must carry these columns (relabel_v2.py).
            stage_key="stage_v2",
            progress_key="progress",
            target_points_key="target_points_v2",
            stage_tokens_key=None,   # no such column: B1KInputs broadcasts stage_v2 to both arms
            cam_pose_key="cam_pose",   # FK precompute column (fk_cam_poses.py); absent -> 3D path inert
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/root/ckpt_full_init/params",   # symlink -> A4 (default) or S1 params: chosen by the S1 eval (2026-09-16)
            missing_regex=".*lora.*|.*stage_head.*|.*map_geo.*|.*depth_aux.*|.*stage_embed.*|.*stage_proj.*|.*progress_.*|.*temp_.*|.*pe3d_.*|.*geo_gain.*|.*geo3_.*|.*hist_.*|.*key_bias_gain.*",  # progress_mlp_* are fresh (zero-init smoke 2026-09-12)
        ),
        # the SigLIP tower is FROZEN so the serve-time gists (computed by the model's own tower) equal the precomputed
        # ones from the A4 tower (precompute_gists.py). Deviation from the A4 recipe, logged in RUN3_CONFIG.md.
        freeze_filter=nnx.Any(nnx_utils.PathRegex(".*map_(proj|registers|alpha|recon).*"), nnx_utils.PathRegex(".*PaliGemma.*img.*")),
        batch_size=32,
        num_train_steps=10_000,   # warm-start fine-tune from A4; tune as needed
        save_interval=2_500,
        log_interval=100,
        num_workers=8,
        exp_name="radio_4d",
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir="./outputs/checkpoints",
    ),
    TrainConfig(
        # G3 arms 3 AND 4: descriptive prompts + AdaLN 3D-point conditioning. This single
        # checkpoint serves BOTH point arms; the oracle-vs-honest split is purely an eval-time
        # difference in who supplies target_points over the serve contract (see comment above
        # and DESIGN.md "Serve-side contract"). Dataset must carry target_points [2,3] float32
        # + target_points_mask [2] bool per frame (scripts/b1k/add_target_points.py).
        name="pi05_g3_lang_point",
        model=pi0_config.Pi0Config(
            action_horizon=32,
            pi05=True,
            point_conditioning=True,
            point_noise_std=0.02,  # 2 cm train-time jitter on valid points (anti-brittleness).
        ),
        data=LeRobotB1KDataConfig(
            repo_id="b1k_g3_gate",
            base_config=DataConfig(
                data_cls=_lerobot_compat.LeRobotDataset,
                dataset_root=G3_GATE_DATASET_ROOT,  # PLACEHOLDER — fill at H100 time.
                prompt_from_task=True,
                dataset_kwargs={"tolerance_s": 5e-4},
            ),
            robot_config_name="b1k/R1Pro",
            prompt_remap=G3_DESCRIPTIVE_PROMPTS,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        save_interval=5_000,
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir="./outputs/checkpoints",
    ),
    TrainConfig(
        # CPU-runnable smoke test for the G3 point arm: dummy gemma variants + fake data that
        # includes target_points/target_points_mask (FakeDataset generates them from the model
        # inputs_spec when point_conditioning=True). Mirrors the "debug_pi05" recipe. Note the
        # SigLIP vision tower is fixed So400m (not dummy), so steps are slow-but-feasible on CPU.
        name="pi05_g3_smoke",
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="dummy",
            action_expert_variant="dummy",
            point_conditioning=True,
            point_noise_std=0.02,
        ),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=2,
        log_interval=1,
        save_interval=100,
        ema_decay=None,
        num_workers=0,
        overwrite=True,
        exp_name="pi05_g3_smoke",
        wandb_enabled=False,
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instructions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    # RoboArena & PolaRiS configs.
    *roboarena_config.get_roboarena_configs(),
    *polaris_config.get_polaris_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]

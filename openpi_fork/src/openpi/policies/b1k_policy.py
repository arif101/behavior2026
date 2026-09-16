import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.models import pi0_config as _pi0_config
from openpi.configs.robots.base_config import RobotConfig


def make_b1k_example() -> dict:
    """Creates a random input example for the Droid policy."""
    return {
        "observation/image_0": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/image_1": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/image_2": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.rand(23),
        "prompt": "do something",
    }


def extract_state_from_proprio(proprio_data, robot_config: RobotConfig) -> np.ndarray:
    """Extract state from proprioception data based on robot configuration.

    We assume perfect correlation for the two gripper fingers.

    Args:
        proprio_data: Raw proprioception data
        robot_config: RobotConfig instance containing robot configuration

    Returns:
        Extracted state array
    """
    state = []
    for proprio in robot_config.proprio:
        if proprio.is_eef:
            # Sum the gripper finger positions to get a single width value
            state.append(proprio_data[..., proprio.indices].sum(axis=-1, keepdims=True))
        else:
            state.append(proprio_data[..., proprio.indices])
    return np.concatenate(state, axis=-1)


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class B1KInputs(transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    # Robot configuration object
    robot_config: RobotConfig = dataclasses.field(default=None)

    # Pack per-arm 3D target points / stage tokens for AdaLN conditioning (see models/pi0.py). Must match the
    # model config's point_conditioning/stage_conditioning flags. When enabled but the source data has no
    # "target_points" key, a zeros + all-invalid-mask sentinel is emitted, which the model embeds as the
    # learned null token (identical to the unconditioned model at init).
    point_conditioning: bool = False
    stage_conditioning: bool = False

    # FOVEATED MEMORY: emit (K, D) map tokens. Train: draw full-vs-blind columns
    # (blind = anti-shortcut stream). Serve: pass through a supplied \"map_tokens\".
    map_tokens_k: int = 0
    map_blind_prob: float = 0.3

    # GT-depth aux label passthrough (patch_depth_aux.py; consumed by models/pi0.py
    # when the model config sets depth_aux=True). Must match the model config's flag.
    depth_aux: bool = False

    # STAGE/PROGRESS head labels (patch_stage_head.py; consumed by models/pi0.py when
    # the model config sets stage_head=True). Must match the model config's flags.
    stage_head: bool = False
    stage_classes: int = 4

    # TEMPORAL FORCING (models/pi0.py temporal_conditioning): pack the K+1 head-camera gists (train: the loader's
    # delta-timestamp stack "gist_head" [K+1, D] oldest..current + "gist_head_is_pad"; serve: a ring buffer [n, D]
    # supplied by the b1k eval wrapper, left-padded with zeros/invalid) and the change targets from "hist_geo".
    temporal_conditioning: bool = False
    temporal_k: int = 8

    def __call__(self, data: dict) -> dict:
        proprio_data = data["observation/state"]
        # extract joint position
        state = extract_state_from_proprio(proprio_data, self.robot_config)
        if "actions" in data:
            action = data["actions"]

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        images, image_masks = [], []
        for camera_id in self.robot_config.observations:
            images.append(_parse_image(data[f"observation/{camera_id}"]))
            image_masks.append(np.True_)
        while len(images) < 3:
            images.append(np.zeros_like(images[0]))
            image_masks.append(np.False_)
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = tuple(images)
                image_masks = tuple(image_masks)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = tuple(images)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = action

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        n_arms = _pi0_config.NUM_POINT_ARMS
        if self.point_conditioning:
            if data.get("target_points") is not None:
                points = np.asarray(data["target_points"], dtype=np.float32).reshape(n_arms, 3)
                if data.get("target_points_mask") is not None:
                    mask = np.asarray(data["target_points_mask"]).astype(bool).reshape(n_arms)
                else:
                    mask = np.ones(n_arms, dtype=bool)
            else:
                # Absent -> "no target" sentinel: zeros + invalid mask (model uses the learned null token).
                points = np.zeros((n_arms, 3), dtype=np.float32)
                mask = np.zeros(n_arms, dtype=bool)
            inputs["target_points"] = points
            inputs["target_points_mask"] = mask

        if self.map_tokens_k > 0:
            if data.get("map_tokens") is not None:          # serve path (online map)
                inputs["map_tokens"] = np.asarray(data["map_tokens"], np.float32).reshape(
                    self.map_tokens_k, -1)
            elif data.get("map_tokens_full") is not None:    # train path (dataset columns)
                _key = "map_tokens_blind" if np.random.random() < self.map_blind_prob else "map_tokens_full"
                inputs["map_tokens"] = np.asarray(data[_key], np.float32).reshape(
                    self.map_tokens_k, -1)
            if data.get("stage") is not None:
                inputs["stage"] = np.int32(np.asarray(data["stage"]).reshape(-1)[0])
            if data.get("aux_pixels") is not None:
                inputs["aux_pixels"] = np.asarray(data["aux_pixels"], np.float32).reshape(9)
        if self.stage_conditioning:
            if data.get("stage_tokens") is not None:
                stage_tokens = np.asarray(data["stage_tokens"], dtype=np.int32).reshape(n_arms)
            elif data.get("stage") is not None:
                # Derive per-arm stage tokens from the per-frame 'stage' column (the conversion emits
                # 'stage' but not 'stage_tokens'). Broadcast the scalar task stage to all arms so the
                # conditioner receives the ACTUAL stage instead of zeros (which would be a silent no-op).
                stage_tokens = np.full(n_arms, int(np.asarray(data["stage"]).reshape(-1)[0]), dtype=np.int32)
            else:
                stage_tokens = np.zeros(n_arms, dtype=np.int32)
            inputs["stage_tokens"] = stage_tokens

        # STAGE/PROGRESS head labels (patch_stage_head.py). 'stage' is the per-frame
        # parquet column in {0..stage_classes-1}. 'progress': episode length is NOT
        # available in the row (the repack transform keeps only mapped keys and the
        # dataset has no per-episode-length column), so frame_index/max_frame_index is
        # not computable here; we use the coarse stage-midpoint fallback
        # (stage + 0.5) / stage_classes. An explicit 'progress' key (future label
        # column + repack entry, or the serve path) takes precedence automatically.
        if self.stage_head and data.get("stage") is not None:
            _st = np.int32(np.asarray(data["stage"]).reshape(-1)[0])
            inputs["stage"] = _st
            if data.get("progress") is not None:
                inputs["progress"] = np.float32(np.asarray(data["progress"]).reshape(-1)[0])
            else:
                inputs["progress"] = np.float32((float(_st) + 0.5) / float(self.stage_classes))

        # GT-depth aux label (patch_depth_aux.py): the precomputed gt_depth_ds parquet
        # column (3 cams x 16x16 patch-mean meters, 0=invalid), added offline by
        # box_scripts/add_depth_aux_labels.py. Absent key (serve path) => no label =>
        # the aux loss is inert.
        if self.depth_aux and data.get("gt_depth") is not None:
            inputs["gt_depth"] = np.asarray(data["gt_depth"], np.float32).reshape(768)

        if self.temporal_conditioning:
            K = self.temporal_k
            g = data.get("gist_head")
            if g is not None:
                g = np.asarray(g, np.float32)
                if g.ndim == 1:
                    g = g[None]
                pad = data.get("gist_head_is_pad")
                m = ~np.asarray(pad, bool).reshape(-1) if pad is not None else np.ones(len(g), bool)
                if len(g) < K + 1:
                    g = np.concatenate([np.zeros((K + 1 - len(g), g.shape[-1]), np.float32), g], axis=0)
                    m = np.concatenate([np.zeros(K + 1 - len(m), bool), m], axis=0)
                inputs["history_gists"] = g[-(K + 1):]
                inputs["history_mask"] = m[-(K + 1):]
            geo = data.get("hist_geo")
            if geo is not None and np.asarray(geo).ndim == 2 and len(np.asarray(geo)) == K + 1:
                geo = np.asarray(geo, np.float32)
                gp = data.get("hist_geo_is_pad")
                gm = ~np.asarray(gp, bool).reshape(-1) if gp is not None else np.ones(len(geo), bool)
                inputs["hist_flow"] = (geo[-1][None] - geo[:-1]).astype(np.float32)   # current - past, [K, 9]
                inputs["hist_flow_mask"] = np.logical_and(gm[:-1], gm[-1])

        return inputs


@dataclasses.dataclass(frozen=True)
class B1KOutputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    def __call__(self, data: dict) -> dict:
        # Only return the first 23 dims.
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}

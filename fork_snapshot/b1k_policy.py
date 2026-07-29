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
        if self.stage_conditioning:
            if data.get("stage_tokens") is not None:
                stage_tokens = np.asarray(data["stage_tokens"], dtype=np.int32).reshape(n_arms)
            else:
                stage_tokens = np.zeros(n_arms, dtype=np.int32)
            inputs["stage_tokens"] = stage_tokens

        return inputs


@dataclasses.dataclass(frozen=True)
class B1KOutputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    def __call__(self, data: dict) -> dict:
        # Only return the first 23 dims.
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}

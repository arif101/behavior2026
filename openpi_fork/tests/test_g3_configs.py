"""CPU tests for the G3 gate-experiment train configs (BEHAVIOR-2026 point-conditioning gate).

Run (from repo root, no GPU needed):

    PYTHONPATH=src:packages/openpi-client/src JAX_PLATFORMS=cpu \
        .venv/bin/python -m pytest tests/test_g3_configs.py -v

Covers:
  * the three G3 training configs (+ smoke) instantiate and carry the right flags
    (3 configs for 4 arms: arms 3/4 share pi05_g3_lang_point and differ only at eval time)
  * data configs resolve: repack mapping / B1KInputs flags follow the model config
  * a fake dataset-shaped batch flows through repack + RemapPrompt + B1KInputs,
    including the new target_points columns
"""

import dataclasses
import os
import sys
import types

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

try:  # minimal CPU test envs may lack lerobot; config.py only needs the class objects as defaults.
    import lerobot.datasets  # noqa: F401
except ImportError:

    class _LeRobotUnavailable:
        def __init__(self, *args, **kwargs):
            raise ImportError("lerobot is not installed in this test environment")

    _fake_ds = types.ModuleType("lerobot.datasets")
    _fake_ds.LeRobotDataset = _LeRobotUnavailable
    _fake_ds.LeRobotDatasetMetadata = _LeRobotUnavailable
    _fake_ds.MultiLeRobotDataset = _LeRobotUnavailable
    _fake_root = types.ModuleType("lerobot")
    _fake_root.datasets = _fake_ds
    sys.modules.setdefault("lerobot", _fake_root)
    sys.modules.setdefault("lerobot.datasets", _fake_ds)

from openpi.policies.b1k_policy import B1KInputs
import openpi.training.config as _config
import openpi.transforms as _transforms

G3_ARM_CONFIGS = ("pi05_g3_taskid", "pi05_g3_lang", "pi05_g3_lang_point")
GATE_TASKS = ("turning_on_radio", "picking_up_trash", "attach_a_camera_to_a_tripod", "thawing_frozen_food")


def _resolve(name: str):
    cfg = _config.get_config(name)
    return cfg, cfg.data.create(cfg.assets_dirs, cfg.model)


def _find_transform(group_inputs, cls):
    found = [t for t in group_inputs if isinstance(t, cls)]
    return found[0] if found else None


# ------------------------------------------------------------- config contents


def test_g3_configs_instantiate_and_flags():
    for name in G3_ARM_CONFIGS:
        cfg = _config.get_config(name)
        assert cfg.model.pi05, name
        assert cfg.num_train_steps == 30_000, name
        assert cfg.save_interval == 5_000, name
        # Full fine-tune: LoRA off (G3 decision) -> nothing frozen.
        assert "lora" not in cfg.model.paligemma_variant, name
        assert "lora" not in cfg.model.action_expert_variant, name
        # Warm start from pi05_base.
        assert "pi05_base" in cfg.weight_loader.params_path, name
        # Same gate dataset for every arm.
        assert cfg.data.repo_id == "b1k_g3_gate", name
        # batch/lr copied from pi05_b1k (which uses the library defaults).
        b1k = _config.get_config("pi05_b1k")
        assert cfg.batch_size == b1k.batch_size, name
        assert cfg.lr_schedule == b1k.lr_schedule, name
        assert cfg.optimizer == b1k.optimizer, name

    assert _config.get_config("pi05_g3_taskid").model.point_conditioning is False
    assert _config.get_config("pi05_g3_lang").model.point_conditioning is False
    lang_point = _config.get_config("pi05_g3_lang_point")
    assert lang_point.model.point_conditioning is True
    assert lang_point.model.point_noise_std == 0.02
    assert lang_point.model.stage_conditioning is False


def test_three_configs_for_four_arms():
    """G3 arms 3 (oracle points at eval) and 4 (honest-head points at eval) intentionally share
    the pi05_g3_lang_point training config: training consumes the dataset's oracle point labels
    either way, so the arm 3/4 split exists only in the eval client's serve-time point source.
    Guard the intent: lang vs lang_point must differ ONLY in the model's point flags."""
    lang = _config.get_config("pi05_g3_lang")
    lang_point = _config.get_config("pi05_g3_lang_point")
    assert dataclasses.replace(
        lang_point.model, point_conditioning=False, point_noise_std=lang.model.point_noise_std
    ) == lang.model
    assert dataclasses.replace(lang_point.data, repo_id=lang.data.repo_id) == lang.data
    # And there is deliberately no fourth training config.
    with pytest.raises(ValueError):
        _config.get_config("pi05_g3_lang_point_honest")


def test_descriptive_prompts_cover_gate_tasks():
    assert set(_config.G3_DESCRIPTIVE_PROMPTS) == set(GATE_TASKS)
    for task, prompt in _config.G3_DESCRIPTIVE_PROMPTS.items():
        assert prompt and prompt != task
        assert len(prompt.split()) >= 6, f"prompt for {task} does not look like a natural instruction"


def test_smoke_config():
    cfg = _config.get_config("pi05_g3_smoke")
    assert cfg.model.pi05 and cfg.model.point_conditioning
    assert cfg.model.paligemma_variant == "dummy"
    assert cfg.model.action_expert_variant == "dummy"
    assert isinstance(cfg.data, _config.FakeDataConfig)
    assert not cfg.wandb_enabled
    # FakeDataset generates target_points/target_points_mask from the model inputs_spec.
    obs_spec, _ = cfg.model.inputs_spec(batch_size=2)
    assert obs_spec.target_points.shape == (2, 2, 3)
    assert obs_spec.target_points_mask.shape == (2, 2)


# --------------------------------------------------------- transform resolution


def test_transforms_resolve_repack_and_flags():
    for name in G3_ARM_CONFIGS:
        cfg, data_cfg = _resolve(name)
        repack = data_cfg.repack_transforms.inputs[0]
        mapping = repack.structure
        expects_points = cfg.model.point_conditioning
        assert ("target_points" in mapping) == expects_points, name
        assert ("target_points_mask" in mapping) == expects_points, name
        assert "stage_tokens" not in mapping, name

        b1k_inputs = _find_transform(data_cfg.data_transforms.inputs, B1KInputs)
        assert b1k_inputs is not None, name
        assert b1k_inputs.point_conditioning == expects_points, name
        assert not b1k_inputs.stage_conditioning, name

        remap = _find_transform(data_cfg.data_transforms.inputs, _transforms.RemapPrompt)
        if name == "pi05_g3_taskid":
            assert remap is None, "taskid arm must keep raw task-name prompts"
        else:
            assert remap is not None, name
            assert remap.mapping == _config.G3_DESCRIPTIVE_PROMPTS
            # RemapPrompt must run before B1KInputs (it rewrites the raw prompt).
            inputs = list(data_cfg.data_transforms.inputs)
            assert inputs.index(remap) < inputs.index(b1k_inputs), name

        assert data_cfg.prompt_from_task, name


# ------------------------------------------------------------- fake batch flow


def _fake_dataset_row(*, with_points: bool) -> dict:
    row = {
        "observation.rgb.zed_link_camera_0": np.random.randint(256, size=(240, 240, 3), dtype=np.uint8),
        "observation.rgb.left_realsense_link_camera_0": np.random.randint(256, size=(240, 240, 3), dtype=np.uint8),
        "observation.rgb.right_realsense_link_camera_0": np.random.randint(256, size=(240, 240, 3), dtype=np.uint8),
        "observation.state": np.random.rand(61).astype(np.float32),
        "action": np.random.rand(32, 23).astype(np.float32),
        "prompt": "picking_up_trash",  # what prompt_from_task injects for this gate task
    }
    if with_points:
        # Flat list[6] float32 + bool[2]: exactly what scripts/b1k/add_target_points.py writes.
        row["target_points"] = np.asarray([0.5, -0.7, -0.1, 0.3, 0.5, -0.1], dtype=np.float32)
        row["target_points_mask"] = np.asarray([True, False])
    return row


def _run_input_pipeline(data_cfg, row: dict) -> dict:
    out = dict(row)
    for tf in (*data_cfg.repack_transforms.inputs, *data_cfg.data_transforms.inputs):
        out = tf(out)
    return out


def test_fake_batch_through_pipeline_lang_point():
    _, data_cfg = _resolve("pi05_g3_lang_point")
    out = _run_input_pipeline(data_cfg, _fake_dataset_row(with_points=True))

    assert out["target_points"].shape == (2, 3)
    assert out["target_points"].dtype == np.float32
    np.testing.assert_allclose(out["target_points"][0], [0.5, -0.7, -0.1])
    np.testing.assert_allclose(out["target_points"][1], [0.3, 0.5, -0.1])
    assert out["target_points_mask"].tolist() == [True, False]
    assert out["prompt"] == _config.G3_DESCRIPTIVE_PROMPTS["picking_up_trash"]
    assert out["state"].shape == (23,)
    assert set(out["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}

    # Frames with no target: the converter writes zeros + all-False mask; must survive packing.
    row = _fake_dataset_row(with_points=True)
    row["target_points"] = np.zeros(6, dtype=np.float32)
    row["target_points_mask"] = np.asarray([False, False])
    out = _run_input_pipeline(data_cfg, row)
    assert not out["target_points_mask"].any()
    assert np.all(out["target_points"] == 0)


def test_fake_batch_through_pipeline_control_arms():
    # Control arms read a dataset that CARRIES the new columns but must never see them:
    # the repack mapping omits the keys, so training stays byte-identical to stock pi05_b1k.
    for name in ("pi05_g3_taskid", "pi05_g3_lang"):
        _, data_cfg = _resolve(name)
        out = _run_input_pipeline(data_cfg, _fake_dataset_row(with_points=True))
        assert "target_points" not in out, name
        assert "target_points_mask" not in out, name
        expected_prompt = (
            "picking_up_trash" if name == "pi05_g3_taskid" else _config.G3_DESCRIPTIVE_PROMPTS["picking_up_trash"]
        )
        assert out["prompt"] == expected_prompt, name


def test_remap_prompt_passthrough():
    remap = _transforms.RemapPrompt(_config.G3_DESCRIPTIVE_PROMPTS)
    # Unknown prompt (e.g. serve client already sends a full instruction) passes through.
    assert remap({"prompt": "put the mug in the sink"})["prompt"] == "put the mug in the sink"
    # Bytes prompt (websocket) is decoded before lookup.
    assert (
        remap({"prompt": b"turning_on_radio"})["prompt"] == _config.G3_DESCRIPTIVE_PROMPTS["turning_on_radio"]
    )
    # No prompt -> no-op.
    assert "prompt" not in remap({"observation/state": np.zeros(3)})


if __name__ == "__main__":
    test_g3_configs_instantiate_and_flags()
    test_three_configs_for_four_arms()
    test_descriptive_prompts_cover_gate_tasks()
    test_smoke_config()
    test_transforms_resolve_repack_and_flags()
    test_fake_batch_through_pipeline_lang_point()
    test_fake_batch_through_pipeline_control_arms()
    test_remap_prompt_passthrough()
    print("\nALL G3 CONFIG TESTS PASSED")

"""Capture RGB + DEPTH + camera/base/object poses from the eval env, for latent back-projection.

Purpose: turn the policy's per-patch latents into a 3D point cloud, to see whether its visual
representation localises the target in metric space. That needs depth, which our Phase-A dataset
does NOT have (the depth streams were declared in info.json but never fetched). Capturing from
the sim instead is strictly better here:
  * exact depth rather than compressed video
  * camera intrinsics/extrinsics straight from the sensor
  * GROUND-TRUTH object pose, so the rendered cloud can be checked against where the radio is
  * the eval distribution -- i.e. the exact states where the policy is failing

Subclasses DefaultWrapper so it reuses the entire eval harness (scene load, robot config,
instance selection); the only additions are enabling the depth modality and dumping frames.

Usage (small max-steps; this is a capture, not a rollout):
    --env-wrapper eval.capture_wrapper.CaptureWrapper --max-steps 40
Env:
    B1K_CAPTURE_DIR   (default /root/capture)
    B1K_CAPTURE_EVERY (default 8 steps)
    B1K_CAPTURE_N     (default 12 frames, then it stops saving)
"""

import json
import os

import numpy as np
from omnigibson.eval.utils.eval_utils import set_sensor_modalities
from omnigibson.eval.wrappers.default_wrapper import DefaultWrapper
from omnigibson.utils.ui_utils import create_module_logger

logger = create_module_logger(module_name=__name__)

OUT_DIR = os.environ.get("B1K_CAPTURE_DIR", "/root/capture")
EVERY = int(os.environ.get("B1K_CAPTURE_EVERY", "8"))
MAX_N = int(os.environ.get("B1K_CAPTURE_N", "12"))
TASK_TARGETS = os.environ.get("B1K_TASK_TARGETS", "/root/g3_pipeline/task_targets.json")


class CaptureWrapper(DefaultWrapper):
    def __init__(self, env):
        super().__init__(env=env)
        self._robot = env.robots[0]
        self._t = 0
        self._n = 0
        os.makedirs(OUT_DIR, exist_ok=True)

        # DefaultWrapper set modalities to {"rgb"} only; we need depth for back-projection.
        # CRITICAL: the new sensor space must be written back into env.observation_space, exactly
        # as DefaultWrapper does. Calling load_observation_space() and discarding the return value
        # leaves the env's declared space stale, and env_base then raises
        #   "Observation space does not match returned observations!"
        # with MISSING (rgb) / EXTRA (depth_linear) keys -- which killed the first capture run.
        for sensor_name, sensor in self._robot.sensors.items():
            if not hasattr(sensor, "image_height"):
                continue
            try:
                set_sensor_modalities(sensor, {"rgb", "depth_linear"})
                sensor_space = sensor.load_observation_space()
                if env.observation_space is not None:
                    env.observation_space.spaces[self._robot.name].spaces[sensor_name] = sensor_space
            except Exception as e:
                logger.warning(f"CaptureWrapper: depth enable failed on {sensor_name} ({e})")
        logger.info("CaptureWrapper: depth enabled and observation space updated")

        self._targets = []
        try:
            task = getattr(getattr(env, "task", None), "activity_name", None) or os.environ.get(
                "B1K_TASK_NAME", ""
            )
            for cat in json.load(open(TASK_TARGETS)).get(task, {}).get("targets", []):
                objs = env.scene.object_registry("category", cat)
                if objs:
                    self._targets.extend(list(objs))
        except Exception as e:
            logger.warning(f"CaptureWrapper: target lookup failed ({e})")

    def _cam_params(self):
        """Per-camera pose + intrinsics, needed to un-project pixels into 3D."""
        out = {}
        for name, sensor in self._robot.sensors.items():
            if not hasattr(sensor, "image_height"):
                continue
            try:
                pos, quat = sensor.get_position_orientation()
                p = sensor.camera_parameters or {}
                out[name] = {
                    "pos": np.asarray(pos, dtype=np.float32).reshape(3),
                    "quat": np.asarray(quat, dtype=np.float32).reshape(4),
                    "H": int(sensor.image_height),
                    "W": int(sensor.image_width),
                    # focal/aperture let us build K without guessing an FOV
                    "focal": float(p.get("cameraFocalLength", 0.0) or 0.0),
                    "h_aperture": float(p.get("cameraAperture", [0.0, 0.0])[0])
                    if isinstance(p.get("cameraAperture"), (list, tuple))
                    else 0.0,
                    "view": np.asarray(p.get("cameraViewTransform", np.zeros(16)), dtype=np.float32),
                }
            except Exception:
                continue
        return out

    def _save(self, obs):
        node = obs.get(self._robot.name)
        if not isinstance(node, dict):
            return
        rec = {}
        for sensor_name, sdata in node.items():
            if not isinstance(sdata, dict):
                continue
            for modality in ("rgb", "depth_linear"):
                if modality in sdata:
                    rec[f"{sensor_name}::{modality}"] = np.asarray(sdata[modality])
        if not rec:
            return
        if "proprio" in node:
            rec["proprio"] = np.asarray(node["proprio"], dtype=np.float32)
        bp, bq = self._robot.get_position_orientation()
        rec["base_pos"] = np.asarray(bp, dtype=np.float32).reshape(3)
        rec["base_quat"] = np.asarray(bq, dtype=np.float32).reshape(4)
        for i, o in enumerate(self._targets[:2]):
            rec[f"target{i}_world"] = np.asarray(
                o.get_position_orientation()[0], dtype=np.float32
            ).reshape(3)
            rec[f"target{i}_name"] = np.array(getattr(o, "name", "?"))
        cams = self._cam_params()
        np.savez_compressed(os.path.join(OUT_DIR, f"frame_{self._n:03d}.npz"), **rec)
        with open(os.path.join(OUT_DIR, f"cams_{self._n:03d}.json"), "w") as f:
            json.dump({k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                           for kk, vv in v.items()} for k, v in cams.items()}, f, indent=1)
        self._n += 1
        logger.info(f"CaptureWrapper: saved frame {self._n}/{MAX_N}")

    def _maybe(self, obs):
        if isinstance(obs, dict) and self._n < MAX_N and self._t % EVERY == 0:
            try:
                self._save(obs)
            except Exception as e:
                logger.warning(f"CaptureWrapper: save failed ({e})")
        self._t += 1
        return obs

    def reset(self):
        return self._maybe(self.env.reset())

    def step(self, action, n_render_iterations=1):
        out = self.env.step(action, n_render_iterations=n_render_iterations)
        if isinstance(out, tuple) and out:
            return (self._maybe(out[0]), *out[1:])
        return self._maybe(out)

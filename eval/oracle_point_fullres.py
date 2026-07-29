"""OraclePointWrapper + the full-resolution RGB-D sensor config, so only ONE variable changes.

WHY THIS EXISTS
---------------
Our q_score=0.0 baseline ran with `RGBDFullResWrapperFixed` (head 720 / wrist 480 + depth) and,
as we later measured post-transform, `target_points` all zeros with `target_points_mask
[False, False]` -- the point conditioning was OFF. The model trained at ~100% point coverage.

`OraclePointWrapper` supplies real points, but it subclasses `DefaultWrapper` (224, RGB-only).
Using it as-is would change TWO things at once (points ON *and* vision downgraded), so a
difference in q would be uninterpretable. This class keeps the full-res sensor setup and adds the
points, leaving exactly one variable versus the baseline.

Order matters: DefaultWrapper.__init__ (reached via OraclePointWrapper) sets every camera to 224
and RGB-only, so the full-res reconfiguration has to run AFTER super().__init__.

It also repeats the fix from rgbd_full_res_fixed.py -- rebuild the CAMERA observation spaces per
sensor and write them back, never `env.load_observation_space()`, which needs live physics and
segfaults at wrapper-construction time (broken upstream in v3.9.0 AND v3.9.1).

VERIFY, DO NOT ASSUME: OraclePointWrapper writes /root/oracle_wrapper_stats.json with
`n_injections`. Its own docstring records a version that loaded cleanly, found the radio, and
injected NOTHING on every step -- three "conditioned" runs were silently unconditioned. Check
n_injections > 0 before believing any number this produces.

    --env-wrapper behavior2026_eval.oracle_point_fullres.OraclePointFullRes
"""

from omnigibson.eval.utils.eval_utils import (
    HEAD_RESOLUTION,
    WRIST_RESOLUTION,
    get_robot_camera_names,
    set_sensor_modalities,
)
from omnigibson.utils.ui_utils import create_module_logger

from behavior2026_eval.oracle_point_wrapper import OraclePointWrapper

logger = create_module_logger(module_name=__name__)


class OraclePointFullRes(OraclePointWrapper):
    """Oracle 3D target points AND the challenge-correct full-resolution RGB-D cameras."""

    def __init__(self, env):
        # Sets up oracle target tracking; also runs DefaultWrapper.__init__, which forces every
        # camera to 224 RGB-only. We undo that immediately below.
        super().__init__(env=env)

        robot = env.robots[0]
        robot_eval_config = getattr(env, "_eval_robot_config", {})
        camera_roles_by_sensor_name = {
            camera_name.split("::")[1]: camera_id
            for camera_id, camera_name in get_robot_camera_names(robot.name, robot_eval_config).items()
        }
        for sensor_name, sensor in robot.sensors.items():
            if not hasattr(sensor, "image_height") or not hasattr(sensor, "image_width"):
                continue
            set_sensor_modalities(sensor, {"rgb", "depth_linear"})
            camera_id = camera_roles_by_sensor_name.get(sensor_name)
            if camera_id == "head":
                sensor.image_height = HEAD_RESOLUTION[0]
                sensor.image_width = HEAD_RESOLUTION[1]
            else:
                sensor.image_height = WRIST_RESOLUTION[0]
                sensor.image_width = WRIST_RESOLUTION[1]
            # per-sensor rebuild + write-back; NEVER env.load_observation_space()
            sensor_space = sensor.load_observation_space()
            if env.observation_space is not None:
                env.observation_space.spaces[robot.name].spaces[sensor_name] = sensor_space

        logger.info(
            "OraclePointFullRes: oracle points ON, cameras at head=%s wrist=%s with depth.",
            HEAD_RESOLUTION,
            WRIST_RESOLUTION,
        )

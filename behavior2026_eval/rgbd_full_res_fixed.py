"""RGBDFullResWrapper with the observation-space reload fixed.

THE BUG (present in BEHAVIOR-1K v3.9.0 AND v3.9.1, verified by diffing both tags)
--------------------------------------------------------------------------------
omnigibson/eval/wrappers/rgbd_full_res_wrapper.py line 41 calls the ENV-level

    env.load_observation_space()

which recurses env -> robot.load_observation_space() -> robot.proprioception_dim ->
get_proprioception() -> get_joint_positions() -> entity_prim.py:859

    joint_positions = self._articulation_view.get_joint_positions().view(self.n_dof)

At wrapper-construction time the simulator is not playing, so the articulation view returns None
and this dies with `AttributeError: 'NoneType' object has no attribute 'view'`, surfacing as a
hydra InstantiationException and then SIGSEGV during unwinding.

DefaultWrapper does NOT hit this, because it only ever calls the PER-SENSOR
sensor.load_observation_space() and writes the result back into env.observation_space. Only the
cameras changed, so only the camera spaces need rebuilding -- there is no reason to rebuild the
robot's proprioception space, and doing so is what requires live physics.

Consequence worth knowing: as shipped, the DEBUG wrapper (224, RGB-only) runs and the CHALLENGE
wrapper (head/wrist full-res + depth) does not. That is a trap -- it quietly pushes you into
benchmarking a configuration that is not the one being scored.

We hit the same bug in our own capture_wrapper.py and fixed it the same way (commit 99f6ad9,
"write the new sensor space back to env.observation_space").

Everything else is byte-for-byte the upstream wrapper: same HEAD_RESOLUTION / WRIST_RESOLUTION,
same modalities {"rgb", "depth_linear"}, same head-vs-wrist role resolution. The only change is
how the observation space is rebuilt.

Usage:
    --env-wrapper behavior2026_eval.rgbd_full_res_fixed.RGBDFullResWrapperFixed
"""

from omnigibson.envs import Environment, EnvironmentWrapper
from omnigibson.eval.utils.eval_utils import (
    HEAD_RESOLUTION,
    WRIST_RESOLUTION,
    get_robot_camera_names,
    set_sensor_modalities,
)
from omnigibson.utils.ui_utils import create_module_logger

logger = create_module_logger(module_name=__name__)


class RGBDFullResWrapperFixed(EnvironmentWrapper):
    """Full-resolution RGB-D wrapper that rebuilds only the camera observation spaces.

    Args:
        env (og.Environment): The environment to wrap.
    """

    def __init__(self, env: Environment):
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
            # THE FIX: per-sensor reload + write-back, exactly as DefaultWrapper does. Never call
            # env.load_observation_space() here -- it needs live physics that does not exist yet.
            sensor_space = sensor.load_observation_space()
            if env.observation_space is not None:
                env.observation_space.spaces[robot.name].spaces[sensor_name] = sensor_space
        logger.info(
            "Reloaded camera observation spaces at full res (head=%s, wrist=%s) with depth.",
            HEAD_RESOLUTION,
            WRIST_RESOLUTION,
        )

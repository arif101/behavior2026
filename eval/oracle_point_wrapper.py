"""Inject ORACLE 3D target points into the eval observation, matching the training convention.

Why this exists: the first Phase-A rollout ran with NO points. `eval_b1k_wrapper.process_input`
only forwards `target_points`/`target_points_mask` if they are present in the obs, and nothing in
the eval path supplies them, so the policy silently got the "no target" sentinel (zeros + invalid
mask). That tests pi0.5+language, not our architecture -- and it failed exactly the way
language-only VLAs do: it drove to a wall-mounted TV instead of the radio on the table.

CONVENTION (must match scripts/b1k/add_target_points.py exactly, or this is worse than nothing):

    obj_base = R_world_base^T @ (p_obj_world - t_world_base)   # object, base frame
    delta    = obj_base - ee_base[arm]                          # ee_base IS the proprio slice
    target_points[arm] = delta                                  # float32, [2, 3] = [left, right]
    target_points_mask = [True, True]                           # BOTH arms get a point, and each
                                                                # picks its NEAREST live target

ee_base comes from the proprio vector itself (state[17:20] left, state[42:45] right), which is
already expressed in the base frame -- so we must NOT recompute it from world coordinates, or the
two terms would live in different frames.

This is the ORACLE point source: it reads object poses straight from the simulator. It exists to
isolate targeting from motor competence. It is NOT a submission path -- the honest source is the
grounding head. If the policy succeeds with oracle points and fails with predicted ones, the gap
is grounding; if it fails with both, the gap is motor.

Usage:
    --env-wrapper eval.oracle_point_wrapper.OraclePointWrapper
"""

import json
import os

import numpy as np
from omnigibson.eval.wrappers.default_wrapper import DefaultWrapper
from omnigibson.utils.ui_utils import create_module_logger

logger = create_module_logger(module_name=__name__)

TASK_TARGETS = os.environ.get("B1K_TASK_TARGETS", "/root/g3_pipeline/task_targets.json")
STATS_PATH = os.environ.get("B1K_ORACLE_STATS", "/root/oracle_wrapper_stats.json")
EEF_LEFT = slice(17, 20)
EEF_RIGHT = slice(42, 45)


def _quat_to_rot(q):
    """xyzw -> 3x3. Matches add_target_points.quat_to_rot."""
    x, y, z, w = [float(v) for v in q]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class OraclePointWrapper(DefaultWrapper):
    def __init__(self, env):
        super().__init__(env=env)
        self._robot = env.robots[0]
        self._targets = self._resolve_targets(env)
        self._n_inject = 0
        self._last = None
        logger.info(f"OraclePointWrapper: tracking {len(self._targets)} target object(s)")
        # The eval kit logs nothing about wrapper instantiation and this module's logger does not
        # reach the run log, so "did conditioning actually happen?" is otherwise unanswerable
        # after the fact. Write it to a file instead of inferring it from behaviour.
        self._dump()

    def _dump(self):
        try:
            json.dump(
                {
                    "loaded": True,
                    "n_targets": len(self._targets),
                    "target_names": [getattr(o, "name", "?") for o in self._targets][:4],
                    "n_injections": self._n_inject,
                    "last_points": None if self._last is None else np.asarray(self._last).tolist(),
                },
                open(STATS_PATH, "w"),
                indent=1,
            )
        except Exception:
            pass

    def _resolve_targets(self, env):
        """Find the task's target objects in the scene, by category then by name substring."""
        task_name = getattr(getattr(env, "task", None), "activity_name", None) or os.environ.get(
            "B1K_TASK_NAME", ""
        )
        cats = []
        try:
            cats = json.load(open(TASK_TARGETS)).get(task_name, {}).get("targets", [])
        except Exception as e:
            logger.warning(f"OraclePointWrapper: could not read task targets ({e})")
        if not cats:
            logger.warning(f"OraclePointWrapper: no targets for task '{task_name}' -- points OFF")
            return []

        found = []
        for cat in cats:
            objs = None
            try:
                objs = env.scene.object_registry("category", cat)
            except Exception:
                objs = None
            if objs:
                found.extend(list(objs))
                continue
            # fall back to a name-substring scan
            for o in env.scene.objects:
                if cat in getattr(o, "name", "").lower() or cat in getattr(o, "category", "").lower():
                    found.append(o)
        if not found:
            logger.warning(f"OraclePointWrapper: categories {cats} not found in scene -- points OFF")
        return found

    def _inject(self, obs):
        if not self._targets or not isinstance(obs, dict):
            return obs
        try:
            key = f"{self._robot.name}::proprio"
            if key not in obs:
                return obs
            prop = np.asarray(obs[key], dtype=np.float64)
            if prop.ndim != 1 or prop.shape[0] < 45:
                return obs

            base_pos, base_quat = self._robot.get_position_orientation()
            base_pos = np.asarray(base_pos, dtype=np.float64).reshape(3)
            r_wb_t = _quat_to_rot(np.asarray(base_quat).reshape(4)).T

            ee = {"left": prop[EEF_LEFT], "right": prop[EEF_RIGHT]}
            pts = np.zeros((2, 3), dtype=np.float32)
            msk = np.zeros(2, dtype=bool)
            for a, arm in enumerate(("left", "right")):
                best, best_d = None, np.inf
                for o in self._targets:
                    p_obj = np.asarray(o.get_position_orientation()[0], dtype=np.float64).reshape(3)
                    if not np.isfinite(p_obj).all():
                        continue
                    obj_base = r_wb_t @ (p_obj - base_pos)
                    d = float(np.linalg.norm(obj_base - ee[arm]))
                    if d < best_d:
                        best, best_d = obj_base, d
                if best is None:
                    continue
                pts[a] = (best - ee[arm]).astype(np.float32)
                msk[a] = True
            obs["target_points"] = pts
            obs["target_points_mask"] = msk
            self._n_inject += 1
            self._last = pts
            if self._n_inject % 50 == 1:  # cheap: first, then every 50th
                self._dump()
        except Exception as e:  # never take down a rollout over conditioning
            logger.warning(f"OraclePointWrapper: injection failed ({e})")
        return obs

    def reset(self):
        return self._inject(self.env.reset())

    def step(self, action, n_render_iterations=1):
        out = self.env.step(action, n_render_iterations=n_render_iterations)
        if isinstance(out, tuple) and out:
            return (self._inject(out[0]), *out[1:])
        return self._inject(out)

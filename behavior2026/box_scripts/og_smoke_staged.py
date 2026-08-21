"""Staged OmniGibson smoke test — isolates WHERE a silent segfault happens.

The plain smoke test died with no Python traceback right after "Imported scene 0", which is a
segfault signature rather than an exception. Paths are not the cause: gm.DATA_PATH resolves to
/root/bw/BEHAVIOR-1K/datasets, and omnigibson.key, behavior-1k-assets, omnigibson-robot-assets
and the task instances are all present there.

Each stage prints a marker and flushes, so the last marker in the log names the exact operation
that killed the process. Every boot costs ~5 min of shader compilation, so this is deliberately
one run that answers all of it rather than a bisect across several.

Two hard-won details carried over:
  * og.shutdown() HANGS in these containers -> os._exit(0).
  * first boot compiles shaders for ~4.5 min with a FROZEN log and pinned CPU. Not a hang.

Robot config note: v3.9 warns that the "type" key is deprecated in favour of lowercase "model"
(e.g. "model": "r1pro"). We use the new form here to remove that variable from the experiment.
"""

import os
import sys


def mark(s):
    print(f"STAGE::{s}", flush=True)
    sys.stdout.flush()


mark("IMPORT_START")
import omnigibson as og  # noqa: E402
from omnigibson.macros import gm  # noqa: E402

mark("IMPORT_OK")

gm.HEADLESS = True
gm.ENABLE_TRANSITION_RULES = False  # asserted by the playback wrapper elsewhere; harmless here

mark(f"DATA_PATH={gm.DATA_PATH}")

# ONE Environment per process, always.
#
# An earlier version of this script created an empty-scene env, closed it, then built a second env
# with the robot. That produced "AssertionError: Simulator must be stopped before loading scene!"
# followed immediately by SIGSEGV (exit 139) -- i.e. OmniGibson does NOT support constructing a
# second Environment in the same process, and env.close() does not put the simulator back into a
# loadable state. The segfault was the cascade from the failed assertion, not a robot problem.
#
# That run DID establish, before it died, that the empty scene creates, steps and closes cleanly
# (ENV_EMPTY_* all OK), so Isaac, Vulkan and physics on this Blackwell box are working. What
# remains untested is the robot + renderer path, which is what this single env now exercises.
mark("ENV_RGB_CREATE")
env = og.Environment(configs={
    "scene": {"type": "Scene"},
    "robots": [{"model": "r1pro", "name": "robot_r1", "obs_modalities": ["rgb"]}],
})
mark("ENV_RGB_OK")

mark("ENV_RGB_STEP")
obs = None
for _ in range(3):
    obs, *_ = env.step(env.action_space.sample())
mark("ENV_RGB_STEP_OK")


def walk(d, p=""):
    if isinstance(d, dict):
        for k, v in d.items():
            walk(v, f"{p}/{k}")
    else:
        shape = getattr(d, "shape", None)
        if shape is not None:
            print(f"  {p}: {tuple(shape)}", flush=True)


mark("OBS_KEYS")
walk(obs)

mark("SMOKE_ALL_OK")
os._exit(0)

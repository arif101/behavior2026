"""Minimal OmniGibson GPU smoke: load a scene + robot, step physics, render one RGB frame.

Two hard-won details:
  * og.shutdown() HANGS in these containers -> os._exit(0) instead.
  * first boot compiles shaders for ~4.5 min with a FROZEN log and pinned CPU. That is not a
    hang; do not kill it.
"""
import os
import numpy as np

import omnigibson as og
from omnigibson.macros import gm

gm.HEADLESS = True

cfg = {
    "scene": {"type": "Scene"},
    "robots": [{
        "type": "R1Pro",
        "name": "robot_r1",          # eval kit requires exactly this name
        "obs_modalities": ["rgb"],
    }],
}
env = og.Environment(configs=cfg)
for _ in range(5):
    obs, *_ = env.step(env.action_space.sample())
print("SMOKE_STEPPED_OK")

def walk(d, depth=0):
    if isinstance(d, dict):
        for k, v in d.items():
            print("  " * depth + str(k))
            walk(v, depth + 1)
    elif isinstance(d, np.ndarray):
        print("  " * depth + f"-> array {d.shape} {d.dtype}")

walk(obs)
print("OG_SMOKE_OK")
os._exit(0)

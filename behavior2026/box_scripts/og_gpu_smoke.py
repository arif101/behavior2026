"""OmniGibson-on-GPU smoke: scene + R1Pro, step physics, render one RGB frame, hard-exit.
Env: OMNIGIBSON_HEADLESS=1 OMNI_KIT_ALLOW_ROOT=1 XDG_RUNTIME_DIR=/tmp/xdg"""
import os, sys, time
os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")
t0 = time.time()
import numpy as np
import omnigibson as og
from omnigibson.macros import gm
gm.HEADLESS = True
cfg = {
    "scene": {"type": "InteractiveTraversableScene", "scene_model": "Rs_int", "load_object_categories": ["floors", "walls"]},
    "robots": [{"type": "R1Pro", "obs_modalities": ["rgb"], "name": "robot_r1"}],
}
env = og.Environment(configs=cfg)
print(f"env ready {time.time() - t0:.0f}s", flush=True)
obs, _ = env.reset()
for _ in range(5):
    a = env.action_space.sample()
    a = {k: np.zeros_like(v) for k, v in a.items()} if isinstance(a, dict) else np.zeros_like(a)
    obs, r, term, trunc, info = env.step(a)
def find_rgb(d, path=""):
    if isinstance(d, dict):
        for k, v in d.items():
            r = find_rgb(v, path + "/" + str(k))
            if r is not None:
                return r
    elif "rgb" in path and hasattr(d, "shape") and len(getattr(d, "shape", ())) == 3:
        return path, np.asarray(d)
    return None
keys, img = find_rgb(obs)
print(f"OG_GPU_SMOKE_OK rgb {img.shape} mean {float(img.mean()):.1f} keys {keys[:3]} total {time.time() - t0:.0f}s", flush=True)
os._exit(0)

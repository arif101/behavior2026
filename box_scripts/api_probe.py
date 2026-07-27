import inspect
from omnigibson.envs import EnvironmentWrapper
from omnigibson.scenes import Scene
print("wrapper methods:", [m for m in dir(EnvironmentWrapper) if not m.startswith("_")][:16])
print("step sig:", inspect.signature(EnvironmentWrapper.step))
print("reset sig:", inspect.signature(EnvironmentWrapper.reset))
print("scene object attrs:", [m for m in dir(Scene) if "object" in m.lower()][:10])

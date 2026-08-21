import sys
import numpy as np
import openpi.training.config as _config
from openpi.policies import policy_config as _policy_config

ck, arm = sys.argv[1], sys.argv[2]
cfg = _config.get_config("pi05_phaseA_point")
policy = _policy_config.create_trained_policy(cfg, ck, default_prompt="smoke")
obs = {f"observation/image_{k}": np.zeros((224, 224, 3), np.uint8) for k in range(3)}
obs.update({"observation/state": np.zeros(61, np.float32), "prompt": "smoke",
            "target_points": np.zeros((2, 3), np.float32),
            "target_points_mask": np.zeros(2, bool)})
a = np.asarray(policy.infer(obs)["actions"])
print("STAGE_SERVE_SMOKE_OK", arm, a.shape, "finite:", bool(np.all(np.isfinite(a))))

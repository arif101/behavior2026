"""Run-2 serving smoke: pi05_radio_run2 config + radio_run2@49999 params -> one infer.
Expected: RUN2_SERVE_SMOKE_OK (32, 23) finite: True
Run: cd /root/openpi_fork && env python (openpi env) this_file
"""
import numpy as np

import openpi.training.config as _config
from openpi.policies import policy_config as _policy_config

cfg = _config.get_config("pi05_radio_run2")
policy = _policy_config.create_trained_policy(cfg, "/root/ckpt",
                                              default_prompt="turning_on_radio")
obs = {f"observation/image_{k}": np.zeros((224, 224, 3), np.uint8) for k in range(3)}
obs.update({
    "observation/state": np.zeros(61, np.float32),
    "prompt": "turning_on_radio",
    "target_points": np.zeros((2, 3), np.float32),
    "target_points_mask": np.zeros(2, bool),
    "map_tokens_full": np.zeros(576, np.float32),
    "map_tokens_blind": np.zeros(576, np.float32),
})
a = np.asarray(policy.infer(obs)["actions"])
print("RUN2_SERVE_SMOKE_OK", a.shape, "finite:", bool(np.all(np.isfinite(a))))

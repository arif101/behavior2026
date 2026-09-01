"""Policy relay worker: load the radio_run2 policy once, then serve infers over a
file RPC in /dev/shm (the sim lives in the `behavior` conda env, the policy in
`openpi` — two processes, one GPU).

Protocol: requester atomically renames relay_req.npz into place (keys: im0, im1,
im2 uint8 224x224x3 [head, left wrist, right wrist], state float32 (61,)).
Worker infers, atomically writes relay_act.npy ((32, 23) float32 chunk), removes
the request. Requester deletes relay_act.npy before posting the next request.

Run:  XLA_PYTHON_CLIENT_PREALLOCATE=false \
      /root/miniconda3/envs/openpi/bin/python -u /root/relay_worker.py
"""
import os
import time

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import openpi.training.config as _config  # noqa: E402
from openpi.policies import policy_config as _policy_config  # noqa: E402

REQ = "/dev/shm/relay_req.npz"
ACT = "/dev/shm/relay_act.npy"
TMP = "/dev/shm/relay_act.tmp.npy"

cfg = _config.get_config("pi05_radio_run2")
policy = _policy_config.create_trained_policy(cfg, "/root/ckpt",
                                              default_prompt="turning_on_radio")
print("RELAY_WORKER_READY", flush=True)

while True:
    if not os.path.exists(REQ):
        time.sleep(0.01)
        continue
    try:
        z = np.load(REQ)
        obs = {
            "observation/image_0": z["im0"],
            "observation/image_1": z["im1"],
            "observation/image_2": z["im2"],
            "observation/state": z["state"].astype(np.float32),
            "prompt": "turning_on_radio",
            "target_points": np.zeros((2, 3), np.float32),
            "target_points_mask": np.zeros(2, bool),
            "map_tokens_full": np.zeros(576, np.float32),
            "map_tokens_blind": np.zeros(576, np.float32),
        }
        t0 = time.time()
        a = np.asarray(policy.infer(obs)["actions"], np.float32)
        np.save(TMP, a)
        os.replace(TMP, ACT)
        os.remove(REQ)
        print(f"INFER ok {a.shape} {time.time() - t0:.2f}s", flush=True)
    except Exception as e:  # noqa: BLE001
        print("RELAY_WORKER_ERR", repr(e), flush=True)
        try:
            os.remove(REQ)
        except OSError:
            pass
        time.sleep(0.1)

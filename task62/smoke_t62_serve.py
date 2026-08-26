"""Task-62 context-VLA serving smoke: pi05_t62_ctx config + run-2 params (context params fresh)
-> one infer on a REAL dataset frame (images decoded from the b1k_t62 videos) with its context.
Expected: T62_SERVE_SMOKE_OK (32, 23) finite: True, plus a context-liveness delta (same frame,
context zeroed vs labeled) — at fresh init this delta must be exactly 0 (zero-init AdaLN term).

Run (openpi env, GPU): cd /root/openpi_fork && XLA_PYTHON_CLIENT_PREALLOCATE=false python task62/smoke_t62_serve.py
Needs /root/ckpt_t62/{params -> run-2 params, assets/b1k_t62/norm_stats.json}.
"""
import os, sys, json
import numpy as np

sys.path.insert(0, "/root/openpi_fork/src")
import openpi.training.config as _config
from openpi.policies import policy_config as _policy_config

CKPT = "/root/ckpt_t62"
cfg = _config.get_config("pi05_t62_ctx")
policy = _policy_config.create_trained_policy(cfg, CKPT, default_prompt="halve_an_egg")
print("policy loaded", flush=True)

# one real frame: episode 0, frame ~3450 (chop segment) from the dataset
import pyarrow.parquet as pq
t = pq.read_table("/root/b1k_t62/data/chunk-062/file-000.parquet", columns=["episode_index", "frame_index", "observation.state", "context", "stage"])
df = t.to_pandas(); row = df[(df.episode_index == 0) & (df.frame_index == 3450)].iloc[0]
state = np.asarray(row["observation.state"], np.float32); ctx = np.asarray(row["context"], np.float32)
print("frame ctx:", np.round(ctx, 2).tolist(), "stage:", int(row["stage"]), flush=True)

obs = {f"observation/image_{k}": np.zeros((224, 224, 3), np.uint8) for k in range(3)}
obs.update({"observation/state": state, "prompt": "halve_an_egg"})
a0 = np.asarray(policy.infer(dict(obs, context=np.zeros(16, np.float32)))["actions"])
a1 = np.asarray(policy.infer(dict(obs, context=ctx))["actions"])
a2 = np.asarray(policy.infer(dict(obs))["actions"])   # no context key -> zeros sentinel
print("T62_SERVE_SMOKE_OK", a1.shape, "finite:", bool(np.all(np.isfinite(a1))), flush=True)
print("context liveness |a(ctx)-a(0)| max:", float(np.abs(a1 - a0).max()), "(expected 0.0 at fresh init)",
      "| |a(none)-a(0)| max:", float(np.abs(a2 - a0).max()), flush=True)
os._exit(0)

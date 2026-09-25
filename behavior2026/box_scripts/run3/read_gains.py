"""4D arm liveness readout (read-only): norms of the attention-side gains and zero-init outs in a checkpoint's params,
against their init values. Flat = the expert never used the geometry / history; moving = the pathway is live.
  JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false python read_gains.py [--ckpt-root /root/openpi_fork/outputs/checkpoints/pi05_radio_4d/radio_4d] [--step N]
"""
import argparse, glob, os, pathlib, re; os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np
from openpi.models import model as _m

def flat(d, p=""):
    for k, v in d.items():
        if isinstance(v, dict): yield from flat(v, p + "/" + k)
        else: yield p + "/" + k, v

ap = argparse.ArgumentParser(); ap.add_argument("--ckpt-root", default="/root/openpi_fork/outputs/checkpoints/pi05_radio_4d/radio_4d"); ap.add_argument("--step", type=int)
a = ap.parse_args(); root = pathlib.Path(a.ckpt_root)
steps = sorted(int(p.name) for p in root.iterdir() if p.name.isdigit() and (p / "params").exists()) if root.exists() else []
if not steps: print("GAINS no committed checkpoint with params under", root); raise SystemExit(0)
step = a.step if a.step is not None else steps[-1]
P = dict(flat(_m.restore_params(str(root / str(step) / "params"), restore_type=np.ndarray)))
def n(k): return float(np.linalg.norm(np.asarray(P[k], np.float32))) if k in P else float("nan")
llm = "/PaliGemma/llm/layers/attn"
g3 = np.asarray(P.get(f"{llm}/geo3_gain", np.zeros(1)), np.float32); kb = np.asarray(P.get(f"{llm}/key_bias_gain", np.ones(1)), np.float32)
print(f"GAINS step={step} (available: {steps})")
print(f"  geo3_gain      norm={n(f'{llm}/geo3_gain'):.4g} (init 0)  per-layer max|g|: {np.abs(g3).reshape(g3.shape[0], -1).max(-1).round(4).tolist() if g3.ndim >= 2 else 'n/a'}")
print(f"  geo3_anchor    kernel norm={n(f'{llm}/geo3_anchor/kernel'):.4g} bias norm={n(f'{llm}/geo3_anchor/bias'):.4g} (init 0)")
print(f"  key_bias_gain  mean={float(kb.mean()):.4f} min={float(kb.min()):.4f} max={float(kb.max()):.4f} (init 1.0; <1 = history tokens made MORE visible)")
print(f"  pe3d_out       kernel norm={n('/pe3d_out/kernel'):.4g} (init 0)   hist_pe_out kernel norm={n('/hist_pe_out/kernel'):.4g} (init 0)")
hi = P.get("/hist_in/kernel"); print(f"  hist_in        |K - I| = {float(np.linalg.norm(np.asarray(hi, np.float32) - np.eye(hi.shape[0], hi.shape[1], dtype=np.float32))):.4g} (init 0)" if hi is not None else "  hist_in absent")
print(f"  temp_out       kernel norm={n('/temp_out/kernel'):.4g} (temporal-forcing gate; FULL ckpt value is the baseline)")
print("GAINS_DONE")

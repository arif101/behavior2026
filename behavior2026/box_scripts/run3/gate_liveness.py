"""Temporal-forcing liveness readout (TEMPORAL_4D_SWEEP §4.5): the zero-init gate is temp_out.kernel; its Frobenius norm
over checkpoints shows flat (inert) / open-hold (adopted) / open-then-collapse (abandoned). Also prints the pre-gate
flow-head weight norm as a proxy for the supervision landing on the pathway.
  python gate_liveness.py /root/openpi_fork/outputs/checkpoints/pi05_radio_full/radio_full [more ckpt dirs...]
"""
import glob, pathlib, sys, numpy as np
from openpi.models import model as _model
for d in sys.argv[1:]:
    steps = sorted(int(p.name) for p in pathlib.Path(d).iterdir() if p.name.isdigit() and (p / "params").exists()) if pathlib.Path(d).is_dir() else []
    cands = [pathlib.Path(d) / str(s) / "params" for s in steps] or [pathlib.Path(d)]
    for c in cands:
        try:
            p = _model.restore_params(c, restore_type=np.ndarray)["params"] if "params" in _model.restore_params(c, restore_type=np.ndarray) else _model.restore_params(c, restore_type=np.ndarray)
        except Exception as e:
            print(c, "restore failed:", e); continue
        gate = p.get("temp_out", {}).get("kernel"); flow = p.get("temp_flow_in", {}).get("kernel"); tin = p.get("temp_in", {}).get("kernel")
        if gate is None: print(c, ": no temporal params"); continue
        print(f"{c}: ||temp_out.kernel||_F = {np.linalg.norm(gate):.4e} (mean|w| {np.abs(gate).mean():.2e}) | ||temp_flow_in|| = {np.linalg.norm(flow):.3f} | ||temp_in|| = {np.linalg.norm(tin):.3f}")

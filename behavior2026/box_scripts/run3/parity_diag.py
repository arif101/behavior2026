"""Parity bisection (2026-09-25): the first 4D parity readout gave geo_rel_diff=0.14 (expected ~0). This lists every
parameter each config has that the A4 checkpoint lacks (with its init norm: non-zero = suspect) and evaluates the flow loss
on ONE batch for the geo/4d models with individual 4D inputs removed (patch_xyz -> pe3d + geo3 off; anchors -> geo3 off;
history_tokens -> hist off), so the deviation is attributed to a code path. Guarded main (spawn loader workers)."""
import os, sys, dataclasses; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, "/root/run3")
import jax, jax.numpy as jnp, numpy as np
from flax import nnx
from openpi.training import config as _c, data_loader as _dl
from openpi.models import model as _m
from parity_smoke_4d import loss_for

def flat(d, p=""):
    for k, v in d.items():
        if isinstance(v, dict): yield from flat(v, p + "/" + k)
        else: yield p + "/" + k, v

def main():
    a4 = dict(flat(_m.restore_params("/root/run3_dl/a4/params", restore_type=np.ndarray)))
    for name in ("pi05_radio_full", "pi05_radio_geo", "pi05_radio_4d"):
        cfg = _c.get_config(name); model = cfg.model.create(jax.random.key(0)); mf = dict(flat(nnx.split(model)[1].to_pure_dict()))
        miss = [k for k in mf if k not in a4]; print(f"FRESH {name}: model leaves {len(mf)} fresh {len(miss)} a4-not-in-model {len([k for k in a4 if k not in mf])}", flush=True)
        for k in miss:
            v = np.asarray(mf[k]); print(f"   {k} {tuple(v.shape)} {v.dtype} norm={float(np.linalg.norm(v.astype(np.float32))):.4g}", flush=True)
        del model, mf
    cfg4 = _c.get_config("pi05_radio_4d"); loader = _dl.create_b1k_data_loader(cfg4, shuffle=False, num_batches=1)
    obs, act = next(iter(loader)); rng = jax.random.key(1)
    def run(tag, name, **drop):
        o = dataclasses.replace(obs, **{k: None for k in drop}) if drop else obs
        l, missing, _ = loss_for(name, o, act, rng); print(f"LOSS {tag}: {l:.6f}", flush=True); return l
    lf = run("full", "pi05_radio_full")
    lg = run("geo", "pi05_radio_geo")
    lg_nopatch = run("geo -patch_xyz(-pe3d,-geo3)", "pi05_radio_geo", patch_xyz=True, patch_valid=True)
    lg_noanch = run("geo -anchors(-geo3)", "pi05_radio_geo", anchors=True)
    l4 = run("4d", "pi05_radio_4d")
    l4_nohist = run("4d -history_tokens", "pi05_radio_4d", history_tokens=True, history_xyz=True, history_valid=True, history_dt=True)
    l4_nopatch = run("4d -patch_xyz -anchors", "pi05_radio_4d", patch_xyz=True, patch_valid=True, anchors=True)
    lf2 = run("full again", "pi05_radio_full")
    print(f"DIAG_RESULT full={lf:.6f} full_again={lf2:.6f} geo={lg:.6f} geo_nopatch={lg_nopatch:.6f} geo_noanchors={lg_noanch:.6f} 4d={l4:.6f} 4d_nohist={l4_nohist:.6f} 4d_nopatch_noanch={l4_nopatch:.6f}", flush=True)
    print("DIAG_DONE", flush=True)

if __name__ == "__main__":
    main()

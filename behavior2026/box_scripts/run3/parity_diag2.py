"""Parity bisection, round 2 (2026-09-25): geo deviates from full even with every 4D input removed, so the cause is in the
MODEL CONSTRUCTION, not the 4D code paths. Flip the geo flags one at a time on top of pi05_radio_full's model config and
evaluate the same batch; also compare the merged A4 weights leaf-by-leaf between the full model and the flipped model."""
import os, sys, dataclasses; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, "/root/run3")
import jax, jax.numpy as jnp, numpy as np
from flax import nnx
from openpi.training import config as _c, data_loader as _dl
from openpi.models import model as _m

def flat(d, p=""):
    for k, v in d.items():
        if isinstance(v, dict): yield from flat(v, p + "/" + k)
        else: yield p + "/" + k, v

A4 = None
def build(model_cfg):
    global A4
    if A4 is None: A4 = _m.restore_params("/root/run3_dl/a4/params", restore_type=np.ndarray)
    model = model_cfg.create(jax.random.key(0)); graphdef, state = nnx.split(model); pure = state.to_pure_dict(); missing = []
    def merge(dst, src, path=""):
        for k, v in dst.items():
            if k in src and isinstance(v, dict) and isinstance(src[k], dict): merge(v, src[k], path + "/" + k)
            elif k in src and not isinstance(v, dict):
                if tuple(np.shape(src[k])) != tuple(np.shape(v)): print(f"   SHAPE MISMATCH {path}/{k}: model {np.shape(v)} a4 {np.shape(src[k])}", flush=True)
                dst[k] = jnp.asarray(src[k], dtype=jnp.asarray(v).dtype)
            else: missing.append(path + "/" + k)
    merge(pure, A4); state.replace_by_pure_dict(pure); return nnx.merge(graphdef, state), pure, missing

def main():
    cfg4 = _c.get_config("pi05_radio_4d"); loader = _dl.create_b1k_data_loader(cfg4, shuffle=False, num_batches=1)
    obs, act = next(iter(loader)); rng = jax.random.key(1)
    base = _c.get_config("pi05_radio_full").model
    variants = {"full": base,
                "full+pe3d": dataclasses.replace(base, pe3d=True),
                "full+geo_attention": dataclasses.replace(base, geo_attention=True, geo_anchors=3, geo_sigma=0.15),
                "full+proprio_noise": dataclasses.replace(base, proprio_noise_std=0.03, proprio_heavy_p=0.2),
                "geo(cfg)": _c.get_config("pi05_radio_geo").model}
    ref_pure = None; res = {}
    for tag, mc in variants.items():
        model, pure, missing = build(mc)
        l = float(jnp.mean(model.compute_loss(rng, obs, act, train=False))); res[tag] = l
        print(f"LOSS2 {tag}: {l:.6f} fresh={len(missing)}", flush=True)
        f = dict(flat(pure))
        if ref_pure is None: ref_pure = f
        else:
            diffs = [(k, float(jnp.max(jnp.abs(jnp.asarray(f[k], jnp.float32) - jnp.asarray(ref_pure[k], jnp.float32))))) for k in f if k in ref_pure and np.shape(f[k]) == np.shape(ref_pure[k])]
            nz = [(k, d) for k, d in diffs if d > 0]; shp = [k for k in f if k in ref_pure and np.shape(f[k]) != np.shape(ref_pure[k])]
            print(f"   weights vs full: {len(diffs)} common leaves, {len(nz)} differ, {len(shp)} shape-differ {shp[:4]}; top: {sorted(nz, key=lambda x: -x[1])[:4]}", flush=True)
        del model
    print("DIAG2_RESULT " + " ".join(f"{k.replace(' ', '_')}={v:.6f}" for k, v in res.items()), flush=True); print("DIAG2_DONE", flush=True)

if __name__ == "__main__":
    main()

"""Parity smoke (ARCH_4D_ATTENTION_SPEC): load the FULL checkpoint (the 4D arm's warm start) into pi05_radio_full,
pi05_radio_geo and pi05_radio_4d, compute
the flow loss on ONE fixed batch (same rng), and report: geo must equal full to float precision (zero-init PE, zero-init
gains, kernel * 0); 4d must be within ~1e-3 relative (history tokens near-invisible: bias -10 -> e^-10 mass) and the
liveness parameters must exist (geo_gain, key_bias_gain, hist_*). Runs on the GPU with on-demand allocation. Guarded main(): the b1k loader uses spawn workers, which re-import this
file (an unguarded module body re-ran the whole smoke inside the worker and died with the multiprocessing bootstrap error, 2026-09-25)."""
import os; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax, jax.numpy as jnp, numpy as np
from flax import nnx
from openpi.training import config as _c, data_loader as _dl
from openpi.models import model as _m
import openpi.shared.nnx_utils as nnx_utils

PARAMS = os.environ.get("PARITY_PARAMS", "/root/run3_dl/full/params")   # MUST be the warm start: with A4 params the full
# config's own trained heads (stage_embed, progress_mlp, temp_*) are random-init and their values depend on how many rng
# draws precede them, so adding pe3d_in shifted them and faked a 14% "deviation" (bisected 2026-09-25, parity_diag2.py).


def loss_for(name, batch_obs, batch_act, rng, model_cfg=None):
    cfg = _c.get_config(name)
    model = (model_cfg or cfg.model).create(jax.random.key(0))
    params = _m.restore_params(PARAMS, restore_type=np.ndarray)
    graphdef, state = nnx.split(model)
    pure = state.to_pure_dict()
    missing = []
    def merge(dst, src, path=""):
        for k, v in dst.items():
            if k in src and isinstance(v, dict) and isinstance(src[k], dict): merge(v, src[k], path + "/" + k)
            elif k in src and not isinstance(v, dict): dst[k] = jnp.asarray(src[k], dtype=jnp.asarray(v).dtype)
            else: missing.append(path + "/" + k)
    merge(pure, params); state.replace_by_pure_dict(pure); model = nnx.merge(graphdef, state)
    loss = model.compute_loss(rng, batch_obs, batch_act, train=False)
    return float(jnp.mean(loss)), missing, cfg

def main():
    names = ["pi05_radio_full", "pi05_radio_geo", "pi05_radio_4d"] + [n for n in os.environ.get("PARITY_EXTRA", "").split(",") if n]
    cfg4 = _c.get_config("pi05_radio_4d")
    loader = _dl.create_b1k_data_loader(cfg4, shuffle=False, num_batches=1)
    batch = next(iter(loader)); obs, act = batch
    rng = jax.random.key(1)
    res = {}
    import dataclasses
    print(f"PARITY params: {PARAMS}", flush=True)
    for n in names:
        l, missing, cfg = loss_for(n, obs, act, rng)
        res[n] = l; print(f"PARITY {n}: loss={l:.6f} fresh_params={len(missing)} ({', '.join(sorted(set(m.split('/')[1] for m in missing))[:8])})", flush=True)
    # 4d flow-only: the history aux heads (hist_ground_*/hist_stage_*) are random-init and add 0.05 x (rail MSE + stage CE)
    # to the loss; switch the aux off to compare the FLOW loss the way the geo arm is compared
    l4f, _, _ = loss_for("pi05_radio_4d", obs, act, rng, model_cfg=dataclasses.replace(cfg4.model, hist_ground_weight=0.0))
    print(f"PARITY pi05_radio_4d(flow only, aux off): loss={l4f:.6f}", flush=True)
    d_geo = abs(res["pi05_radio_geo"] - res["pi05_radio_full"]) / max(res["pi05_radio_full"], 1e-9)
    d_4d = abs(l4f - res["pi05_radio_full"]) / max(res["pi05_radio_full"], 1e-9)
    d_4d_aux = abs(res["pi05_radio_4d"] - res["pi05_radio_full"]) / max(res["pi05_radio_full"], 1e-9)
    for n in names[3:]:
        print(f"PARITY_EXTRA {n} rel_diff={abs(res[n] - res['pi05_radio_full']) / max(res['pi05_radio_full'], 1e-9):.2e} (vs full; expect <=1e-5 for a train-only change)", flush=True)
    print(f"PARITY_RESULT geo_rel_diff={d_geo:.2e} (expect ~0) 4d_rel_diff={d_4d:.2e} (flow only, expect <1e-2) 4d_with_aux_rel_diff={d_4d_aux:.2e} inputs: patch_xyz={obs.patch_xyz is not None} hist={obs.history_tokens is not None}", flush=True)


if __name__ == "__main__":   # the loader spawns workers that re-import this file: module-level work must not run in them
    main()

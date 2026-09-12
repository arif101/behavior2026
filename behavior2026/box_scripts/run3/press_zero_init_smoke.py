"""Zero-init no-op smoke for the press-fix stack (stage_conditioning + progress_conditioning).
Loads A4's params into BOTH pi05_radio_run2 (Run-3 arch) and pi05_radio_press (new modules),
draws ONE real batch, and compares compute_loss under the same rng. PASS = the press loss equals
the run2 loss (new modules are zero-init no-ops) and every fresh leaf is admitted by missing_regex.
Run on a box with A4 params at /root/ckpt_a4/params and a LeRobot root at $SMOKE_ROOT.
  cd /root/openpi_fork && XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python press_zero_init_smoke.py
"""
import dataclasses, gc, os, re, sys, time
os.environ["B1K_STAGE_OVERSAMPLE"] = "0"
import jax, jax.numpy as jnp, numpy as np
import flax.nnx as nnx
import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.weight_loaders as _wl
from openpi.training import data_loader as _dl

ROOT = os.environ.get("SMOKE_ROOT", "/root/b1k_radio_map_smoke")
PARAMS = os.environ.get("SMOKE_PARAMS", "/root/ckpt_a4/params")


def flat_keys(tree, prefix=""):
    out = set()
    if isinstance(tree, dict):
        for k, v in tree.items():
            out |= flat_keys(v, f"{prefix}/{k}" if prefix else str(k))
    else:
        out.add(prefix)
    return out


def build(cfg_name):
    cfg = _config.get_config(cfg_name)
    cfg = dataclasses.replace(cfg, batch_size=2, num_workers=0,
                              data=dataclasses.replace(cfg.data, base_config=dataclasses.replace(cfg.data.base_config, dataset_root=ROOT)),
                              weight_loader=_wl.CheckpointWeightLoader(PARAMS, missing_regex=cfg.weight_loader.missing_regex))
    return cfg


loaded = _model.restore_params(PARAMS, restore_type=np.ndarray)
ck = flat_keys(loaded)
print(f"A4 params: {len(ck)} leaves", flush=True)
results = {}
batch = None
for name in ("pi05_radio_run2", "pi05_radio_press"):
    cfg = build(name)
    t0 = time.time()
    model = cfg.model.create(jax.random.key(0))
    pure = nnx.state(model, nnx.Param).to_pure_dict()
    km = flat_keys(jax.tree.map(lambda x: 0, pure))
    fresh, orphans = sorted(km - ck), sorted(ck - km)
    rx = re.compile(cfg.weight_loader.missing_regex)
    uncovered = [k for k in fresh if not rx.search(k)]
    print(f"[{name}] tree {len(km)} leaves; fresh {len(fresh)} {fresh[:12]}; orphans {orphans[:5]}; uncovered-fresh {uncovered[:8]}", flush=True)
    assert not orphans, f"{name}: checkpoint keys missing from model tree"
    assert not uncovered, f"{name}: fresh params not admitted by missing_regex"
    params = cfg.weight_loader.load(jax.tree.map(lambda x: np.asarray(x), pure))
    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(params)
    model = nnx.merge(graphdef, state)
    if batch is None:
        dl = _dl.create_b1k_data_loader(cfg, shuffle=True, num_batches=1)
        batch = next(iter(dl))
        obs, act = batch
        print(f"batch: state {obs.state.shape} actions {act.shape} stage {None if obs.stage is None else np.asarray(obs.stage)[:2]}", flush=True)
    obs, act = batch
    loss = model.compute_loss(jax.random.key(1), obs, act, train=False)
    loss = float(jnp.mean(loss))
    results[name] = loss
    print(f"[{name}] loss {loss:.6f}  ({time.time() - t0:.0f}s)", flush=True)
    del model, params, state, graphdef, pure
    import gc; gc.collect(); jax.clear_caches()
d = abs(results["pi05_radio_press"] - results["pi05_radio_run2"])
print(f"PRESS_ZERO_INIT_SMOKE {'PASS' if d < 1e-4 else 'FAIL'}: run2 {results['pi05_radio_run2']:.6f} press {results['pi05_radio_press']:.6f} |delta| {d:.2e}", flush=True)

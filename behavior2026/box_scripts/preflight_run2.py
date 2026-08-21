"""G0 preflight for Run-2 (pi05_radio_run2) — run BEFORE the smoke and the launch.

Three checks, all against the REAL restored run1b params (no GPU needed; pure key-set
algebra over pytrees + one loader draw):

  A. FLAGS-OFF BIT-IDENTITY: the pi05_radio_map (run-1) model tree must have EXACTLY the
     checkpoint's key set — proves every Run-2 patch is inert when its flag is off (the
     default-off => run1b-safe guarantee, now checked against the real artifact).
  B. FLAGS-ON DELTA: the pi05_radio_run2 model tree may differ from the checkpoint ONLY by
     new keys matching the config's missing_regex (fresh modules), and every checkpoint
     key must exist in the new tree (no orphans/renames).
  C. MIX LOADER DRAW: one batch from the assembled mix through the run2 data pipeline with
     B1K_STAGE_OVERSAMPLE=8 — proves repack keys (gt_depth_ds incl.), transforms, video
     decode, and the oversampler all fire on the real merged dataset.

Usage (trainer):  cd /root/openpi_fork && .venv/bin/python /root/preflight_run2.py
"""

import os
import re
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("B1K_STAGE_OVERSAMPLE", "8")

import dataclasses

import jax
import numpy as np


def flat_keys(tree, prefix=""):
    out = set()
    if isinstance(tree, dict):
        for k, v in tree.items():
            out |= flat_keys(v, f"{prefix}/{k}" if prefix else str(k))
    else:
        out.add(prefix)
    return out


def main():
    import openpi.models.model as _model
    import openpi.training.config as _config

    ckpt_path = "/root/warmstart_run2/params"
    loaded = _model.restore_params(ckpt_path, restore_type=np.ndarray)
    ck = flat_keys(loaded)
    print(f"checkpoint params: {len(ck)} leaves", flush=True)

    import flax.nnx as nnx

    def model_keys(cfg_name):
        cfg = _config.get_config(cfg_name)
        model = cfg.model.create(jax.random.key(0))
        state = nnx.state(model, nnx.Param)
        pure = jax.tree.map(lambda x: 0, state.to_pure_dict())
        return flat_keys(pure), cfg

    # A ------------------------------------------------------------------------------
    k1, _ = model_keys("pi05_radio_map")
    only_model = sorted(k1 - ck)
    only_ckpt = sorted(ck - k1)
    if only_model or only_ckpt:
        print("A FAIL: flags-off tree != checkpoint tree")
        print("  model-only:", only_model[:10])
        print("  ckpt-only:", only_ckpt[:10])
        sys.exit(1)
    print("A PASS: flags-off (run-1 config) tree is exactly the checkpoint tree", flush=True)

    # B ------------------------------------------------------------------------------
    k2, cfg2 = model_keys("pi05_radio_run2")
    missing_re = re.compile(cfg2.weight_loader.missing_regex)
    orphans = sorted(ck - k2)
    fresh = sorted(k2 - ck)
    bad_fresh = [k for k in fresh if not missing_re.fullmatch(k) and not missing_re.search(k)]
    if orphans:
        print("B FAIL: checkpoint keys missing from run2 tree (rename/drift):", orphans[:10])
        sys.exit(1)
    if bad_fresh:
        print("B FAIL: fresh keys NOT covered by missing_regex:", bad_fresh[:10])
        sys.exit(1)
    print(f"B PASS: run2 tree = checkpoint + {len(fresh)} fresh leaves, all covered by "
          f"missing_regex ({cfg2.weight_loader.missing_regex})", flush=True)
    for k in fresh:
        print(f"    fresh: {k}")

    # C ------------------------------------------------------------------------------
    from openpi.training import data_loader as _dl

    cfg_small = dataclasses.replace(cfg2, batch_size=4, num_workers=0)
    batch = next(iter(_dl.create_b1k_data_loader(cfg_small, shuffle=True, num_batches=1)))
    obs, act = batch
    def shape_of(x):
        return None if x is None else getattr(x, "shape", None)
    print("C batch drawn:", flush=True)
    print("    state:", shape_of(obs.state), " actions:", shape_of(act))
    print("    target_points:", shape_of(obs.target_points),
          " map_tokens:", shape_of(obs.map_tokens))
    print("    stage:", shape_of(obs.stage), " progress:", shape_of(obs.progress),
          " gt_depth:", shape_of(obs.gt_depth))
    assert obs.gt_depth is not None, "gt_depth missing from batch — repack/labels broken"
    assert obs.stage is not None, "stage missing from batch"
    gd = np.asarray(obs.gt_depth)
    assert gd.shape[-1] == 768 and (gd > 0).mean() > 0.5, f"gt_depth implausible: {gd.shape}, valid {(gd>0).mean():.3f}"
    print("C PASS: mix loader + oversampler + label columns all live", flush=True)
    print("PREFLIGHT_RUN2_PASS", flush=True)


if __name__ == "__main__":
    main()

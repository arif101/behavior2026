"""G0 preflight for the Run-3 data arms — run BEFORE the smoke and the launch (CPU only).

  A. TREE IDENTITY: the run3 model tree must have EXACTLY the Run-2 checkpoint's key set
     (no fresh leaves, no orphans) — every module restores; missing_regex admits nothing.
  B. PER ARM (with that arm's launch env): build the REAL b1k loader; assert the sampler is
     the WeightedRandomSampler the loader patch installs; read its weights back and report
     the effective sampling mass per source (joined on meta/run3_sources.json) + the
     down-weighted fraction; draw one batch (repack keys incl. gt_depth_ds, stage, points,
     map tokens, video decode all fire). a0 must show NO down-weighting; a1..a5 must.
Writes /root/run3_logs/preflight_<arm>.json.

Usage:  cd /root/openpi_fork && .venv/bin/python /root/run3/preflight_run3.py a0 a1 a2 a3 a4
"""
import dataclasses
import json
import os
import pathlib
import re
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["B1K_STAGE_OVERSAMPLE"] = "8"

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


def main(arms):
    import flax.nnx as nnx
    import openpi.models.model as _model
    import openpi.training.config as _config
    from openpi.training import data_loader as _dl

    out_dir = pathlib.Path("/root/run3_logs"); out_dir.mkdir(exist_ok=True)
    cfg0 = _config.get_config(f"pi05_radio_run3_{arms[0]}")
    loaded = _model.restore_params(cfg0.weight_loader.params_path, restore_type=np.ndarray)
    ck = flat_keys(loaded)
    print(f"checkpoint params: {len(ck)} leaves from {cfg0.weight_loader.params_path}", flush=True)
    model = cfg0.model.create(jax.random.key(0))
    pure = jax.tree.map(lambda x: 0, nnx.state(model, nnx.Param).to_pure_dict())
    km = flat_keys(pure)
    fresh, orphans = sorted(km - ck), sorted(ck - km)
    if fresh or orphans:
        print("A FAIL: run3 tree != checkpoint tree; fresh:", fresh[:10], "orphans:", orphans[:10])
        sys.exit(1)
    print(f"A PASS: run3 model tree == Run-2 checkpoint tree ({len(km)} leaves, nothing fresh)", flush=True)
    del loaded, model, pure

    for arm in arms:
        if arm == "a0":
            os.environ.pop("B1K_SAMPLE_WEIGHT_COL", None)
        else:
            os.environ["B1K_SAMPLE_WEIGHT_COL"] = "sample_weight"
        cfg = _config.get_config(f"pi05_radio_run3_{arm}")
        root = pathlib.Path(cfg.data.base_config.dataset_root)
        info = json.loads((root / "meta" / "info.json").read_text())
        feats = info["features"]
        assert "sample_weight" in feats and "gt_depth_ds" in feats, f"{arm}: columns not registered"
        assert not any(k.startswith("observation.depth_linear") for k in feats), f"{arm}: depth streams still registered"
        srcs = json.loads((root / "meta" / "run3_sources.json").read_text())["sources"]
        cfg_small = dataclasses.replace(cfg, batch_size=4, num_workers=0)
        dl = _dl.create_b1k_data_loader(cfg_small, shuffle=True, num_batches=1)
        torch_dl = dl._data_loader._data_loader  # DataLoaderImpl -> TorchDataLoader -> torch DataLoader
        sampler = torch_dl.sampler
        import torch.utils.data as tud
        assert isinstance(sampler, tud.WeightedRandomSampler), f"{arm}: sampler is {type(sampler)} — weighting NOT active"
        w = np.asarray(sampler.weights, dtype=np.float64)
        n = len(w)
        assert n == info["total_frames"], f"{arm}: sampler len {n} != total_frames {info['total_frames']}"
        # per-source effective mass via episode ranges -> frame ranges (episodes are contiguous in the merge)
        import pyarrow.parquet as pq
        import glob
        eps = np.concatenate([pq.read_table(f, columns=["episode_index"])["episode_index"].to_numpy()
                              for f in sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))])
        assert len(eps) == n
        rep = {"arm": arm, "root": str(root), "total_frames": int(n), "env_sample_weight_col": os.environ.get("B1K_SAMPLE_WEIGHT_COL", ""),
               "frac_frames_below_1": float((w < 1).mean()), "frac_frames_above_1": float((w > 1).mean()),
               "w_min": float(w.min()), "w_max": float(w.max()), "sources": []}
        for s in srcs:
            m = (eps >= s["episodes"][0]) & (eps <= s["episodes"][1])
            rep["sources"].append({"source": s["source"], "frames": int(m.sum()), "frame_frac": float(m.mean()),
                                   "effective_mass_frac": float(w[m].sum() / w.sum()),
                                   "mean_weight": float(w[m].mean())})
        if arm == "a0":
            assert (w < 1).sum() == 0, "a0 must not be down-weighted"
        else:
            assert (w < 1).sum() > 0, f"{arm}: expected poison down-weighting"
        # NORMALIZED-MAGNITUDE GATE (added 2026-09-12 after the A5 divergence): a source whose
        # action/state differs on a dim that is ~constant in the norm stats (std ~ 1e-10, e.g. torso
        # joint 4) normalizes to ~1e6 and blows the loss up from step 0. Fail early, per source.
        import json as _json
        _ns = _json.loads((pathlib.Path(cfg.assets_dirs) / "b1k_radio" / "norm_stats.json").read_text())["norm_stats"]
        _am, _as = np.asarray(_ns["actions"]["mean"]), np.asarray(_ns["actions"]["std"])
        _sm, _ss = np.asarray(_ns["state"]["mean"]), np.asarray(_ns["state"]["std"])
        import pyarrow.compute as _pc
        for s in srcs:
            _fs = sorted(glob.glob(str(pathlib.Path(s["source"]) / "data" / "**" / "*.parquet"), recursive=True))
            _t = pq.read_table(_fs[0], columns=["action", "observation.state"])
            _a = _pc.list_flatten(_t["action"]).to_numpy(zero_copy_only=False).reshape(_t.num_rows, -1)
            _st = _pc.list_flatten(_t["observation.state"]).to_numpy(zero_copy_only=False).reshape(_t.num_rows, -1)
            _za = np.abs((_a[:, :_am.size] - _am) / (_as + 1e-6)).max(0)
            _zs = np.abs((_st[:, :_sm.size] - _sm) / (_ss + 1e-6)).max(0)
            # 2026-09-15: a z-score is only dangerous when the RAW offset is non-negligible. The Run-3 factory/episodes
            # sources carry <= 2e-4 rad on torso joint 4 (z ~ 140-210 through the +1e-6 epsilon): a constant the model
            # learns in a few steps (A2-A4 converged on it; the joint is locked at eval, 2e-4 rad has no effect). The old
            # A5 clips had 1.76 rad VARYING there (z ~ 1e6) -> divergence. Gate on z AND raw offset (actions: > 1e-2).
            _ra = np.abs(_a[:, :_am.size] - _am).max(0); _rs = np.abs(_st[:, :_sm.size] - _sm).max(0)
            _bad = [(f"action d{i}", float(v), float(_ra[i])) for i, v in enumerate(_za) if v > 50 and _ra[i] > 1e-2] + \
                   [(f"state d{i}", float(v), float(_rs[i])) for i, v in enumerate(_zs) if v > 5000 and _rs[i] > 1e-2]
            _negl = [(f"action d{i}", round(float(v), 1), f"raw {float(_ra[i]):.1e}") for i, v in enumerate(_za) if v > 50 and _ra[i] <= 1e-2]
            if _negl: print(f"    norm-gate {pathlib.Path(s['source']).name:22s} NEGLIGIBLE-OFFSET dims (z>50 but raw<=1e-2, learnable constant): {_negl}", flush=True)
            assert not _bad, f"{arm}: {pathlib.Path(s['source']).name} has normalized magnitudes that would explode the loss (dim, z, raw): {_bad}"
            print(f"    norm-gate {pathlib.Path(s['source']).name:22s} max|z| action {_za.max():.1f} state {_zs.max():.1f}", flush=True)
        obs, act = next(iter(dl))
        gd = np.asarray(obs.gt_depth)
        assert gd.shape[-1] == 768 and (gd > 0).mean() > 0.5, f"gt_depth implausible {gd.shape} valid {(gd>0).mean():.3f}"
        assert obs.stage is not None and obs.target_points is not None and obs.map_tokens is not None
        rep["batch"] = {"state": list(obs.state.shape), "actions": list(act.shape), "gt_depth_valid": float((gd > 0).mean())}
        (out_dir / f"preflight_{arm}.json").write_text(json.dumps(rep, indent=2))
        print(f"B PASS {arm}: " + json.dumps({k: v for k, v in rep.items() if k != "sources"}), flush=True)
        for s in rep["sources"]:
            print(f"    {pathlib.Path(s['source']).name:22s} frames {s['frames']:7d} ({s['frame_frac']:.3%})  "
                  f"effective mass {s['effective_mass_frac']:.3%}  mean w {s['mean_weight']:.2f}", flush=True)
    print("PREFLIGHT_RUN3_PASS", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["a0", "a1", "a2", "a3", "a4"])

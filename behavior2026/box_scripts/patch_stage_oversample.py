"""Stage-transition oversampling for Run 1 (the run-4 exhibit's fix).

Grasp-descent initiation lives in the ACQUIRE->MANIPULATE transition frames — a few dozen per
episode, ~1.5% of data, and the one behavior every legal-baseline failure lacked (run 4: 88%
injection, conf 0.91, arm never extended). This patch weights the sampler so frames within
+/-25 steps of a 1->2 stage transition are drawn OVERSAMPLE_X more often (default 8x -> such
frames become ~10-12% of each batch instead of ~1.5%).

Mechanics: create_b1k_data_loader builds per-index weights from the dataset's stage column
(parquet row order == dataset index order — ASSERTED against dataset length; the loader smoke
also spot-verifies alignment) and passes a WeightedRandomSampler. Toggle via env var
B1K_STAGE_OVERSAMPLE (unset/0 = stock uniform shuffle — the A/B off-arm).
"""

import py_compile

P = "/root/openpi_fork/src/openpi/training/data_loader.py"
s = open(P).read()
if "B1K_STAGE_OVERSAMPLE" in s:
    raise SystemExit("already patched")

anchor = """    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = create_b1k_dataset(data_config=data_config, action_horizon=config.model.action_horizon)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
    )"""
assert anchor in s, "create_b1k_data_loader anchor not found"
repl = """    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = create_b1k_dataset(data_config=data_config, action_horizon=config.model.action_horizon)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Stage-transition oversampling (patch_stage_oversample.py): weight frames near the
    # ACQUIRE->MANIPULATE boundary (grasp-descent initiations). Env-gated for clean A/B.
    sampler = None
    _ov = float(os.environ.get("B1K_STAGE_OVERSAMPLE", "0") or 0)
    if _ov > 1 and shuffle:
        import glob as _glob

        import numpy as _np
        import pyarrow.parquet as _pq
        import torch.utils.data as _tud

        _eps, _stg = [], []
        for _fp in sorted(_glob.glob(os.path.join(data_config.dataset_root, "data", "**", "*.parquet"),
                                     recursive=True)):
            _t = _pq.read_table(_fp, columns=["episode_index", "stage"])
            _eps.append(_t["episode_index"].to_numpy())
            _stg.append(_t["stage"].to_numpy())
        _eps = _np.concatenate(_eps)
        _stg = _np.concatenate(_stg)
        _w = _np.ones(len(_stg), _np.float64)
        _trans = _np.where((_stg[1:] == 2) & (_stg[:-1] == 1) & (_eps[1:] == _eps[:-1]))[0] + 1
        for _ti in _trans:
            _lo, _hi = max(0, _ti - 25), min(len(_w), _ti + 26)
            _w[_lo:_hi] = _ov
        _n = len(dataset)
        assert _n == len(_w), f"sampler/dataset misalignment: {_n} vs {len(_w)}"
        sampler = _tud.WeightedRandomSampler(_w.tolist(), num_samples=_n, replacement=True)
        print(f"[stage-oversample] {_ov}x on {len(_trans)} transitions; "
              f"weighted frames {(_w > 1).mean():.1%} of {_n}")

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
    )"""
s = s.replace(anchor, repl, 1)

# TorchDataLoader already accepts sampler (DDP support) — no class patch needed.
if "import os" not in s.split("\n\n")[0] and "\nimport os\n" not in s:
    s = s.replace("import jax\n", "import os\n\nimport jax\n", 1)

compile(s, P, "exec")
open(P, "w").write(s)
py_compile.compile(P, doraise=True)
print("stage-oversampling patched into create_b1k_data_loader (env B1K_STAGE_OVERSAMPLE)")

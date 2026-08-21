import sys, os, logging, itertools
sys.path.insert(0, "/root/openpi/src")
logging.basicConfig(level=logging.INFO, format="%(message)s")
import numpy as np
import openpi.training.config as _config
import openpi.training.data_loader as dl

cfg = _config.get_config("pi05_phaseA_point")
dc = cfg.data.create(cfg.assets_dirs, cfg.model)
ds = dl.transform_dataset(dl.create_b1k_dataset(data_config=dc, action_horizon=cfg.model.action_horizon), dc)

for flag in ("0", "1"):
    os.environ["B1K_CONTACT_WEIGHTS"] = flag
    s = dl._b1k_contact_sampler(ds, seed=0)
    print(f"--- B1K_CONTACT_WEIGHTS={flag}: sampler={type(s).__name__ if s else None}")
    if s is None:
        continue
    # empirical check: what fraction of draws land on commit frames?
    hi = set(s._hi.tolist())
    draws = list(itertools.islice(iter(s), 200_000))
    frac = sum(1 for d in draws if int(d) in hi) / len(draws)
    print(f"    empirical commit-frame share of draws: {frac*100:.1f}%  (target ~28%)")
    print(f"    distinct indices in 200k draws: {len(set(int(d) for d in draws))}")

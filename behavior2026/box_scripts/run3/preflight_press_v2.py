"""CPU preflight for the v2-label press config: build the loader from `pi05_radio_press` with dataset_root pointed at a
v2 mix, pull real batches and check that stage/progress/target_points come from the v2 columns.
  PYTHONPATH=<patched fork src> JAX_PLATFORMS=cpu python preflight_press_v2.py /root/b1k_radio_mix_a5_v2
"""
import dataclasses, pathlib, sys
import numpy as np
from openpi.training import config as _config
from openpi.training import data_loader as _dl

root = pathlib.Path(sys.argv[1]); n_batches = int(sys.argv[2]) if len(sys.argv) > 2 else 6
cfg = _config.get_config("pi05_radio_press")
assert cfg.data.stage_key == "stage_v2" and cfg.data.progress_key == "progress" and cfg.data.target_points_key == "target_points_v2", "press config not on v2 keys"
cfg = dataclasses.replace(cfg, batch_size=32, num_workers=0, data=dataclasses.replace(cfg.data, base_config=dataclasses.replace(cfg.data.base_config, dataset_root=str(root))))
dl = _dl.create_b1k_data_loader(cfg, shuffle=True, num_batches=n_batches)
stages = []; progs = []; tps = []; toks = []
for obs, act in dl:
    stages.append(np.asarray(obs.stage).reshape(-1)); progs.append(np.asarray(obs.progress).reshape(-1))
    tps.append(np.asarray(obs.target_points).reshape(-1, 2, 3)); toks.append(np.asarray(obs.stage_tokens).reshape(-1, 2))
st = np.concatenate(stages); pr = np.concatenate(progs); tp = np.concatenate(tps); tk = np.concatenate(toks)
print("batches", len(stages), "frames", len(st))
print("stage_v2 histogram", np.bincount(st, minlength=4).tolist(), "| stage_tokens == stage broadcast:", bool((tk[:, 0] == st).all() and (tk[:, 1] == st).all()))
print("progress: min %.3f max %.3f | mean by stage %s" % (pr.min(), pr.max(), [round(float(pr[st == c].mean()), 3) if (st == c).any() else None for c in range(4)]))
zR = tp[:, 1, 2]
print("target z (R) mean by stage:", [round(float(zR[st == c].mean()), 3) if (st == c).any() else None for c in range(4)], "(pre-lift should sit ~0.141 above post-lift buttons)")
ok = st.min() >= 0 and st.max() <= 3 and 0 <= pr.min() and pr.max() <= 1 and (tk[:, 0] == st).all()
print("PREFLIGHT_PRESS_V2_PASS" if ok else "PREFLIGHT_PRESS_V2_FAIL")

"""CPU preflight for the FULL-STACK config (pi05_radio_full): real batches from a mix that carries the v2 labels AND the
temporal columns (gist_head / hist_geo). Checks the v2 keys (as preflight_press_v2) plus the history stack: shapes
[K+1, 2048], validity masks (episode starts padded), flow-target magnitudes, and that gists are not constant.
  PYTHONPATH=<fork src> JAX_PLATFORMS=cpu python preflight_full.py /root/b1k_radio_mix_full [n_batches]
"""
import dataclasses, pathlib, sys
import numpy as np
from openpi.training import config as _config
from openpi.training import data_loader as _dl

root = pathlib.Path(sys.argv[1]); n_batches = int(sys.argv[2]) if len(sys.argv) > 2 else 6
cfg = _config.get_config("pi05_radio_full")
assert cfg.model.temporal_conditioning and cfg.data.stage_key == "stage_v2", "not the full config"
cfg = dataclasses.replace(cfg, batch_size=32, num_workers=0, data=dataclasses.replace(cfg.data, base_config=dataclasses.replace(cfg.data.base_config, dataset_root=str(root))))
dl = _dl.create_b1k_data_loader(cfg, shuffle=True, num_batches=n_batches)
K = cfg.model.temporal_k
st = []; pr = []; gists = []; masks = []; flows = []; fmasks = []
for obs, act in dl:
    st.append(np.asarray(obs.stage).reshape(-1)); pr.append(np.asarray(obs.progress).reshape(-1))
    gists.append(np.asarray(obs.history_gists)); masks.append(np.asarray(obs.history_mask)); flows.append(np.asarray(obs.hist_flow)); fmasks.append(np.asarray(obs.hist_flow_mask))
st = np.concatenate(st); pr = np.concatenate(pr); g = np.concatenate(gists); m = np.concatenate(masks); fl = np.concatenate(flows); fm = np.concatenate(fmasks)
print("frames", len(st), "| history_gists", g.shape, "history_mask", m.shape, "hist_flow", fl.shape, "hist_flow_mask", fm.shape)
print("stage_v2 histogram", np.bincount(st, minlength=4).tolist(), "| progress range %.3f..%.3f" % (pr.min(), pr.max()))
print("valid slots per sample: mean %.2f of %d | current slot valid: %.3f | all-past-valid fraction: %.3f" % (m.sum(1).mean(), K + 1, m[:, -1].mean(), m[:, :-1].all(1).mean()))
gn = np.linalg.norm(g, axis=-1); print("gist norms (valid slots): mean %.1f std %.1f | zero gists among valid: %d" % (gn[m].mean(), gn[m].std(), int((gn[m] < 1e-6).sum())))
diff = np.linalg.norm(g[:, -1] - g[:, -2], axis=-1); print("|gist_t - gist_t-32| mean %.2f (should be > 0: frames differ)" % diff[m[:, -2] & m[:, -1]].mean())
valid = fm.astype(bool); ee = np.linalg.norm(fl[..., 0:3], axis=-1); bt = np.linalg.norm(fl[..., 6:9], axis=-1)
print("EE_L flow |current-past| by offset (valid): %s m | button flow: %s m" % ([round(float(ee[:, k][valid[:, k]].mean()), 3) if valid[:, k].any() else None for k in range(K)], [round(float(bt[:, k][valid[:, k]].mean()), 3) if valid[:, k].any() else None for k in range(K)]))
ok = g.shape[1:] == (K + 1, cfg.model.temporal_gist_dim) and fl.shape[1:] == (K, 9) and m[:, -1].all() and (gn[m] > 1.0).all() and np.isfinite(fl).all()
print("PREFLIGHT_FULL_PASS" if ok else "PREFLIGHT_FULL_FAIL")

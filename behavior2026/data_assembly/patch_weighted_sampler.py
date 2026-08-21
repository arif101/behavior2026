"""Enable CONTACT OVERSAMPLING: use the per-frame `sample_weight` column (written by the
Phase-A conversion) via a WeightedRandomSampler. Without this the column is INERT.

Measured justification: commit instants (gripper open<->close) are 3.12% of frames; uniform
sampling spends ~97% of gradient budget on transit. Scaling episodes does NOT change that
ratio — weighted sampling does. Gated by env B1K_CONTACT_WEIGHTS=1 so the A/B is a clean
on/off with identical data. Idempotent."""
P = "/root/openpi/src/openpi/training/data_loader.py"
src = open(P).read()
if "B1K_CONTACT_WEIGHTS" in src:
    print("already patched"); raise SystemExit

old = """    sampler = None
    if framework == "pytorch":"""
new = '''    sampler = None
    # --- contact oversampling (A/B via env B1K_CONTACT_WEIGHTS=1) ---
    if os.environ.get("B1K_CONTACT_WEIGHTS") == "1":
        import numpy as _np
        _w = None
        try:
            _hf = getattr(getattr(dataset, "_dataset", dataset), "hf_dataset", None)
            if _hf is not None and "sample_weight" in _hf.column_names:
                _w = _np.asarray(_hf["sample_weight"], dtype=_np.float64)
        except Exception as _e:
            logging.warning(f"contact weights: could not read sample_weight ({_e})")
        if _w is not None and len(_w) == len(dataset) and float(_w.max()) > 1.0:
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=torch.as_tensor(_w, dtype=torch.double),
                num_samples=len(dataset), replacement=True)
            logging.info(
                f"CONTACT OVERSAMPLING ON: {(_w > 1).mean()*100:.2f}% of frames weighted "
                f"{_w.max():.2f}x -> ~{(_w[_w>1].sum()/_w.sum())*100:.1f}% of draws")
        else:
            logging.warning("contact weights requested but sample_weight unusable — uniform sampling")
    if sampler is None and framework == "pytorch":'''
assert old in src
src = src.replace(old, new, 1)
if "\nimport os" not in src.split("def ")[0]:
    src = src.replace("import logging", "import logging\nimport os", 1)
open(P, "w").write(src)
import ast; ast.parse(src)
print("patched: WeightedRandomSampler gated by B1K_CONTACT_WEIGHTS; syntax OK")

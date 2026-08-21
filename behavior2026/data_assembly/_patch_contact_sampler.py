"""Fix contact oversampling: put it on the code path train_b1k.py actually uses, and
sample without torch.multinomial (which caps at 2**24 categories < our 18.8M frames).

Two bugs being fixed:
  1. The weights were applied in create_torch_data_loader(), but train_b1k.py calls
     create_b1k_data_loader() -- a different function that builds TorchDataLoader directly.
     The env var was set, the column was present, and NOTHING happened: no log line either way.
  2. Even on the right path, WeightedRandomSampler -> torch.multinomial would have raised
     "number of categories cannot exceed 2**24" (16,777,216) on 18,837,213 frames.

Idempotent: re-running is a no-op.
"""
import re
import pathlib

P = pathlib.Path("/root/openpi/src/openpi/training/data_loader.py")
src = P.read_text()
orig = src

HELPER = '''
class _TwoTierWeightedSampler(torch.utils.data.Sampler):
    """Exact weight-proportional sampling for a TWO-VALUED weight column.

    torch's WeightedRandomSampler is unusable at this scale: it calls torch.multinomial,
    which caps at 2**24 categories, and this dataset has ~18.8M frames. Because the weights
    take only two values (1.0 on ordinary frames, w_hi on gripper-commit frames), drawing
    proportional to weight is exactly equivalent to a Bernoulli choice between the two index
    pools followed by a uniform draw inside the chosen pool. No category cap, O(1) per draw.
    """

    def __init__(self, hi_idx, lo_idx, p_hi: float, num_samples: int, seed: int = 0):
        import numpy as _np

        self._hi = _np.asarray(hi_idx, dtype=_np.int64)
        self._lo = _np.asarray(lo_idx, dtype=_np.int64)
        self._p_hi = float(p_hi)
        self._n = int(num_samples)
        self._seed = int(seed)
        self._epoch = 0

    def __len__(self) -> int:
        return self._n

    def __iter__(self):
        import numpy as _np

        rng = _np.random.default_rng(self._seed + self._epoch)
        self._epoch += 1
        n_hi = int(rng.binomial(self._n, self._p_hi))
        out = _np.empty(self._n, dtype=_np.int64)
        out[:n_hi] = self._hi[rng.integers(0, len(self._hi), n_hi)]
        out[n_hi:] = self._lo[rng.integers(0, len(self._lo), self._n - n_hi)]
        rng.shuffle(out)
        return iter(out)


def _b1k_contact_sampler(dataset, seed: int = 0):
    """Build the contact-oversampling sampler, or None to fall back to uniform.

    Gated on B1K_CONTACT_WEIGHTS=1 so the A/B differs in exactly one variable. Every
    fallback path logs LOUDLY -- a silent fallback would make the two arms identical
    while still labelling one of them "weighted".
    """
    import os as _os

    if _os.environ.get("B1K_CONTACT_WEIGHTS") != "1":
        return None
    import numpy as _np

    try:
        node = dataset
        hf = None
        for _ in range(6):
            hf = getattr(node, "hf_dataset", None)
            if hf is not None:
                break
            nxt = getattr(node, "_dataset", None) or getattr(node, "dataset", None)
            if nxt is None or nxt is node:
                break
            node = nxt
        if hf is None or "sample_weight" not in getattr(hf, "column_names", []):
            logging.error("CONTACT WEIGHTS REQUESTED BUT NO sample_weight COLUMN — uniform sampling")
            return None
        w = hf.data.column("sample_weight").to_numpy(zero_copy_only=False).astype(_np.float64)
    except Exception as e:
        logging.error(f"CONTACT WEIGHTS REQUESTED BUT UNREADABLE ({e}) — uniform sampling")
        return None

    if len(w) != len(dataset):
        logging.error(
            f"CONTACT WEIGHTS LENGTH MISMATCH ({len(w)} vs dataset {len(dataset)}) — uniform sampling"
        )
        return None
    uniq = _np.unique(w)
    if len(uniq) != 2 or float(uniq.max()) <= 1.0:
        logging.error(f"CONTACT WEIGHTS NOT TWO-VALUED (values={uniq[:5]}) — uniform sampling")
        return None

    hi = _np.flatnonzero(w > 1.0)
    lo = _np.flatnonzero(w <= 1.0)
    w_hi = float(uniq.max())
    p_hi = (w_hi * len(hi)) / (w_hi * len(hi) + len(lo))
    logging.info(
        "CONTACT OVERSAMPLING ON: %.2f%% of frames weighted %.2fx -> %.1f%% of draws "
        "(%d commit / %d ordinary frames)",
        100.0 * len(hi) / len(w),
        w_hi,
        100.0 * p_hi,
        len(hi),
        len(lo),
    )
    return _TwoTierWeightedSampler(hi, lo, p_hi, len(dataset), seed=seed)


'''

# --- 1. insert helper before create_b1k_data_loader -------------------------
anchor = "def create_b1k_data_loader(\n"
if "_TwoTierWeightedSampler" not in src:
    assert anchor in src, "anchor create_b1k_data_loader not found"
    src = src.replace(anchor, HELPER.lstrip("\n") + anchor, 1)

# --- 2. wire it into create_b1k_data_loader ---------------------------------
old_call = """    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
    )"""
new_call = """    sampler = _b1k_contact_sampler(dataset, seed=config.seed)

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        sampler=sampler,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
    )"""
if "sampler=sampler," not in src.split("def create_torch_data_loader")[0]:
    assert old_call in src, "create_b1k_data_loader TorchDataLoader call not found verbatim"
    src = src.replace(old_call, new_call, 1)

# --- 3. replace the old (wrong-path, multinomial-capped) block --------------
old_block = re.search(
    r"[ \t]*# --- contact oversampling.*?uniform sampling\"\)\n",
    src,
    re.DOTALL,
)
if old_block:
    src = src.replace(
        old_block.group(0),
        "    sampler = _b1k_contact_sampler(dataset, seed=seed)\n",
        1,
    )

P.write_text(src)
import ast

ast.parse(src)
print("PATCH OK — changed:", src != orig)
print("  helper present:", "_TwoTierWeightedSampler" in src)
print("  wired into create_b1k_data_loader:", "sampler=sampler," in src)
print("  old multinomial block gone:", "WeightedRandomSampler" not in src)

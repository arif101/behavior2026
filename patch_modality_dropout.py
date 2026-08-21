"""Patch the openpi fork with MODALITY DROPOUT (train-only, Run-2 mandate).

Anchored, assertion-guarded edits (the patch_*.py convention — never blind-overwrite fork
files, they carry box-only changes). Every file is compile()d BEFORE writing and
py_compile'd after, so a bad edit never lands on disk.

WHY: Run-2 stacks conditioning streams (target points -> AdaLN, map geometry -> AdaLN, map
prefix tokens, wrist cams). Independent per-stream dropout p≈0.2 keeps the policy robust to
any stream being absent/degraded at serve and stops the stack from collapsing onto a single
crutch stream. This GENERALIZES patch_antishortcut.py (which hardcoded P(drop point)=0.4,
P(drop wrist)=0.2 for run1b): enable EXACTLY ONE of anti_shortcut / modality_dropout_p in a
config — both firing would compound drop rates.

Streams dropped, each independently with prob p per sample (head cam NEVER dropped):
  target_points  -> mask=False        (the model embeds the learned per-arm null token)
  map_tokens     -> all-zero tokens   (the learned null-token convention, b1k_policy.py;
                                       geo-AdaLN sees zeros -> its learned 'blank map' bias;
                                       the map_recon/aux not_blind gate already treats
                                       zero-T0 rows as blind, so aux losses stay consistent)
  wrist cams     -> image_mask=False  (per camera; the GT-depth aux weights by image_masks
                                       AFTER this block, so dropped cams also stop
                                       contributing depth-aux gradient)

RNG DISCIPLINE: all randomness derives from jax.random.fold_in(rng, 8206) — the
preprocess/noise/time streams stay bit-identical to stock when the flag is off AND when it
is on (the fold_in precedent of the point-noise rng).

  1. models/pi0_config.py   modality_dropout_p field (default 0.0 / inert)
  2. models/pi0.py          attr in __init__ (no params — checkpoint-trivial) + the dropout
                            block in compute_loss BEFORE the point-noise block.

  models/model.py, policies/b1k_policy.py, training/config.py, training/data_loader.py,
  training/weight_loaders.py need NO changes (no new data keys, no new params).

Usage:  FORK_ROOT=/path/to/src/openpi python3 patch_modality_dropout.py
        (default FORK_ROOT = /root/openpi_fork/src/openpi)
"""

import os
import py_compile

FORK = os.environ.get("FORK_ROOT", "/root/openpi_fork/src/openpi")

MARKER = "patch_modality_dropout"


def patch(path, subs):
    s = open(path).read()
    if MARKER in s:
        print(f"SKIP {path}: already patched")
        return
    for pat, rep in subs:
        assert pat in s, f"anchor not found in {path}: {pat[:70]!r}"
        assert s.count(pat) == 1, f"anchor not unique in {path}: {pat[:70]!r}"
        s = s.replace(pat, rep, 1)
    compile(s, path, "exec")  # syntax-check BEFORE writing — a bad edit never lands on disk
    open(path, "w").write(s)
    py_compile.compile(path, doraise=True)
    print(f"patched {path}")


# ---- 1. models/pi0_config.py ----------------------------------------------------------
A = '    pytorch_compile_mode: str | None = "max-autotune"\n'
patch(f"{FORK}/models/pi0_config.py", [(
    A,
    "    # MODALITY DROPOUT (patch_modality_dropout.py): train-only independent per-stream\n"
    "    # drop prob (target_points -> null mask, map_tokens -> zeros/null, each wrist cam\n"
    "    # -> image_mask False; head cam never dropped). Generalizes anti_shortcut — enable\n"
    "    # exactly ONE of the two. 0.0 => byte-identical behavior to baseline.\n"
    "    modality_dropout_p: float = 0.0\n"
    + A,
)])

# ---- 2. models/pi0.py — attr + compute_loss block -------------------------------------
INIT_ANCHOR = (
    "        # This attribute gets automatically set by model.train() and model.eval().\n"
    "        self.deterministic = True\n"
)
INIT_BLOCK = (
    "        # MODALITY DROPOUT (patch_modality_dropout.py): no params — train-only input\n"
    "        # regularizer; checkpoint restore untouched at any flag value.\n"
    '        self.modality_dropout_p = getattr(config, "modality_dropout_p", 0.0)\n'
    "\n"
)
LOSS_ANCHOR = (
    "        if self.point_conditioning and train and self.point_noise_std > 0 and observation.target_points is not None:\n"
)
LOSS_BLOCK = (
    "        # MODALITY DROPOUT (patch_modality_dropout.py): independent per-stream drop,\n"
    "        # train-only. Runs BEFORE point noise (noise on a dropped point is discarded by\n"
    "        # the mask) and BEFORE the loss blocks that read image_masks (GT-depth aux).\n"
    "        # fold_in(8206) keeps every stock rng stream bit-identical (point-noise\n"
    "        # precedent). Enable exactly ONE of anti_shortcut / modality_dropout_p.\n"
    "        if self.modality_dropout_p > 0 and train:\n"
    "            import dataclasses as _mdc\n"
    "            _mdp = self.modality_dropout_p\n"
    "            _mdb = observation.state.shape[0]\n"
    "            _r_pt, _r_map, _r_wl, _r_wr = jax.random.split(jax.random.fold_in(rng, 8206), 4)\n"
    "            if observation.target_points_mask is not None:\n"
    "                _mdkeep = jax.random.bernoulli(_r_pt, 1.0 - _mdp, (_mdb,))\n"
    "                observation = _mdc.replace(\n"
    "                    observation,\n"
    "                    target_points_mask=jnp.logical_and(\n"
    "                        observation.target_points_mask, _mdkeep[:, None]),\n"
    "                )\n"
    "            if observation.map_tokens is not None:\n"
    "                _mdkeepm = jax.random.bernoulli(_r_map, 1.0 - _mdp, (_mdb,))\n"
    "                observation = _mdc.replace(\n"
    "                    observation,\n"
    "                    map_tokens=observation.map_tokens\n"
    "                    * _mdkeepm[:, None, None].astype(observation.map_tokens.dtype),\n"
    "                )\n"
    "            _mdmasks = dict(observation.image_masks)\n"
    '            for _mdr, _mdnm in ((_r_wl, "left_wrist_0_rgb"), (_r_wr, "right_wrist_0_rgb")):\n'
    "                if _mdnm in _mdmasks:\n"
    "                    _mdkc = jax.random.bernoulli(_mdr, 1.0 - _mdp, _mdmasks[_mdnm].shape)\n"
    "                    _mdmasks[_mdnm] = jnp.logical_and(_mdmasks[_mdnm], _mdkc)\n"
    "            observation = _mdc.replace(observation, image_masks=_mdmasks)\n"
    "\n"
)
patch(f"{FORK}/models/pi0.py", [
    (INIT_ANCHOR, INIT_BLOCK + INIT_ANCHOR),
    (LOSS_ANCHOR, LOSS_BLOCK + LOSS_ANCHOR),
])

print("no changes needed: models/model.py, policies/b1k_policy.py, training/config.py,")
print("  training/data_loader.py, training/weight_loaders.py")
print("REMINDER: enable exactly ONE of anti_shortcut / modality_dropout_p in a TrainConfig")
print("ALL MODALITY-DROPOUT PATCHES APPLIED")

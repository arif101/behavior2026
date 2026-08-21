"""Patch the openpi fork with a GT-DEPTH auxiliary head on the prefix image tokens.

Anchored, assertion-guarded edits (the patch_*.py convention — never blind-overwrite fork
files, they carry box-only changes). Every file is compile()d BEFORE writing and
py_compile'd after, so a bad edit never lands on disk.

WHY (Run-2 mandate + the aux-leak law): the policy input is RGB-ONLY; depth exists in the
dataset (12-bit log videos, sensor-derived, train+eval legal) but never reaches the model.
Predicting per-patch depth from the prefix image-token features forces genuine 3D reading,
and the target is unobtainable from any other input stream (the α-stall lesson: an aux
target derivable elsewhere is a leak, not a teacher). Depth stays a LABEL, never an input.

LABEL CONTRACT: a per-row parquet column  gt_depth_ds  = float32 (768,) = 3 cameras x 16x16
patch-mean METRIC DEPTH in meters, camera order [head zed, left wrist, right wrist] (the
prefix image-token order), row-major v-then-u per camera; 0.0 = invalid (no depth / masked
patch). Produced OFFLINE by box_scripts/add_depth_aux_labels.py (decodes the depth videos
once at dataset-assembly time — no train-time video decode). KNOWN APPROXIMATION: train-time
image augmentation (0.95 crop + ±5° rotate on the head cam) is not applied to the target —
the same accepted misalignment as the aux_pixels heatmap CE.

  1. models/pi0_config.py   depth_aux / depth_aux_weight fields (default-off / inert)
  2. models/model.py        Observation.gt_depth + from_dict + preprocess passthrough
  3. models/pi0.py          head params in __init__ (AFTER all existing modules: rng stream
                            of every existing param unchanged; no params when
                            depth_aux=False -> run1b restore bit-identical) + masked L1 on
                            log1p(depth) in compute_loss (loss-inert without labels)
  4. policies/b1k_policy.py B1KInputs depth_aux field + gt_depth packing
  5. training/config.py     gated repack gt_depth <- gt_depth_ds + B1KInputs kwarg.
                            !! RepackTransform KeyErrors on absent columns: EVERY dataset in
                            a depth_aux=True mix must carry gt_depth_ds first.

  training/data_loader.py and training/weight_loaders.py need NO changes. Warm-starting a
  depth_aux=True run from a headless checkpoint requires the TrainConfig's
  CheckpointWeightLoader to include ".*depth_aux.*" in missing_regex.

Usage:  FORK_ROOT=/path/to/src/openpi python3 patch_depth_aux.py
        (default FORK_ROOT = /root/openpi_fork/src/openpi)
"""

import os
import py_compile

FORK = os.environ.get("FORK_ROOT", "/root/openpi_fork/src/openpi")

MARKER = "patch_depth_aux"


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
    "    # GT-DEPTH auxiliary head (patch_depth_aux.py): per-patch log1p(depth) L1 from the\n"
    "    # prefix image-token features (16x16 per camera), masked on validity + image mask.\n"
    "    # Depth is a LABEL only (policy input stays RGB-only). False => no params created,\n"
    "    # init and checkpoint restore bit-identical to baseline (run1b safe).\n"
    "    depth_aux: bool = False\n"
    "    depth_aux_weight: float = 0.05\n"
    + A,
)])

# ---- 2. models/model.py — 'gt_depth' trap-triple --------------------------------------
A1 = '    aux_pixels: at.Float[ArrayT, "*b 9"] | None = None\n'
A2 = '            aux_pixels=data.get("aux_pixels"),\n'
A3 = "        aux_pixels=observation.aux_pixels,\n"
patch(f"{FORK}/models/model.py", [
    (A1,
     A1
     + "    # GT-depth aux label (patch_depth_aux.py): 3 cams x 16x16 patch-mean metric depth\n"
     "    # (meters, 0=invalid), camera order = prefix image-token order. Label-only.\n"
     '    gt_depth: at.Float[ArrayT, "*b 768"] | None = None\n'),
    (A2, A2 + '            gt_depth=data.get("gt_depth"),\n'),
    (A3, A3 + "        gt_depth=observation.gt_depth,\n"),
])

# ---- 3. models/pi0.py — head params + loss --------------------------------------------
INIT_ANCHOR = (
    "        # This attribute gets automatically set by model.train() and model.eval().\n"
    "        self.deterministic = True\n"
)
INIT_BLOCK = (
    "        # GT-DEPTH aux head (patch_depth_aux.py): tiny MLP shared across all prefix\n"
    "        # image tokens -> per-patch log1p(depth). Created AFTER every stock/optional\n"
    "        # module above so the rng stream (and thus the init) of all existing params is\n"
    "        # unchanged; depth_aux=False creates no params at all, so restoring run1b\n"
    "        # checkpoints is bit-identical. Warm-starting WITH the head needs missing_regex\n"
    "        # '.*depth_aux.*' in the CheckpointWeightLoader.\n"
    '        self.depth_aux = getattr(config, "depth_aux", False)\n'
    "        if self.depth_aux:\n"
    "            self.depth_aux_weight = config.depth_aux_weight\n"
    "            self.depth_aux_in = nnx.Linear(paligemma_config.width, 64, rngs=rngs)\n"
    "            self.depth_aux_out = nnx.Linear(64, 1, rngs=rngs)\n"
    "\n"
)
LOSS_ANCHOR = "        return loss\n"
LOSS_BLOCK = (
    "        # GT-DEPTH aux (patch_depth_aux.py): masked L1 on log1p(metric depth) over the\n"
    "        # 3x256 prefix image tokens (camera order = image embedding order). Weights:\n"
    "        # validity (gt > 0.01 m) AND the per-camera image mask (a dropped/absent camera\n"
    "        # contributes nothing — composes with modality dropout, which mutates\n"
    "        # image_masks BEFORE this block runs). Per-sample scalar broadcast over the\n"
    "        # horizon dim like every aux loss; loss-inert when the batch has no labels.\n"
    "        if self.depth_aux and observation.gt_depth is not None:\n"
    "            _gd = observation.gt_depth.astype(jnp.float32).reshape(-1, 3, 256)\n"
    "            _dpred = self.depth_aux_out(jax.nn.gelu(self.depth_aux_in(\n"
    "                prefix_out[:, : 3 * 256, :])))[..., 0].astype(jnp.float32).reshape(-1, 3, 256)\n"
    "            _dvalid = (_gd > 0.01).astype(jnp.float32)\n"
    "            _dcmask = jnp.stack(\n"
    "                [observation.image_masks[_c].astype(jnp.float32)\n"
    '                 for _c in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")], axis=1)\n'
    "            _dw = _dvalid * _dcmask[:, :, None]\n"
    "            _derr = jnp.abs(_dpred - jnp.log1p(_gd)) * _dw\n"
    "            _e_depth = _derr.sum((-1, -2)) / jnp.maximum(_dw.sum((-1, -2)), 1.0)\n"
    "            loss = loss + (self.depth_aux_weight * _e_depth).astype(loss.dtype)[:, None]\n"
)
patch(f"{FORK}/models/pi0.py", [
    (INIT_ANCHOR, INIT_BLOCK + INIT_ANCHOR),
    (LOSS_ANCHOR, LOSS_BLOCK + LOSS_ANCHOR),
])

# ---- 4. policies/b1k_policy.py --------------------------------------------------------
F_ANCHOR = "    map_blind_prob: float = 0.3\n"
F_BLOCK = (
    "\n"
    "    # GT-depth aux label passthrough (patch_depth_aux.py; consumed by models/pi0.py\n"
    "    # when the model config sets depth_aux=True). Must match the model config's flag.\n"
    "    depth_aux: bool = False\n"
)
C_ANCHOR = "        return inputs\n"
C_BLOCK = (
    "        # GT-depth aux label (patch_depth_aux.py): the precomputed gt_depth_ds parquet\n"
    "        # column (3 cams x 16x16 patch-mean meters, 0=invalid), added offline by\n"
    "        # box_scripts/add_depth_aux_labels.py. Absent key (serve path) => no label =>\n"
    "        # the aux loss is inert.\n"
    "        if self.depth_aux and data.get(\"gt_depth\") is not None:\n"
    "            inputs[\"gt_depth\"] = np.asarray(data[\"gt_depth\"], np.float32).reshape(768)\n"
    "\n"
)
patch(f"{FORK}/policies/b1k_policy.py", [
    (F_ANCHOR, F_ANCHOR + F_BLOCK),
    (C_ANCHOR, C_BLOCK + C_ANCHOR),
])

# ---- 5. training/config.py ------------------------------------------------------------
R_ANCHOR = "        repack_transform = _transforms.Group(inputs=[_transforms.RepackTransform(repack_mapping)])\n"
R_BLOCK = (
    "        # GT-depth aux (patch_depth_aux.py): repack the precomputed gt_depth_ds column\n"
    "        # only when the model consumes it. !! RepackTransform KeyErrors on absent\n"
    "        # columns — every dataset in a depth_aux=True training mix must have been run\n"
    "        # through add_depth_aux_labels.py first.\n"
    '        if getattr(model_config, "depth_aux", False):\n'
    '            repack_mapping["gt_depth"] = "gt_depth_ds"\n'
)
B_ANCHOR = "                map_tokens_k=map_tokens_k,\n"
B_BLOCK = '                depth_aux=getattr(model_config, "depth_aux", False),\n'
patch(f"{FORK}/training/config.py", [
    (R_ANCHOR, R_BLOCK + R_ANCHOR),
    (B_ANCHOR, B_ANCHOR + B_BLOCK),
])

print("no changes needed: training/data_loader.py, training/weight_loaders.py")
print("REMINDER: a depth_aux=True TrainConfig warm-started from a headless ckpt must set")
print("  CheckpointWeightLoader(..., missing_regex including '.*depth_aux.*')")
print("REMINDER: run add_depth_aux_labels.py over EVERY dataset in the mix BEFORE enabling")
print("ALL DEPTH-AUX PATCHES APPLIED")

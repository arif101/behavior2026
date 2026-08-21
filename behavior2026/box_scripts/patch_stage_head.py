"""Patch the openpi fork with a STAGE/PROGRESS PREDICTION HEAD on the action expert.

Anchored, assertion-guarded edits (the patch_*.py convention — never blind-overwrite fork
files, they carry box-only changes). Every file is compile()d BEFORE writing and
py_compile'd after, so a bad edit never lands on disk.

  1. models/pi0_config.py   stage_head / stage_classes / stage_loss_weight /
                            progress_loss_weight fields (all default-off / inert)
  2. models/model.py        Observation.progress + from_dict + preprocess passthrough.
                            NOTE: Observation.stage ALREADY exists (patch_aux_v2.py) with
                            its full trap-triple; only 'progress' is new here.
  3. models/pi0.py          head params in __init__ (created AFTER all existing modules so
                            the rng stream of every existing param is unchanged; no params
                            at all when stage_head=False -> run1b restore bit-identical),
                            stage CE + progress MSE added to compute_loss like the existing
                            aux losses. sample_actions returns a bare Actions array (no
                            output-dict convention) -> NO inference-time stage output.
  4. policies/b1k_policy.py B1KInputs stage_head/stage_classes fields + packing of
                            'stage' (parquet column) and 'progress' (explicit column if
                            supplied, else the coarse (stage+0.5)/stage_classes fallback —
                            episode length is NOT available in the row).
  5. training/config.py     repack 'stage' when stage_head is on (idempotent with the map
                            path) + B1KInputs kwargs. NO 'progress' repack: RepackTransform
                            KeyErrors on absent columns and the parquet has no 'progress'.

  training/data_loader.py and training/weight_loaders.py need NO changes. Warm-starting a
  stage_head=True run from a headless checkpoint (e.g. run1b) requires the TrainConfig's
  CheckpointWeightLoader to include ".*stage_head.*" in missing_regex (same recipe as the
  map params in patch_map_fork.py).

Usage:  FORK_ROOT=/path/to/src/openpi python3 patch_stage_head.py
        (default FORK_ROOT = /root/openpi_fork/src/openpi)
"""

import os
import py_compile

FORK = os.environ.get("FORK_ROOT", "/root/openpi_fork/src/openpi")

MARKER = "patch_stage_head"


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
A = "    stage_conditioning: bool = False\n"
patch(f"{FORK}/models/pi0_config.py", [(
    A,
    A
    + "\n"
    "    # STAGE/PROGRESS prediction head (patch_stage_head.py): 2-layer MLP (hidden 256,\n"
    "    # gelu) over mean-pooled action-expert suffix features -> stage_classes logits +\n"
    "    # 1 sigmoid progress scalar, trained with CE / MSE aux losses. False => no params\n"
    "    # created, init and checkpoint restore bit-identical to baseline (run1b safe).\n"
    "    stage_head: bool = False\n"
    "    stage_classes: int = 4\n"
    "    stage_loss_weight: float = 0.05\n"
    "    progress_loss_weight: float = 0.02\n",
)])

# ---- 2. models/model.py — 'progress' trap-triple ('stage' already exists, aux_v2) -----
A1 = '    aux_pixels: at.Float[ArrayT, "*b 9"] | None = None\n'
A2 = '            aux_pixels=data.get("aux_pixels"),\n'
A3 = "        aux_pixels=observation.aux_pixels,\n"
patch(f"{FORK}/models/model.py", [
    (A1,
     A1
     + "    # Normalized episode progress in [0, 1] (patch_stage_head.py). Coarse\n"
     "    # stage-midpoint fallback when no explicit label column exists (see B1KInputs).\n"
     '    progress: at.Float[ArrayT, "*b"] | None = None\n'),
    (A2, A2 + '            progress=data.get("progress"),\n'),
    (A3, A3 + "        progress=observation.progress,\n"),
])

# ---- 3. models/pi0.py — head params + losses ------------------------------------------
INIT_ANCHOR = (
    "        # This attribute gets automatically set by model.train() and model.eval().\n"
    "        self.deterministic = True\n"
)
INIT_BLOCK = (
    "        # STAGE/PROGRESS prediction head (patch_stage_head.py): 2-layer MLP (hidden 256,\n"
    "        # gelu) over mean-pooled action-expert suffix features. Created AFTER every\n"
    "        # stock/optional module above so the rng stream (and thus the init) of all\n"
    "        # existing params is unchanged; stage_head=False creates no params at all, so\n"
    "        # restoring run1b checkpoints is bit-identical. Warm-starting WITH the head\n"
    "        # needs missing_regex '.*stage_head.*' in the CheckpointWeightLoader.\n"
    '        self.stage_head = getattr(config, "stage_head", False)\n'
    "        if self.stage_head:\n"
    "            self.stage_classes = config.stage_classes\n"
    "            self.stage_loss_weight = config.stage_loss_weight\n"
    "            self.progress_loss_weight = config.progress_loss_weight\n"
    "            self.stage_head_in = nnx.Linear(action_expert_config.width, 256, rngs=rngs)\n"
    "            self.stage_head_out = nnx.Linear(256, config.stage_classes + 1, rngs=rngs)\n"
    "\n"
)
LOSS_ANCHOR = "        return loss\n"
LOSS_BLOCK = (
    "        # STAGE/PROGRESS head (patch_stage_head.py): pool the action-expert suffix\n"
    "        # features that feed action_out_proj (mean over the action tokens), one MLP\n"
    "        # trunk -> stage_classes logits + 1 progress channel. Losses are per-sample\n"
    "        # scalars broadcast over the horizon dim exactly like the aux losses above;\n"
    "        # each term is skipped when its label is absent from the batch (serve /\n"
    "        # unlabeled data), so the head is loss-inert without labels.\n"
    "        if self.stage_head:\n"
    "            _sh_feat = suffix_out[:, -self.action_horizon :, :].mean(axis=1)\n"
    "            _sh = self.stage_head_out(jax.nn.gelu(self.stage_head_in(_sh_feat))).astype(jnp.float32)\n"
    "            if observation.stage is not None:\n"
    "                _sh_lsm = jax.nn.log_softmax(_sh[:, : self.stage_classes], axis=-1)\n"
    "                _sh_st = jnp.clip(observation.stage.reshape(-1).astype(jnp.int32), 0, self.stage_classes - 1)\n"
    "                _e_stage = -jnp.take_along_axis(_sh_lsm, _sh_st[:, None], axis=-1)[:, 0]\n"
    "                loss = loss + (self.stage_loss_weight * _e_stage).astype(loss.dtype)[:, None]\n"
    "            if observation.progress is not None:\n"
    "                _sh_prog = jax.nn.sigmoid(_sh[:, self.stage_classes])\n"
    "                _sh_tgt = jnp.clip(observation.progress.reshape(-1).astype(jnp.float32), 0.0, 1.0)\n"
    "                _e_prog = jnp.square(_sh_prog - _sh_tgt)\n"
    "                loss = loss + (self.progress_loss_weight * _e_prog).astype(loss.dtype)[:, None]\n"
)
patch(f"{FORK}/models/pi0.py", [
    (INIT_ANCHOR, INIT_BLOCK + INIT_ANCHOR),
    (LOSS_ANCHOR, LOSS_BLOCK + LOSS_ANCHOR),
])

# ---- 4. policies/b1k_policy.py --------------------------------------------------------
F_ANCHOR = "    map_blind_prob: float = 0.3\n"
F_BLOCK = (
    "\n"
    "    # STAGE/PROGRESS head labels (patch_stage_head.py; consumed by models/pi0.py when\n"
    "    # the model config sets stage_head=True). Must match the model config's flags.\n"
    "    stage_head: bool = False\n"
    "    stage_classes: int = 4\n"
)
C_ANCHOR = "        return inputs\n"
C_BLOCK = (
    "        # STAGE/PROGRESS head labels (patch_stage_head.py). 'stage' is the per-frame\n"
    "        # parquet column in {0..stage_classes-1}. 'progress': episode length is NOT\n"
    "        # available in the row (the repack transform keeps only mapped keys and the\n"
    "        # dataset has no per-episode-length column), so frame_index/max_frame_index is\n"
    "        # not computable here; we use the coarse stage-midpoint fallback\n"
    "        # (stage + 0.5) / stage_classes. An explicit 'progress' key (future label\n"
    "        # column + repack entry, or the serve path) takes precedence automatically.\n"
    "        if self.stage_head and data.get(\"stage\") is not None:\n"
    "            _st = np.int32(np.asarray(data[\"stage\"]).reshape(-1)[0])\n"
    "            inputs[\"stage\"] = _st\n"
    "            if data.get(\"progress\") is not None:\n"
    "                inputs[\"progress\"] = np.float32(np.asarray(data[\"progress\"]).reshape(-1)[0])\n"
    "            else:\n"
    "                inputs[\"progress\"] = np.float32((float(_st) + 0.5) / float(self.stage_classes))\n"
    "\n"
)
patch(f"{FORK}/policies/b1k_policy.py", [
    (F_ANCHOR, F_ANCHOR + F_BLOCK),
    (C_ANCHOR, C_BLOCK + C_ANCHOR),
])

# ---- 5. training/config.py ------------------------------------------------------------
R_ANCHOR = (
    '            repack_mapping["stage"] = "stage"\n'
    '            repack_mapping["aux_pixels"] = "aux_pixels"\n'
)
R_BLOCK = (
    "        # STAGE/PROGRESS head (patch_stage_head.py): repack the parquet 'stage' column\n"
    "        # whenever the model's stage_head consumes it (idempotent with the map path\n"
    "        # above). NO 'progress' repack: RepackTransform KeyErrors on absent columns and\n"
    "        # the parquet has no 'progress' column; B1KInputs derives the coarse fallback.\n"
    '        stage_head = getattr(model_config, "stage_head", False)\n'
    "        if stage_head:\n"
    '            repack_mapping["stage"] = "stage"\n'
)
B_ANCHOR = "                map_tokens_k=map_tokens_k,\n"
B_BLOCK = (
    "                stage_head=stage_head,\n"
    '                stage_classes=getattr(model_config, "stage_classes", 4),\n'
)
patch(f"{FORK}/training/config.py", [
    (R_ANCHOR, R_ANCHOR + R_BLOCK),
    (B_ANCHOR, B_ANCHOR + B_BLOCK),
])

print("no changes needed: training/data_loader.py, training/weight_loaders.py")
print("REMINDER: a stage_head=True TrainConfig warm-started from a headless ckpt must set")
print("  CheckpointWeightLoader(..., missing_regex including '.*stage_head.*')")
print("ALL STAGE-HEAD PATCHES APPLIED")

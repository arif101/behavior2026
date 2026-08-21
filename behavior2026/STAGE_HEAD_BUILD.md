# STAGE/PROGRESS Head — Build Note (patch_stage_head.py, 2026-08-04)

Patch: `box_scripts/patch_stage_head.py` (compile-before-write, anchored, idempotent via the
`patch_stage_head` marker; `FORK_ROOT` env overrides the default `/root/openpi_fork/src/openpi`).
Line refs below are against the PRE-patch `fork_snapshot/` files.

## Attach point
Mean-pool of the ACTION EXPERT's final-layer suffix features over the action tokens:
`suffix_out[:, -self.action_horizon:, :].mean(axis=1)` — the exact features that feed
`action_out_proj` at `fork_snapshot/pi0.py:365` (`suffix_out` produced by the combined
prefix+suffix forward at pi0.py:362-364). Head = Linear(width→256) → gelu → Linear(256→C+1);
first C channels = stage logits, last channel → sigmoid = progress. Modules are created
immediately before `self.deterministic = True` (pi0.py:168-169), i.e. AFTER every stock and
optional module, preserving the rng stream of all existing params (same trick as pi0.py:135-138).

## Losses (added inside compute_loss, before `return loss` at pi0.py:415)
- `L_stage = stage_loss_weight (0.05) * CE(log_softmax(logits), clip(stage, 0, C-1))`
- `L_prog  = progress_loss_weight (0.02) * (sigmoid(p) - clip(progress, 0, 1))^2`
Both are per-sample scalars added as `(w * e).astype(loss.dtype)[:, None]` broadcast over the
horizon dim — byte-for-byte the convention of the existing aux losses (pi0.py:389/397/414).
Each term is skipped when its label is None (serve / unlabeled batches): loss-inert without labels.

## Passthrough triple (the new-Obs-field trap)
1. `models/model.py` — `progress` field after `aux_pixels` (model.py:121), `from_dict`
   (model.py:151), `preprocess_observation` reconstruction (model.py:236). NOTE `stage` already
   existed with its full triple from patch_aux_v2.py — only `progress` was added.
2. `training/config.py` — `repack_mapping["stage"] = "stage"` gated on `stage_head` (after
   config.py:476; idempotent with the map-path repack at config.py:475) + `stage_head` /
   `stage_classes` kwargs into `B1KInputs` (after config.py:491).
3. `policies/b1k_policy.py` — B1KInputs fields (after b1k_policy.py:72) + packing before
   `return inputs` (b1k_policy.py:149): `stage` from the parquet column; `progress` from an
   explicit `progress` key if present, else the COARSE FALLBACK `(stage + 0.5) / stage_classes`
   — episode length is not in the row (RepackTransform keeps only mapped keys and would
   KeyError on a nonexistent `progress` column, transforms.py:101, so none is repacked).

## Checkpoint compatibility
Flag-guarded module creation: `stage_head=False` creates zero params → `BaseModelConfig.load`'s
`check_pytree_equality` (model.py:268) sees the identical tree → run1b restores bit-identical.
`stage_head=True` warm-started from a headless ckpt needs `CheckpointWeightLoader` with
`missing_regex` including `.*stage_head.*` (default is `.*lora.*`, weight_loaders.py:49; map
precedent: patch_map_fork.py used `.*lora.*|.*map_.*`).

## Open questions
- No inference output: `sample_actions` returns a bare Actions array (pi0.py:480) — no output-dict
  convention, so stage logits are NOT returned at serve (would need a helper method / policy change).
- Training-time features see NOISY action tokens x_t (flow training), so the pooled feature varies
  with the flow timestep; a prefix-pooled variant would be t-invariant. Chosen per spec anyway.
- No TrainConfig entry added — activation = clone a config with `stage_head=True` + missing_regex.
- Overlaps with the aux_v2 prefix-feature stage CE (map-gated, pi0.py:391-397); both can coexist.
- Head trains at full LR; STAGE_HEAD_SPEC_v2 suggests head-only weight decay ~1e-3 (not wired).
- Fallback progress is piecewise-constant per stage; a real label column supersedes it automatically.

# RUN-2 Fork Patches — Build Note (2026-08-06)

The three remaining Run-2 patches vs `fork_snapshot/`, following the `patch_stage_head.py`
conventions exactly: anchored assertion-guarded edits, compile-before-write, idempotent via
marker, flag-guarded module creation AFTER all stock modules (rng-stream preservation),
default-off => run1b restore bit-identical. All four patches (incl. `patch_stage_head.py`)
verified together on scratch fork trees in BOTH application orders: apply-OK, idempotency
(SKIP on re-run), and structural assertions (see scratchpad `verify_patches.sh`; run
2026-08-06: ALL PASSED). JAX-level init/restore checks happen at Run-2 bring-up (G0
preflight with restored weights), same as the stage head.

## 1. patch_map_adaln.py — map GEOMETRY → AdaLN
- **Why**: G1 measured the prefix-token route consumed-but-unpaid at 90% visibility
  (FROZEN), while B0 geometry-only was best-approach — and AdaLN displacement is the route
  that produced every conversion. Run-2 pools map geometry into the SAME adaRMS cond vector.
- **Input**: map-token rows [T1,T3,T4,T5,T6,T7] (432 = 6x72), T5/T7[28:32] zeroed (the
  exact `blind_view()` target-derived fields); T0/T2 excluded → **blind-invariant by
  construction** (map_blind_prob can't leak target info through this route).
- tanh(x/4) bounding (map_recon convention), MLP 432→width/2→width, zero-init out.
- Touches only `pi0_config.py` + `pi0.py` — map_tokens already flow train- and serve-side.
- Requires `pi05=True` + `map_tokens_k == 8` (frozen token contract), enforced in post_init.
- Warm start: missing_regex must include `.*map_geo.*`.

## 2. patch_depth_aux.py — GT-depth aux head (+ box_scripts/add_depth_aux_labels.py)
- **Why**: depth is sensor-legal but the policy input is RGB-only; predicting per-patch
  depth from prefix image tokens forces 3D reading. Aux-leak law satisfied: dense depth is
  unobtainable from any other input stream.
- **Label contract**: parquet column `gt_depth_ds` float32 (768,) = 3 cams × 16×16
  patch-mean METERS (0=invalid), camera order = prefix image-token order. Precomputed
  OFFLINE by `add_depth_aux_labels.py` (lerobot `dequantize_depth`, masked patch means) —
  no train-time video decode. Run it over EVERY dataset in the mix BEFORE enabling the
  flag (RepackTransform KeyErrors on absent columns).
- Head: shared Linear(width→64)→gelu→Linear(64→1) per image token; masked L1 on
  log1p(depth); weights = validity ∧ per-camera image_mask (composes with modality
  dropout, which runs first). Loss-inert without labels (serve-safe).
- Known approximation: train-time image augmentation (0.95 crop, ±5° rotate, head cam) is
  not applied to the target — same accepted misalignment as the aux_pixels heatmap CE.
- Warm start: missing_regex must include `.*depth_aux.*`.

## 3. patch_modality_dropout.py — per-stream dropout p≈0.2
- Train-only; independent per-sample drops: target_points→null mask, map_tokens→zeros
  (null-token convention; aux `not_blind` gate already handles zeroed rows), each wrist
  cam→image_mask False. Head cam never dropped. No params.
- RNG via `fold_in(rng, 8206)` → all stock streams bit-identical (point-noise precedent).
- Ordering (asserted by the harness): dropout → point-noise → ... → depth-aux loss.
- **Enable exactly ONE of `anti_shortcut` / `modality_dropout_p`** — both would compound.

## Run-2 config checklist (item 4 of the plan; NOT wired by these patches)
- Clone a TrainConfig with: `stage_head=True`, `map_geo_conditioning=True`,
  `depth_aux=True`, `modality_dropout_p=0.2`, `anti_shortcut=False`, `map_tokens_k=8`,
  `point_conditioning=True`.
- `CheckpointWeightLoader(missing_regex=".*lora.*|.*stage_head.*|.*map_geo.*|.*depth_aux.*")`
  (map prefix params restore from run1b; only the NEW modules are missing).
- Prefix-route freeze (G1: K=8 frozen): freeze_filter over
  `.*map_(proj|registers|alpha|recon).*` — config-time decision, decide whether the aux
  heads stay trainable.
- aux_v2 prefix stage CE vs the new suffix stage head: DECIDE ONE (STAGE_HEAD_BUILD.md).
- Datasets: run `add_depth_aux_labels.py` over b1k_radio_map + b1k_radio_corrective +
  b1k_radio_rac + gold A7 before enabling depth_aux.

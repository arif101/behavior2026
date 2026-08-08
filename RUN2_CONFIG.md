# RUN-2 Config + Assembly (item 4) — 2026-08-08

TrainConfig `pi05_radio_run2` lands via `box_scripts/patch_run2_config.py` (verified on a
scratch fork tree stacked on all four model patches; idempotent). This doc is the
bring-up order + the decisions and their reasons.

## Trainer bring-up order (A100 pair, fork_snapshot restore path)
1. Restore fork + run1b params (`/root/warmstart_run2/params` = run1b final).
2. Apply patches in any order (verified): patch_stage_head, patch_map_adaln,
   patch_depth_aux, patch_modality_dropout, then patch_run2_config,
   patch_stage_oversample.
3. Assemble data: fetch/restore the three sources; `label_map_depth_from_source.py`
   output for b1k_radio_map is IN the data parquets (travels with the backup — verify
   `gt_depth_ds` present on all three); `assemble_run2_mix.py --overwrite` (asserts
   schema equality + gt_depth_ds everywhere; writes meta/run2_sources.json with
   measured advisory weights).
4. Recompute norm stats over the mix (assets repo_id `b1k_radio`).
5. G0 preflight WITH restored weights (the standing rule), incl.: run1b restore is
   bit-identical under flags-off; flags-on tree restores with only the new modules
   missing; one forward/backward on a real batch; loss finite with and without labels.
6. Launch with `B1K_STAGE_OVERSAMPLE=8`.
7. PRE-REGISTER the eval protocol BEFORE looking at any results (G1 lesson).

## Decisions (single-variable-change discipline vs run1b)
- **Flags**: stage_head + map_geo_conditioning + depth_aux + modality_dropout_p=0.2;
  anti_shortcut=False (dropout supersedes it — never both). K=8 unchanged.
- **Warm start**: missing_regex `.*lora.*|.*stage_head.*|.*map_geo.*|.*depth_aux.*` —
  map/aux params EXIST in run1b and restore (unlike run1's fresh-init regex).
- **Freeze**: `.*map_(proj|registers|alpha|recon).*` — the prefix map-token route only
  (G1: consumed-but-unpaid at 90% visibility). Aux decode heads stay trainable exactly
  as run1b. Considered and NOT taken: also neutralizing e_map/e_recon losses (they
  still shape the trunk through frozen heads) — that's a second variable; revisit only
  if Run-2 shows trunk regression.
- **Both stage losses coexist** (aux_v2 prefix CE + new suffix head). The SUFFIX head is
  the pre-registered consumer (progress → chunk-critic → RL reward). Prefix CE kept for
  run1b continuity.
- **Sampling**: stage-transition oversample x8 (run-1 machinery, env-toggled) + source
  weights from `run2_sources.json` (corrective weight computed from the measured
  transition-neighborhood mass; rac advisory 2.0 — far-field approach-recovery,
  measured 2026-08-07: p50 closest 1.06 m, zero episodes <0.35 m).
- **Mix**: map (200 eps / 429,928 fr) + corrective (68 / 20,925) + rac (105 / 16,245)
  = 373 eps / ~467k frames. Gold A7 RETIRED as training data (campaign saved no obs —
  same measurement as the RaC v1 verdict). Held-out split: keep map episodes 180-199
  held out (run-1 convention) — corrective/rac fully in-train.
- save_interval 3_650 ≈ 1 epoch of the mix at 32×4.

## Open items at launch time
- `/root/warmstart_run2/params` staging (run1b ckpt lives on the box at /root/ckpt →
  push/pull via HF backup).
- Eval protocol pre-registration doc (write BEFORE first eval; primary = conversion
  rate on the arm-A serving config; n fixed in advance).

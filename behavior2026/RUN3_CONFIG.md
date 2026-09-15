# RUN-3 Config + bring-up (trainer: 1x A100-SXM4-80GB, 300 GB disk, 251 GB cgroup) — 2026-09-06

Scripts: `behavior2026/box_scripts/run3/` (patch_run3_configs, dl_run3, prep_run3_data,
preflight_run3, run3_driver, run3_hf) + top-level `add_sample_weights.py`,
`assemble_run3_mix.py`. Recipe: RUN3_PACKING.md. Bars: RUN3_EVAL_PREREG.md.

## Fork restore (no HF tarball needed)
`openpi_fork/` source tree (base + point-AdaLN, = the HF tarball) → overlay
`fork_snapshot/` 7 files (run-1 layer, rescue branch) → overlay origin/main
`behavior2026/fork_snapshot/data_loader.py` (Run-3 sample_weight layer) → the five Run-2
patches in order (stage_head, map_adaln, depth_aux, modality_dropout, run2_config) →
`patch_run3_configs.py`. `uv sync` (uv 0.12, py3.11); ffmpeg via apt. Verified: jax 0.5.3
sees the GPU; `pi05_radio_run2` and all six `pi05_radio_run3_a*` configs construct.

## Arm configs (`_run3_cfg` in config.py) — what is identical across arms
model = Run-2 flags (pi05, point_conditioning, map_tokens_k=8, modality_dropout 0.2,
map_geo AdaLN, depth_aux, stage_head); warm start `/root/warmstart_run3/params` = Run-2
@49999 with `missing_regex=".*lora.*"` (every module exists → nothing fresh; preflight
check A asserts tree identity); freeze = prefix map-token route (as Run-2); AdamW defaults;
`CosineDecaySchedule(warmup 500, peak 2.5e-5, decay_steps = num_train_steps, floor
2.5e-6)`; batch 32; seed 42; EMA 0.99; save_interval 2,500; keep_period 5,000; norm
stats = Run-2's `assets/b1k_radio/norm_stats.json` copied into every arm's assets dir
(NOT recomputed per mix — identical input normalisation, matches the init checkpoint).
Steps: `_RUN3_STEPS = 15,000` (fixed 2026-09-06 before any launch, identical for all arms):
15k × 32 = 480k samples ≈ 1.0 epoch of the A4 mix; at the measured 3.83 s/step (pure
compute, batch 32, `memprobe_direct.py`) ≈ 16 h per arm, under the one-GPU-day cap.

## Single-GPU memory (measured 2026-09-06)
Batch 32 fits on ONE A100-80GB only under the Run-2 launch flag
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.92` (preallocated): JIT 155 s, then 3.83 s/step. With
`XLA_PYTHON_CLIENT_PREALLOCATE=false` both batch 32 and batch 16 OOM at ~61 GB used + a
17-20 GB allocation (allocator fragmentation, not batch scaling). Fixed state per GPU is
~67 GB (params + grads + Adam μ/ν + EMA at 3.36 B fp32 params). Launch flags are therefore
not optional. `PYTHONUNBUFFERED=1` is set so the loader's weighting lines reach the log
immediately (the driver's launch gate reads them).

## What differs (the single variable)
`dataset_root` per arm (`prep_run3_data.sh` builds `/root/b1k_radio_mix_a{1..5}` with
`assemble_run3_mix.py`, hardlinked videos) and the launch env: `B1K_SAMPLE_WEIGHT_COL`
unset for A0, `=sample_weight` for A1..A5. `B1K_STAGE_OVERSAMPLE=8` for all.

## Source weights (sample_weight column, decided at launch as RUN3_PACKING.md allows)
map 1.0 with the 10,000 poison-window rows → 0.1; factory 4.78; episodes 2.0; approach
1.0 (the assembler's measured advisory: factory mass ≈ the map's stage-1→2 transition-
neighbourhood mass). The loader multiplies the column into the frame sampler
(WeightedRandomSampler; 464k frames < the 2^24 torch.multinomial cap). The preflight
reads the sampler weights back and records effective mass per source per arm.

## Data gates (prep_run3_data.sh)
restore_map_videos (organizer repo, ~5 GB) → register gt_depth_ds → sample weights (always
`--overwrite-col`, deterministic) → dry-run must print 296 eps / 464,242 frames → per-arm
assembly → deregister depth streams (converter pts jitter; training never decodes depth) →
register sample_weight/gt_depth_ds in each mix → warm start + norm stats. Then
`preflight_run3.py a0 a1 a2 a3 a4` (CPU): tree identity, sampler class + mass, one real
batch per arm. Then a 20-step GPU smoke with a checkpoint write. Then the driver.

## Driver (run3_driver.sh)
Sequential arms A0, A2, A1, A3, A4, (A5 when its mix exists). Per arm: launch via
`scripts/b1k/train_b1k.py` (the only entrypoint that routes `create_b1k_data_loader`),
require the loader's `[stage-oversample]` (+ `[sample-weight]` for A1+) lines within 30 min
or declare a failed launch; every 5 min prune train_state from all but the newest committed
ckpt and push the newest params off-box to `arif101/b26-run3-params/<arm>/ckpt_<step>/`;
≤2 auto-resumes; finalize = upload `<arm>/params`, `<arm>/assets`, `<arm>/provenance/`
(logs, preflight JSON), verify ≥10 GB, slim local to params-only. Any failure stops the
driver (`RUN3_DRIVER_FAILED`).

## Bring-up log 2026-09-06 (token landed 03:00 UTC; all gates green)
- Downloads: map backup 6.3 G, manufactured 34 G (incl. raw segments), Run-2 params+assets 12 G — 4 min.
- Prep: dry run **296 eps / 464,242 frames** (exact); map 10,000 rows (2.33 %) → 0.1; factory 4.78;
  episodes 2.0. Mixes: a1 200/429,928 · a2 238/439,448 · a3 258/454,722 · a4 296/464,242.
- Holdout provenance: raw factory segments = demos {10,20,40,...,420} (38 demos), **no d200**;
  instance 301 is an eval instance and never enters any mix. Manufactured `raw_episode_id = -1`.
- Preflight A: run3 model tree == Run-2 checkpoint tree (92 leaves, nothing fresh).
- Preflight B (real sampler weights read back; stage oversample ×8 on strict 1→2 transitions):

| arm | frames | poison rows | source → effective sampling mass (mean w) |
|---|---|---|---|
| a0 | 429,928 | none (0 %) | map 100 % (1.21) |
| a1 | 429,928 | 2.33 % at 0.1 | map 100 % (1.13) |
| a2 | 439,448 | 2.28 % | map 81.7 % (1.13) · factory **18.3 %** (11.4) |
| a3 | 454,722 | 2.15 % | map 84.2 % · episodes 15.8 % (3.67) |
| a4 | 464,242 | 2.15 % | map 70.9 % · factory 15.9 % · episodes 13.3 % |
  (factory mean weight 11.4 = 4.78 × the ×8 oversample landing on the clips' own grasp transitions.)

## Checkpoint-save stall on the single GPU (found + fixed 2026-09-06, before launch)
The first 20-step smoke finished its steps (3.5 s/it on real data, loss 0.05-0.065 from the
warm start) and then hung in orbax's train_state device→host transfer: params item written in
24 s, train_state still 0 bytes after 19 min, one thread at 100 % CPU, RSS growing ~10 MB/s
(≈70 min per 40 GB save). Isolated `save_test.py` on the same state: default path 93 s, pinned
33 s — so the stall needs the training-process context (8 loader workers + page cache; the
cgroup peaked at 231 GB of its 251 GB limit during a save). Raw `np.asarray` on this box:
0.63 GB/s pageable (kernel-time bound) vs 2.5 GB/s via the GPU's `pinned_host` memory kind.
**Fix:** `patch_pinned_ckpt.py` → `save_state` uses `PyTreeSave(enable_pinned_host_transfer=True)`
for both items (same on-disk format; restore untouched). Second smoke: 20 steps + committed
checkpoint, save ≈ 35 s, SMOKE_OK. `ckpt_sampler.sh` logs cgroup usage / busiest thread /
tmp-checkpoint size every 10 s for the whole run.

## LAUNCHED 2026-09-06 03:46 UTC
`run3_driver.sh` arms a0 → a2 → a1 → a3 → a4 → (a5 when its mix exists), 15k steps each,
`--keep-period 5000`, off-box push of the newest committed checkpoint every 5 min to
`arif101/b26-run3-params/<arm>/ckpt_<step>/`, per-arm finalize to `<arm>/params|assets|provenance`.

## A5 wired 2026-09-06 20:10 UTC (while A2 trains)
`b1k_radio_approach` pulled (12 eps / 6,942 frames, HF 05:37 UTC); **source weight 2.0** (same as the
other complete-episode manufactured source; 1.0 would make A5-vs-A4 a 1.5 %-mass perturbation that
cannot be read — operator may override with `W_APPROACH` and re-run the a5 steps before A5 starts
≈ 09-09). Mix a5 = 308 eps / 471,184 frames, depth de-registered, columns registered. Preflight a5
PASS: tree identity; effective mass map 67.6 % · factory 15.1 % · episodes 12.7 % · approach 4.6 %
(mean w 4.70); poison 2.12 % at 0.1; one real batch drawn. The driver picks a5 up after a4.

## Progress
- A0 DONE 2026-09-06 19:02 UTC — 14,999 steps, 3.6 s/it, loss 0.029→0.017; `arif101/b26-run3-params/a0/`.
- A2 launched 19:02 UTC (loader lines verified); ETA ≈ 10:10 UTC 09-07. Then a1 → a3 → a4 → a5.

## A4 DONE · A5 VOID (2026-09-09)
- A4 finalized 08:21 UTC (loss →0.017; `arif101/b26-run3-params/a4/`). Four arms + control on HF: a0 a1 a2 a3 a4.
- **A5 diverged from step 0 (loss 6.4e7, grad norm 2e4; 4e8 by step 1600) and was stopped at 10:24 UTC;
  no checkpoint was written; the run dir was removed.** Root cause (measured, `normcheck.py` /
  `torsocheck.py` in box_scripts/run3): Run-2's action norm stats have **std = 6.38e-10 on action dim 6
  (torso joint 4)** — every teleop demo holds it at exactly 0, factory/episode clips too — while the
  approach clips command it over **−0.30…1.76 rad on 54 % of frames, all 12 episodes** (action == state
  d56, corr 1.000: the 11-DOF servo's trunk flex lives on that joint). openpi normalizes by
  (std + 1e-6) → targets ≈ 1.8e6 → loss ≈ 1e7–1e8. Every other column of the approach parquet is in
  range (schema-identical, no NaN/inf).
- Consequence beyond normalization: the served policy un-normalizes dim 6 with the same std, so it can
  NEVER output that joint — the approach clips teach a trunk-flex the policy's action space cannot
  execute. Fix options: (1) re-manufacture the approach clips with torso joint 4 locked at 0 (the
  demo convention; arm 7 + torso 1–3 = 10 DOF) — the principled fix; (2) stopgap: zero action dim 6
  (and state d56) in the approach parquet and rerun a5 prep + preflight — trains, but the observed
  trunk poses are then unexplained by the labels. Not chosen unilaterally; trainer idle (GPU 0 %).
- A5 preflight (which draws one batch and checks depth/stage/points) did not catch this: add a
  per-source **max |normalized action| / |normalized state| gate** to preflight_run3.py before any
  future source is admitted (threshold e.g. 50).

## S1 = arm a5 on the FIXED approach data — LAUNCHED 2026-09-15 23:39 UTC
Box: fresh 1x A100-80GB (154.54.102.23:19349; the Run-3 box is gone). Bring-up 3 min (trainer_bringup_s1.sh), prep 2 min,
preflight PASS, smoke 20 steps (loss 0.105 -> 0.053) + checkpoint write OK, driver a5 15k steps (~16 h, ETA ~15:40 UTC 09-16).
Mix_a5 = map (200 eps / 429,928 fr, poison 0.1) + factory (38 / 9,520, 4.78) + episodes (58 / 24,794, 2.0) +
b1k_radio_approach_v2 (19 / 8,919; stage-1 approach frames 2.0, transport tail 0.5) = 315 eps / 473,161 frames.
Warm start = Run-2 final params (same init as A0-A4) -> single variable vs A4 = "the honest approach source".
Preflight norm-gate (refined 09-15: z>50 AND raw>1e-2): map 9.3, factory 207.8 (raw 2e-4 rad on torso j4, learnable
constant A2-A4 converged on), episodes 143.0 (same), approach_v2 5.9. Sources 1-3 byte-identical to A4.
Readout: run3_eval_arm.sh on the RTX sim box, instance 301 n=25, grasp completion vs A4 (2/25) per RUN3_EVAL_PREREG.md.
NOT in S1 (by the single-variable rule): stage/progress conditioning, granular targets, temporal forcing — the 09-12
full-stack retrain; build order in PRESS_FIX_SPEC.md; S1's params seed it if the approach source helps.

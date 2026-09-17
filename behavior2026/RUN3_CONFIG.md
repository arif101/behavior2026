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

## S1 verdict + FULL-STACK arm LAUNCHED 2026-09-17 00:36 UTC
- S1 (a5 = A4 mix + b1k_radio_approach_v2, 15k steps, loss 0.018; params on HF a5/params): eval on the RTX box, parity
  serving (SERVE_FORWARD_MAP_TOKENS=0), assisted-grasp weld telemetry: 5/5 rollouts stall with the right hand >= 0.56 m from
  the radio, 0 grasps (remaining 20 rollouts run for the record). Harness validity on the same box: A4 rollouts reach 0.36 m
  and 0.07 m (campaign median 0.26) -> the harness is valid; the approach source HURT the approach. Suspect: every clip
  opens with a 25 cm base RETREAT at 15-20 cm from the radio, up-weighted as "approach" frames. To confirm from the S1
  films; fix = drop/down-weight the retreat frames, not the source.
- FULL-STACK arm `full` = pi05_radio_full (press fix: stage_v2/progress conditioning + target_points_v2 rail->button +
  temporal forcing K=8 @32, zero-init gate, feat RMSNorm, aux 0.05/0.05 @ 0.3 m units), warm start A4 (/root/ckpt_full_init
  -> run3_dl/a4/params), vision tower FROZEN, mix_full (315 eps / 473,161 fr), 10k steps, batch 32, keep_period 5000,
  FORK_SRC=/root/openpi_fork_v2/src. GPU smoke: loss 0.64-0.73, grad-norm 5.6-7.2. ETA ~11:30 UTC 09-17. Checkpoints push
  to HF (b26-run3-params/full/) — uploads restored after the user enabled auto-recharge.
- Eval plan: run3_eval_arm.sh full 25 with POLICY_CONFIG=pi05_radio_full WRAP=...StageV2AffordanceWrapper
  SERVE_FORWARD_MAP_TOKENS=1, then HISTORY_MODE=off|repeat|shuffle arms (TAG=_hist_*) on the same checkpoint;
  gate_liveness.py over the checkpoints. Primary metric unchanged: grasp completion vs A4 2/25; secondary: task success.
- 00:41 UTC health check: the RESOURCE_EXHAUSTED traceback at train_full.log:75-90 belongs to the PREMATURE 21:51 launch
  (pid 1720286, shared the GPU with smoke4; killed, no checkpoint). The live run (pid 1735318, log line 93 onward) is clean:
  step 74 at 3.1 s/it, GPU 100 %, revised ETA ~09:15 UTC 09-17. Watcher reads only the live portion (tail -n +93).
- Sim box chain `chain_evals_0917.sh` (box_scripts/run3/): waits for the A4 validity rollouts (5) -> resumes the S1 record
  eval a5 runs 6..25 (parity) -> polls HF for full/{params,assets,provenance} -> `run3_eval_arm.sh full 25` with the full
  serving stack -> SUMMARY (grasp count vs A4 2/25). A4 validity so far: run 1 minL 0.321, run 2 minL 0.046 (no weld).
- 00:50 UTC S1 failure-mode readout (action logs + wrapper stats + films, a5 runs 1-5 vs a4 runs 1-3): the base-RETREAT
  hypothesis is FALSIFIED — no S1 rollout ever commands backward base motion (vx < -0.05 on 0 % of steps) and S1's forward
  command integral is >= A4's (73-103 vs 76-84). The actual mode is STOP-SHORT-AND-FREEZE: the base halts with the radio
  ~0.8-1 m ahead, both arms drop to a rest posture (arm command norm ~1.65 vs 2.33 at t=0 and 2.8-3.0 when A4 reaches;
  wrist ~80-94 deg), and nothing changes from step ~800 to 3225 (hand 0.57-0.69 m). A4 run 3 sits in the SAME mode
  (min 0.63 m, norm 1.6); A4 runs 1-2 instead extend (0.32 m / 0.05 m, norm 2.8-3.0, wrist 30-51 deg). So the approach
  source did not add a new behaviour; it made an existing A4 failure mode the only outcome (5/5 vs 1/3 here, campaign
  median minL 0.26). The rest posture is NOT in the approach clips (clip arm norm p10/50/90 = 2.81/3.24/3.66, 0 % of
  10,148 frames in [1.5, 1.85]; base moving on only 2-28 % of clip frames). Working hypothesis for the fix: the clips'
  predominantly stationary base + close-range radio view taught "radio in view -> base 0"; they were up-weighted as
  stage-0 approach frames although they contain no far-field approach. Any reuse must restrict them to the <= 25 cm
  arm-reach segment (stage 1 labels), not stage 0. Filmstrips: scratch films/{a5_run_1,a4_run_2,a4_run_3}_big.jpg.
- 01:21 UTC A4 harness-validity eval complete (n=5, same box/serving as S1): 0 grasps, right-hand minima 0.36 / 0.07 / 0.71
  / 0.31 / 0.40 m (4/5 within 0.4 m; one stop-short-and-freeze). 0/5 is consistent with the campaign's 2/25 (P = 0.66), and
  the 0.07 m reach shows the harness produces full reaches -> S1's 5/5 stalls at >= 0.57 m are the policy, not the box.
  Chain moved on: S1 record eval runs 6..25 started 01:21 (serving /root/ckpt_a5, parity).
- 02:35 UTC S1 record eval, run 7 = TASK SUCCESS on instance 301 (q=1.0, 2,037 steps; right-hand weld on radio_89 at step
  1685, left-hand min 0.16 m, film: base drives up, right arm grasps the rail, lifts and rotates the radio, presses).
  Runs 6/8/9 are stop-short-and-freeze again (0.57 / 0.69 / 0.50 m). S1 tally 1/9 grasps, 1/9 successes; the verdict
  "approach source hurt" now rests on 8/9 freezes vs A4's 1/5 on this box — hold it as provisional until 25. Trainer:
  step 2,270 / 10k, loss 0.053, grad-norm 1.02, clean.

## Freeze diagnostic (2026-09-17, sim box chain `freeze_diag.sh`; DIAGNOSTIC ONLY, no eval arm, no training data)
- Question: is the wrist-orient transition the ONLY blocker (so recovery clips would suffice), and is the freeze pinned by
  the injected target points? Training-data check (mix_full): base parked at 0.6-0.8 m is normal in the demos (base moving
  on 8 % of stage-0 frames there, arm extending, norm 2.75-2.96); the freeze posture (parked + arm norm 1.55-1.75) is
  0.5 % of the data at 0.4-1.0 m and 1.1 % beyond 1 m (episode-start idle). So the policy parks like a human and drops
  into an idle pose the demos almost never leave from.
- Pieces: eval/freeze_harvest.py (dumps og.sim state once motionless 150 steps after step 450; ends the episode),
  eval/freeze_continue.py (reset = restore a harvested state, optional ORIENT: 7-DOF DLS servo of the right hand to the
  canonical rail-grasp attitude + 0.25 m advance = S1 run 7's own escape; control = restore only), AFF_TAU env override
  in affordance_map_fullres.py (AFF_TAU=2 -> points never injected), serve_arm.sh (EXTRA_SERVE_ARGS for --action-horizon).
- Chain: S1 record eval finishes -> no-server smoke (--policy local) -> harvest on 301 + train ids 0-5 (301 seeds are
  never training seeds) -> continue from each state, orient=1 and orient=0 -> points-off x2 -> commit-to-chunk
  (--action-horizon 32) x3. YIELDS to the full-stack n=25 eval the moment its params land on HF (at most one rollout late),
  then resumes; ends with HISTORY_MODE=off x10 on the full checkpoint. Outputs: /root/freeze_diag/{harvest,continue,
  pointsoff,commit32}.log, states /root/freeze_states/*.npz.
- Readouts: (a) orient=1 grasp rate vs orient=0 -> sufficiency of the orient transition; (b) freeze on train layouts?;
  (c) freeze persists with points off? (run 7 had the lowest injection fraction, 75 % vs 80-93 %); (d) does committing
  to 32-step chunks let S1 escape on its own (runs 5/6 sampled a rotation and were pulled back at the next replan)?
- 07:18 UTC S1 RECORD COMPLETE (a5, n=25, parity serving, instance 301): grasp 2/25, success 1/25 (run 7 q=1.0; run 23
  weld at step 3113, out of time). Equal to A4's 2/25 grasp reference. Failure modes: stop-short-and-freeze in ~19/25
  (hand parked 0.44-0.71 m), reach-without-grasp in ~4 (0.29-0.37 m). Verdict update: the approach source did not lower
  grasp completion (2 vs 2); it did shift the failure mode toward the freeze. Full-vs-S1 (equal data) is the clean read
  of the full stack; full-vs-A4 remains the pre-registered bar.
- Freeze diagnostic chain: attempt 1 failed at instantiation (my `_dump(obs)` shadowed OraclePointWrapper._dump()), attempt 2
  harvested a state (zero-action smoke, step 150) but the continue wrapper crashed in og.sim.load_state (numpy vs torch);
  fixed (th.as_tensor; servo skipped on the evaluator's pre-instance reset) and relaunched 09:12 UTC as attempt 3.
- 09:19 UTC FULL-STACK ARM TRAINED: 10k steps, final loss 0.032 (from 0.68 at launch), grad-norm 0.54; params on HF
  (b26-run3-params/full, 29 files, 11.6 GB) + assets + provenance; intermediates 5000/7500 pushed. Trainer idle.
- 09:28 UTC full-stack n=25 eval STARTED on the sim box (pi05_radio_full, StageV2AffordanceWrapper, map tokens forwarded,
  HISTORY_MODE=normal); the diagnostic chain yielded to it and resumes afterwards (~17:00 UTC).
- Diagnostic attempt 3: smoke OK (harvest at step 150; continue restored the state and the arm-only DLS servo brought the
  wrist from 1.23 to 0.006 rad of the canonical grasp attitude, +16 cm of the 25 cm advance). Harvest on 301 (S1, cap
  1600): NO stationary trigger, but the film-strip numbers show a run-7-style escape IN PROGRESS: parked at ~900, idle
  with the arm dithering (cmd-norm 2.05, std 0.15) through 1200, wrist 85 -> 22 deg and arm extending 2.73 at 1300-1400,
  hand 0.58 m at the cap. The 0.02 m / 0.03 rad bars never fire on a dithering idle -> loosened to 0.04 m / 0.10 rad /
  0.02 twist, FREEZE_METRICS printed every 100 steps for calibration, and a FREEZE_CAP=1500 fallback dump (meta
  stationary=false) so every non-grasping harvest yields a seed. 301 must be re-harvested after the chain (not in loop).

## FULL-STACK ARM VERDICT (2026-09-17 17:03 UTC, n=25, instance 301, full serving stack)
- grasp 1/25 (run 3, right-hand weld at step 1727, out of time), success 0/25. Pre-registered bar A4 2/25: NOT beaten.
  Equal-data comparison S1 (A4 mix + approach_v2, old stack) 2/25 grasps, 1/25 success: NOT beaten either.
- Failure-mode shift is real even though the bar is not: hand within 0.25 m in 12/25 (S1 5/25, A4 validity 1/5), freeze
  (>= 0.4 m) 11/25 (S1 19/25). Median closest hand 0.29 m (S1 0.57, A4 0.31); median wrist angle 65 deg (S1 84, A4 46).
  So stage/progress conditioning + temporal forcing roughly halved the freeze rate at equal data, but reaches do not
  convert: 12 reaches -> 1 weld. Reach-without-grasp is now the dominant mode (as it was for A4).
- Reading: the stack moved the policy out of the idle attractor more often, which is consistent with the diagnosis; the
  grasp itself (last 10 cm: wrist attitude + closure) is the next bottleneck and was never the target of this arm.
- Next: (1) freeze diagnostic (resumed 17:04, S1 server) -> sufficiency of the orient transition; (2) HISTORY_MODE=off
  x10 on the full checkpoint (queued at the end of the chain) -> is the temporal channel doing the freeze reduction;
  (3) a reach-to-grasp readout on the 12 full-arm reaches (films: closure attempted? attitude off?) before choosing the
  next arm. Do not build the recovery-clip arm before (1) and (3).

## ROOT-CAUSE DIAGNOSIS (2026-09-17 17:30 UTC) — why no arm breaks 2/25
- Record: A0 0/12, A2 1/12, A4 2/12 -> 2/25, S1 2/25, full 1/25 grasps; success 0-1/25. Three data doses, v2 labels,
  temporal forcing and the serving fixes moved the FAILURE MODE (freeze vs reach-short) but never the grasp rate. That is
  the signature of a cap outside everything varied.
- Newly measured cap candidate: the POINTER. Every arm trains on exact sim-state target points (20 % modality dropout) and
  is served the affordance head's point. Over all 55 rollouts on 301 the head's error is 0.21 m median and 0.32 m at the
  moment of closest approach, at conf 0.95+ (null fallback never fires). Reaches stop 0.11 m short on average; the three
  grasps had the lowest pointer error at closure (0.13/0.14/0.23 m). corr(closest hand, pointer error) = 0.44. The head's
  "2.6 cm held-out" was on training-distribution frames. Second train/serve mismatch on the A-arms: map tokens never
  forwarded (fixed for `full` only).
- Freeze attractor is a policy property: S1 froze on train layout 0 too (step 929, base 0.79 m out, conf 0.56).
- Lineage pi05_base -> run1b -> Run-2 (50k) -> A-arms (15k) -> S1/full: ~90k fine-tune steps on ~200 demos, loss 0.02-0.03.
  Not poisoned by bad labels (press-bug windows are down-weighted) but TRAPPED: further fine-tuning from A4 on the same
  data has no gradient at the failure states, and the inherited over-trust in an exact pointer never gets corrected.
- Probes queued (after_diag.sh, after the freeze chain): P1 oracle pointer on the full ckpt (DIAG_ORACLE_POINT=1, ceiling),
  P2 pointer off (AFF_TAU=2), P3 oracle pointer on A4, P4 A4 with map tokens forwarded; n=10 each. Diagnostic arms only.
- Fix candidates (ranked): F1 train on PREDICTED points (run the head over the training frames; condition on its output +
  null when conf < tau) so the policy learns how far to trust the channel; F2 close-range pointer from the wrist cameras
  + depth surface snap (301's radio sits on a glass table; check the z-error sign); F3 a legal visual-servo grasp
  primitive for the last 20 cm (v13d DLS servo with a wrist-cam target instead of sim state) so the closure no longer
  depends on BC; F4 fresh start from pi05_base with F1 baked in and the approach clips replaced by recovery clips.

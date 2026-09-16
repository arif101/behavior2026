# Temporal forcing — build note (2026-09-16)

Implements research/TEMPORAL_4D_SWEEP_2026_09_06.md §4 in the tracked fork. Flag `temporal_conditioning` on
`Pi0Config`; config `pi05_radio_full` = press fix (stage_v2 / progress / target_points_v2) + this pathway.

## Mechanism (pi0.py)
1. **Content.** K=8 past HEAD-camera observations at stride 32 frames (one action chunk) + the current frame, each as a
   2048-D *gist* = mean of the 256 SigLIP tokens after the PaliGemma projector. Gists are PRECOMPUTED per frame by the
   FROZEN tower of the warm start (`precompute_gists.py` → parquet column `gist_head`), so training costs no extra
   vision compute; the loader stacks the K+1 slots through LeRobot delta-timestamps (`gist_head` [K+1, 2048] oldest…
   current, `gist_head_is_pad` marks slots before the episode start). The tower is FROZEN in the temporal arm
   (`freeze_filter` on `PaliGemma.img`) so the serve-time gist from the model's own tower equals the precomputed one.
   *Deviation from the A4 recipe (which trained the tower); logged.*
2. **Compression.** `temp_in` (2048→1024) + learned slot embeddings (K+1) → RMSNorm → 4 learnable temporal queries
   through 2 cross-attention + MLP blocks (8 heads) → mean over queries = the pre-gate feature `feat` [1024].
3. **Route.** `temp_out` (1024→1024, ZERO-INIT kernel) is added to the adaRMS conditioning vector of the action
   expert, next to the timestep / point / stage / progress / map-geometry terms. Never the VLM prefix.
   Zero-init ⇒ exact no-op at init (measured: loss and sampled actions bit-identical with the pathway on/off).
4. **Supervision on the pathway (pre-gate, training-only heads).** `temp_flow_*`: masked MSE on current-minus-past
   [EE_L, EE_R, button] displacements (0.1 m units) over the K offsets (`hist_flow` from the `hist_geo` column stack;
   padded slots masked); weight `temporal_flow_weight` = 1.0. `temp_stage_*`: stage CE + progress MSE readout,
   weight 0.2. The gradient lands on the queries BEFORE the gate, so the channel cannot stay dead (Temporal Forcing
   row-3 failure mode).
5. **Serve.** `Policy.compute_gist` (model tower, jitted) + a stride-aligned ring buffer in `B1KPolicyWrapper`
   (`process_input`): slots every `TEMPORAL_STRIDE` env steps, current gist at every inference, left-padded/masked
   when short. Liveness controls via `HISTORY_MODE=normal|off|repeat|shuffle` (mask-off delta, repeat-current and
   history-shuffle arms of the same checkpoint, TEMPORAL_4D_SWEEP §4.5). Gate readout: `gate_liveness.py`
   (‖temp_out.kernel‖ over checkpoints: flat = inert, open-hold = adopted, open-then-collapse = abandoned).
6. **Serving fix found on the way.** `B1KPolicyWrapper.process_input` forwarded only target_points / mask /
   stage_tokens; `map_tokens` (attached by patch_map_passthrough2) and the v2 keys were dropped before the policy —
   the K=8 map path saw NO tokens at serve in every Run-2/Run-3 eval (its gate alpha=0.036 kept the damage small).
   Now forwarded: map_tokens, target_points_v2, stage_v2, progress, gist_head.

## Smokes (CPU, trainer box, 2026-09-16 01:xx UTC)
- `smoke_temporal.py` PASS (01:40 UTC): 41 temporal param leaves; temp_out kernel = 0; loss on/off max|Δ| = 0.0;
  sample_actions max|Δ| = 0.0; loss with hist_flow + stage labels finite (+0.85 from the aux terms on random targets);
  grad norms temp_in 4.28, temp_flow_in 1.93, temp_out (gate) 8.67, action_out_proj 3.53. Caveat learned: pi0.5's adaRMS
  modulation Dense layers are ZERO-INIT, so at random init dL/dcond = 0 for EVERY conditioning term (progress_mlp_out
  too) — the gate gradient only exists with trained modulation kernels (A4: 2/3 non-zero); the smoke perturbs them.
- `preflight_full.py` on the mix with gist columns: pending the map precompute.
- GPU 20-step smoke + checkpoint write: after S1 finishes (the A100 is full).

## Data (precompute_gists.py, A4 tower, on the S1 box beside training)
approach_v2 8,919 fr / 110 s; factory 9,520 / 134 s; episodes 24,794 / 578 s; map 429,928 / ~2 h (decode-bound,
~80 fps; costs S1 ≈ 30 % step rate while it runs). Columns: gist_head f32[2048], hist_geo f32[9].

## Cost
+41 param leaves (~12 M params: 2×(attn 4·1024² + MLP 4·1024²) + temp_in 2 M + heads). Training-time overhead is the
loader's 9-slot stack (72 KB/sample) + a 9-token cross-attention: expected ≪ +20 % wall-clock.
- `preflight_full.py` PASS (04:15 UTC) on `/root/b1k_radio_mix_full` (315 eps / 473,161 frames, real batches): history_gists
  [9, 2048], valid slots 7.0/9 on average (current always valid; 65 % of samples have all 8 past slots — episode starts are
  padded), gist norm 258 ± 8.5, |g_t − g_t−32| = 11.2; change targets: EE_L 0.04 → 0.16 m, button 0.10 → 0.36 m over the
  8 offsets. Gists: parallel decode (6 workers, ~88 fps each) + tower pass from cache (map 430k frames ≈ 25 min on the
  shared GPU); frame caches under /root/frame_cache (≈ 71 GB) kept for the S1 re-tower.

## Demand-side readout (aliasing_rate.py, 04:58 UTC, 20k queries over 473,161 rows, 128-D JL per slot)
| NN by | median chunk distance (std.) | NN from a different stage_v2 | flow-loss floor proxy |
|---|---|---|---|
| random pair | 36.1 | – | 0.481 |
| current gist | 20.3 | 21.7 % | 0.193 |
| history stack (9 slots) | 19.9 | 14.8 % | 0.182 |
| next frame (floor) | 0.28 | – | – |
History improves action predictability by ~2 % (loss floor −5 %) — the sweep's "channel not demanded by BC" signature —
but cuts cross-stage confusion by a third. Stage is what the v2 labels + serve mirror already supply exactly, so the
press-fix conditioning captures most of that gain; expect a small gate opening and a small history-off delta. Caveat:
the pooled gist is a coarse proxy for the full prefix (images + proprio + points). Launch unchanged (pathway is a
zero-init no-op if unneeded); `ARMS=press` is the one-word alternative for a cleaner attribution.

## GPU smokes of pi05_radio_full (A4 warm start, mix_full, 40 steps; 2026-09-16 18:20–22:20 UTC) — loss/grad-norm tuning
| variant | loss @ steps 20–35 | grad norm | note |
|---|---|---|---|
| S1 (a5) warm start, reference | 0.05–0.10 | ~1 | |
| press-only (v2 labels, no temporal) | 0.58–0.69 | 2.1–2.4 | the jump is the v2 TARGET shift (rail vs button on pre-lift frames) into the trained point-conditioning; the run adapts |
| full, flow targets 0.1 m units, weights 1.0/0.2 | 5.0–5.4 | 38–42 | aux MSE dominated; with the 1.0 global-norm clip the policy gradient was scaled ~1/40 |
| full, 0.3 m units, 0.2/0.2 | 1.07–1.22 | 13–18 | |
| full, 0.05/0.05 | 0.67–0.75 | 10–13 | norm not from the aux weights |
| full, + RMSNorm on the pooled feature before the gate (FINAL) | 0.64–0.73 | 5.6–7.2 | the gate kernel's gradient = feat ⊗ dL/dcond; unnormalized feat was the norm source |
Gate after 20 steps (first smoke): ‖temp_out‖_F 3.4e-4 (opening). Under Adam a uniform clip scale barely changes the
update, so the residual 3× norm vs press-only is acceptable; heads settle within a few hundred steps.
Operational: S1's HF pushes are blocked (403 storage billing) → params copied box-to-box; the decision watcher wrote "a4"
on a failed (0-rollout) eval and the driver launched prematurely (killed within 2 min, no checkpoint) → watcher now
requires ≥ 20 rollouts; eval driver's readiness check uses /dev/tcp (no ss/netstat on the v6 box).

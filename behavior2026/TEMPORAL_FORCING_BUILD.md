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

# RUN-3 EVAL PRE-REGISTRATION — frozen 2026-09-06 (before any Run-3 checkpoint exists)

Registered at trainer bring-up, before the first arm launched and before any Run-3 eval
result has been observed. Per the G1 lesson, nothing below changes after results arrive;
deviations get logged here with timestamps, never silently.

## The thesis under test
The 200 teleop demos are honest EXCEPT the grasp window (a scripted rig slid the object
25-100 cm into the closing gripper; 2.33 % of frames, `poison_windows.json`). BC on the raw
demos therefore teaches "close at a distance and the object arrives"; at eval the policy
stalls short of the object. Run-3 asks ONE question: does adding manufactured, physically
honest grasp data (certified by the eval's own assisted-grasp rule) make the policy close
the last ~10-33 cm and grasp on held-out instances? **A2 vs A0 is the comparison.**

## Arms (single-variable; identical model / optimizer / schedule / seed / steps / norm stats)
| arm | data | env |
|---|---|---|
| A0 | map only, no down-weight (control / floor) | `B1K_STAGE_OVERSAMPLE=8` |
| A1 | map + poison down-weight (0.1 in the rigged windows) | + `B1K_SAMPLE_WEIGHT_COL=sample_weight` |
| A2 | A1 + b1k_radio_factory (38 honest grasp+transport segments, 9,520 fr, source weight 4.78) | same |
| A3 | A1 + b1k_radio_episodes (58 complete honest episodes, 24,794 fr, source weight 2.0) | same |
| A4 | A1 + factory + episodes | same |
| A5 | A4 + b1k_radio_approach (pre-contact approach clips; wired, data lands later) | same |
Init = Run-2 final params (arif101/b26-run2-params @49999); every arm is a continuation
fine-tune of the same checkpoint. Training order on the single A100: A0, A2, A1, A3, A4, A5
(the go/no-go pair first).

## Held-out evaluation set (NEVER in training)
- turning_on_radio instance 301 (public_test index 0) — the frozen eval instance.
- demo d200 restore states — the frozen grasp-probe demo.
Every training mix is built from map episodes 0-199 + manufactured clips; the manufactured
sources' provenance is checked at prep time to contain neither d200 nor instance 301.

## Primary metric — GRASP COMPLETION (THE metric; A0 is the floor)
Per rollout from the standard task start on instance 301: does the fingertip close the
remaining gap and the eval's assisted-grasp weld condition fire on the radio (contact ∧
between-finger ray ∧ sustained closing)? Read from the sim (AG weld event on the radio),
not from the video. **n = 25 rollouts per arm, same seeds across arms, default timeout.**
- Go/no-go rule (fixed): **A2 must beat A0.** With n=25 each, "beats" = A2 − A0 ≥ 3
  completions AND one-sided Fisher p < 0.10; anything less = NOT beaten → STOP and report,
  the pivot (2025 winner's demos + inference heuristics) is the user's call.
- STRONG: A2 ≥ 8/25 (32 %) grasp completion with A0 ≤ 2/25.

## Secondary metrics
1. Full-task success (toggled_on at episode end), n = the same 25 rollouts. Bar to beat:
   26 % (2025 winner). Reported for every arm; never substitutes for the primary.
2. Fingertip-approach-distance probe: per rollout, the minimum fingertip-to-radio-rail
   distance before the first close command; report median per arm (Run-2 reached 1-2 cm
   with no closure — the stall signature).
3. Copycat guard: action-chunk autocorrelation / state-conditioned vs unconditioned MSE on
   held-out map episodes; flag if an arm's actions are explainable by the previous chunk.
4. Liveness (mechanistic): AdaLN affordance-point channel and map-geo channel probes at
   the standard denoise steps (Run-2 measured DEAD — n=64 pinned frames, ±3 cm key
   displacement). Watch for revival; a still-dead channel is reported, not hidden.
5. Training-curve telemetry per arm: loss, depth-aux L1, stage-head accuracy, and the
   loader's printed sampling report (effective mass per source; preflight JSON).

## Guards
- Stage 1→2 oversample ×8 applies to EVERY arm (Run-2 setting), so A0's floor includes
  8× emphasis of its own poisoned windows — that is the honest control (Run-2's recipe).
- Negative control: held-out BC action MSE on map episodes 180-199 per arm; an arm that
  regresses >10 % vs Run-2 is flagged (grasp gains must not come from breaking the rest).
- A1 vs A0 isolates the down-weight; A3/A4 vs A2 isolate the learned in-hand press data.

## Hygiene
- First eval on each arm's FINAL checkpoint only; mid-checkpoint evals are exploratory.
- Eval needs an RT-core sim box (`/setup-sim-box`); the A100 trainer never renders.
- Checkpoints leave the box continuously (arif101/b26-run3-params/<arm>/...).

## Serving-spec clarification (logged 2026-09-08, before any Run-3 arm rollout; no bar changes)
Driver: `behavior2026/box_scripts/run3/run3_eval_arm.sh` (serve line from rate_legal.sh, eval loop
from campaign_run1.sh — the committed Run-2 pieces). Identical for every arm:
- Wrapper `behavior2026_eval.affordance_map_fullres.AffordanceMapFullRes` (+ patch_wrapper_arms):
  LEGAL affordance-head target points → AdaLN, online FoveatedMap → map tokens. NOT the oracle
  wrapper (sim-state reads; G3 diagnostic only) and NOT the default EnvironmentWrapper (delivers
  no points: the policy then gets the null sentinel and wanders — the "drives to the TV" failure).
- Evaluator must carry patch_point_passthrough + patch_map_passthrough2: without them the wrapper's
  target_points/map_tokens are dropped before the websocket payload (measured, docstring).
- `MAP_ARM=B` (live map tokens). Ambiguity on record: RUN2_EVAL_REPORT says "prefix map tokens
  zeroed" AND "live FoveatedMap geometry → AdaLN", but both routes read the same `obs.map_tokens`
  array (pi0.py), so the committed code can only deliver both-live (MAP_ARM=B) or both-zero
  (MAP_ARM=A); the only mechanism for prefix-null + geometry-live is `MAP_TOKENS_INVISIBLE=1` on
  the server. Choice for Run-3: MAP_ARM=B, flag unset — the training-time observation convention
  (tokens present with live content 70 % of the time, zeros under dropout); the frozen prefix route
  contributes through map_alpha = 0.036 (read from the params, identical in run2@49999 and a0)
  exactly as in training. Harness validity is established by reproducing Run-2's approach on the
  Run-2 checkpoint under this same config BEFORE any arm is scored.
- `--policy.config pi05_radio_run2` (architecturally identical; serve-time it only supplies model
  flags, transforms, asset_id) with `--repo-id b1k_radio` (REQUIRED: default is the task name and
  would resolve the wrong assets path); `--policy.dir` = `<arm>/` from HF (params/ +
  assets/b1k_radio/norm_stats.json, md5 a6053883b29666925566ec02c433e562 = Run-2's stats, identical
  across arms). Serving tree = /root/openpi_fork built by serve_run2.sh (NOT /root/openpi_adaln,
  the G3/Phase-A tree without stage_head/map_geo/depth_aux modules).
- No `--max-steps` (task default 3,225 steps, as Run-2). `run_eval_arm.sh`'s 500 was Phase-A only.
- Harness validation gate (Run-2 checkpoint, this config, ≥5 rollouts): affordance stats
  n_inject > 0 with conf_p50 ≈ 0.7–0.8; per-rollout min EE-to-target ≤ ~0.15 m (Run-2 rollouts
  worked the 5–10 cm shell); campaign minimum over 25 was L 0.021 / R 0.030 m. 0.5–0.8 m = no
  points delivered. Metric = EE-to-affordance-target (radio), the wrapper's dist_L/dist_R series.

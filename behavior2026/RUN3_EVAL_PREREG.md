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

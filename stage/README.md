# Stage Prediction Head (Phase 2)

Per-arm task-progress perception for BEHAVIOR-2026: which official skill each
arm is executing, phase within it, progress, active goal literal, and an
instantaneous per-literal P(satisfied) that the director latches into the
ledger. Spec: `../STAGE_HEAD_OWNERSHIP.md` (+ v1.1 addenda); wiring:
`../architecture.html`.

## Design decision (vs the VLM-orchestrator sketch)

Two candidate architectures existed: (a) a VLM (Opus-class) orchestrator that
*tells* a stage head which subtask is active, with a separate learned recovery
head; (b) this repo's design — a programmed director, with the stage head as
*perception* reporting per-arm stage distributions upward. We build (b); it
benchmarks higher on the 2026 metric because:

1. **Latency/serve budget.** Episodes are 30-75 min at ~13.5 FPS with a 24GB
   local serving cap. An LLM in the inner loop cannot run every step; the
   director must. The LLM survives as an optional boundary-time consultant
   (~10-30 calls/episode, strict timeout, programmed fallback).
2. **Robustness to drift.** Command-direction stage ("you are in subtask x")
   breaks silently when execution diverges from plan; perception-direction
   stage re-grounds every step and feeds stage-keyed recovery (the 2025
   winner's single biggest lever, ~2x on long-horizon, was a stage head +
   stage-keyed voting — Larchenko, arXiv 2512.06951).
3. **Label availability.** Perception-stage trains free from the organizers'
   official annotations + our privileged-state extractor; a command-direction
   head has no ground truth at all off-plan.
4. **Recovery = trigger + branch, not a head.** Metacog (conformal P(fail))
   triggers; the director's exceptional branch acts. Both consume this head's
   outputs (entropy, stage_age_ratio, ledger).

## Model (`model.py`, 12.6M params, <20M budget)

No backbone owned — consumes the shared frozen DINO 37x37 patch-token pass
(one pass per step, read by grounding + stage). Two asymmetric branches:

- **Current frame, full detail:** patch tokens + depth-lifted xyz (grounding's
  ray convention) -> 2-layer encoder -> 4-query attention pool.
- **History, cheap (WINDOW 2-5s):** per past frame only the mean patch token
  (free at serve — computed at its own step) + 61-d proprio -> 2-layer GRU.
  Training caches 768 floats/frame instead of re-running the backbone.

Goal literals (predicate + hashed target/reference category) become tokens;
two arm queries cross-attend over [spatial | temporal | literal] tokens.
Outputs per arm: official-taxonomy stage distribution (task-masked logits),
6-phase, sincos progress, active-literal attention; shared: per-literal ledger
logits. Exports: `z_stage = Σ pᵢEᵢ` soft mixture (policy AdaLN; v1.1 soft
conditioning) and distribution entropy (metacog feature). `stage_age_ratio`
is a serve-side stopwatch vs label-median durations, not a model output.

## Labels — two complementary sources

| | official `skill_annotation` (HF demos) | Phase-1 extractor |
|---|---|---|
| stage over official taxonomy | yes (frame ranges, free) | no |
| per-arm attribution | no (single track) | yes (validated bimanual) |
| phase / progress / ledger | partial | yes (privileged state) |

`official_taxonomy.json`: 34 skills + per-task masks, extracted from
`behavior-1k/2026-challenge-demos` `annotations/` (3 eps x 100 tasks;
rebuild: `python taxonomy.py --annot_dir <full dump>`). Merge rule
(`official_annotations.merge_arm_labels`): navigation -> both arms;
manipulation -> the arm whose extractor track owns the manipulating object;
official-only mode (no extractor labels yet) supervises both arms and masks
phase/ledger. Boundary frames within ±0.3 s blend soft targets (v1.1).
Ledger v1 proxy: literal flips at completed place/insert/close segments —
replaced by Phase-1's real BDDL predicate eval when it lands (`ledger_valid`).

## Pipeline

```
prep_data.py    # pod: download task subset of HF demos (LeRobot v3) -> manifest
build_cache.py  # decode zed rgb/depth per episode span, DINO glob feats,
                # proprio, labels -> extends the v0.5 grounding cache layout
train.py        # bf16, uniform-per-task sampler, masked multi-loss,
                # writes stage_medians.json (stage_age_ratio denominator)
eval.py         # spec metrics: per-arm stage acc, boundary timing (frames),
                # ledger F1 — held-out episodes AND held-out tasks
serve.py        # StageEstimator.update(dino_tokens, depth, proprio) -> contract
smoke_test.py   # CPU, no data: shapes, masks, soft boundaries, serve loop
```

Gates (ownership doc): overlay validation on ≥4 task families before training
is trusted; stage head v1 + offline metric table ~Aug 8; serve integration
behind the G3 point-conditioned policy ~Aug 15.

# Stage head v2 — first training run (2026-07-19, A40)

Setup: 4 tasks x 4 episodes from `behavior-1k/2026-challenge-demos`
(turning_on_radio, picking_up_trash, tidying_bedroom,
collecting_childrens_toys), official-annotation labels only (no Phase-1
extractor tracks yet), dinov2_vitb14 backbone, 8000 steps bs 32, held-out
EPISODE per task (no held-out-task split at n=4 tasks). Checkpoint:
box `/root/stage_ckpt/best.pt` (val stage-acc 0.5746 @ step 6000).

## Held-out-episode metrics (eval_heldout_eps.json)

| task | stage acc | bnd med (fr) | bnd recall | ledger F1 | prog MAE | mono viol | H bnd/mid |
|---|---|---|---|---|---|---|---|
| collecting_childrens_toys | .684 | 10.0 | .833 | .800 | .177 | .014 | .280/.178 |
| picking_up_trash | .359 | 9.5 | .333 | .726 | .225 | .004 | .193/.223 |
| tidying_bedroom | .477 | 8.0 | .643 | .803 | .232 | .011 | .346/.189 |
| turning_on_radio | .635 | 7.0 | .750 | .000 | .174 | .014 | .264/.086 |

Mean stage acc 0.539 over 4 tasks; boundary median 7-10 cache frames
(1.2-1.7 s @ 6 Hz); monotonicity violations ~1% of nominal frames.

## Reading

- Calibration is directionally right on 3/4 tasks (entropy higher at
  boundaries than mid-segment); picking_up_trash inverts it and has the
  weakest stage acc/boundary recall -- it is also the most bimanual-parallel
  task, i.e. exactly where official-only (single-track) labels are wrong per
  arm. Phase-1 extractor per-arm attribution (enrichment layer 1) is the
  expected fix, not more steps.
- turning_on_radio ledger F1 = 0.0 is a LABEL artifact, not a model failure:
  the v1 ledger proxy flips literals only on completed place/insert/close
  segments, and the toggle-family literal never fires. Replaced by real
  predicate evaluation over recorded sim state (v3.9 KnowledgeBase API,
  enrichment layer 4).
- 3 train episodes per task is far below the spec regime; treat these numbers
  as pipeline-proof, not the Aug 8 gate. Gate needs: more episodes, held-out
  TASKS, the kill ablation vs temporal-position bins, and the overlay review.

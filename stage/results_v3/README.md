# Stage head v3 run — 2026-07-20 (L40S)

Second full training run, scaled from 4 → 10 task families with a proper
held-out-**task** split. Same v2 architecture and losses; two additions this
run: serve now emits a `grounding_query` per arm (routed back to the
orchestrator per the updated system design), and `render_overlay.py` produces
live demo videos of the serve loop.

## Setup

- 8 training tasks × 4 eps (last ep per task held out): turning_on_radio,
  picking_up_trash, tidying_bedroom, collecting_childrens_toys,
  picking_up_toys, putting_shoes_on_rack, storing_food, sorting_vegetables
- 2 **held-out tasks** × 4 eps, never trained: putting_away_toys,
  tidying_living_room
- 24 train episodes (29.5k windows), 12k steps, bs 32, bf16, DINOv2 ViT-B/14
- Best val stage-acc **0.6711** @ step 6000 (v2 run: 0.5746)

## Headline

| split | mean stage-acc |
|---|---|
| held-out episodes (8 tasks) | **0.619** |
| held-out tasks (2 tasks) | **0.624** |

**No generalization gap**: unseen tasks score the same as unseen episodes of
training tasks — the task-agnostic skill taxonomy + hashed literal encoding
transfer as designed. putting_away_toys (never trained) hit 0.705, beating
five of the eight training tasks.

## Per-task (held-out episodes)

storing_food 0.811 · putting_shoes_on_rack 0.791 · collecting_childrens_toys
0.731 · turning_on_radio 0.589 · sorting_vegetables 0.583 · picking_up_toys
0.524 · tidying_bedroom 0.473 · picking_up_trash 0.449

Known label issues persist and still dominate the tail: picking_up_trash
(bimanual-parallel arms share one label — needs per-arm attribution) and
turning_on_radio ledger F1 = 0.000 (degenerate p_sat targets — needs real
KnowledgeBase predicate eval). Same failure modes as v2, now with more
counter-examples of healthy tasks (storing_food ledF1 0.731, boundary median
6 frames, recall 0.935).

## Data-pipeline fix found this run

`official_annotations.load_segments` crashed on multi-interval
`frame_duration` (skill interrupted and resumed, e.g. `[[8664,9403],
[9743,9951]]` in sorting_vegetables ep4002) — now emits one segment per
interval. scipy also had to be added to the box deps for eval's median filter.

## Files

- `eval_heldout_eps.json` / `eval_heldout_tasks.json` — full per-task metrics
- `best.pt` — checkpoint (val 0.6711 @ 6000); `train_log.json`, `stage_medians.json`
- `overlay_turning_on_radio_ep003_2x_v3ckpt.mp4` — live serve-loop demo
  (per-arm stage vs GT, progress, p_sat, entropy, grounding query). Live acc
  on this episode: L 0.597 / R 0.579 (the v2-checkpoint render of the same
  episode: L 0.635 / R 0.637 — single-episode noise; offline radio acc rose
  0.55 → 0.589 between checkpoints)

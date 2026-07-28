# Stage head v4-corrected — 2026-07-28 (L40S)

Re-run of v4 with the **kinematic predicate evaluator fixed**. v4's
`object_states` OnTop/Inside/NextTo returned all-zero under playback (contact
data disabled), collapsing tidying_bedroom + putting_shoes_on_rack ledger F1 to
0.000. This run evaluates kinematic relations from object AABBs
(`stage/aabb_predicates.py`); logical/toggle predicates keep object_states.

**Result: the target bug is fixed, but the run surfaced two follow-on issues —
one already fixed, one open. Not yet a clean final.**

## Setup

Same architecture/scale as v3/v4 (12.69M params, DINOv2, 12k steps, full
10-family split). Predicate sweep re-run on the 4 real-ledger families
(turning_on_radio, picking_up_trash, putting_shoes_on_rack, tidying_bedroom;
16 eps) with the AABB-fixed evaluator; other 6 families proxy. Cross-run
validation first: picking_up_trash/201 `inside()` gave pos-rate 0.212 with a
clean flip at 79% (can dropped in bin) — real signal where v4 was degenerate.

## Ledger F1 — the headline

| task | v3 | v4-buggy | **v4-corrected** |
|---|---|---|---|
| putting_shoes_on_rack (touching/nextto) | 0.126 | 0.000 | **0.966** |
| tidying_bedroom (nextto/ontop) | 0.828 | 0.000 | **0.940** |
| picking_up_trash (inside) | 0.593 | 0.264 | **0.444** |
| turning_on_radio (toggled_on) | 0.000 | 0.952 | **0.087** |

Shoes and bedroom recovered from dead (0.000) to 0.94–0.97. The genuine
placement literals flip correctly: `touching(sandal,hall_tree)` on at frame
2660, `nextto(sandal,bed)` at 3370, `ontop(hardback,booth)` at 9380 — 0→1 when
objects are actually placed. **The evaluator fix works.**

## Two follow-on issues

**1. Self-comparison inflation (FOUND + FIXED, commit after this run).**
Several shoes/bedroom literals read 1.0 (satisfied every frame). Cause: same-
category relations like `nextto(sandal,sandal)` bound the same instances as
both target and reference, so an object was compared to *itself* — trivially
True (AABB gap 0). These spurious always-1 literals inflate the shoes/bedroom
F1 above (their true placement literals are sparser: pos 0.05–0.67). Fixed by
skipping `r is t` in the evaluator; a re-run will report the un-inflated F1.

**2. Radio regression: 0.952 → 0.087 (OPEN).**
Radio is a toggle on the unchanged object_states path — its targets are
identical to v4. The regression is an *interaction effect*: the p_sat head is
a single head shared across tasks. In buggy-v4 shoes/bedroom fed empty ledgers,
so the head specialized on radio's clean toggle; now they feed dense (and
partly spurious-1) ledgers, shifting the shared head's prior toward "satisfied"
and degrading radio's calibration. The self-comparison fix removes the spurious
density; if radio doesn't recover after that, the head needs per-task p_sat
balancing (normalize base rates / reweight) so sparse toggles and dense
kinematic ledgers coexist.

## Stage accuracy

| split | v3 | v4-buggy | v4-corrected |
|---|---|---|---|
| held-out episodes (8) | 0.619 | 0.6205 | 0.6045 |
| held-out tasks (2) | 0.624 | 0.5394 | **0.6005** |

Held-out **tasks** recovered (0.54 → 0.60, near v3) — the buggy-v4
generalization gap largely closed. Per-task stage-acc for the kinematic tasks
stays low (trash 0.316, bedroom 0.363) — a separate transition-class / per-arm
effect, unchanged by the ledger fix.

## Kill ablation — still passes

Position-bin head: 0.037 self-labeled / 0.16 on semantic labels (chance ≈ 0.028
over 36 bins) vs semantic 0.60. Semantic labeling carries the structure.

## Verdict & next

The bug you set out to fix — all-zero kinematic ledgers — **is fixed**; shoes
and bedroom carry real, learnable placement signal again. The run then exposed
(1) a same-category self-comparison artifact (now fixed in code) and (2) a
shared-p_sat-head imbalance that regressed radio. The clean next run: re-sweep
the 4 families with the self-comparison fix, retrain, and check whether radio
recovers; if not, add per-task p_sat base-rate balancing. Only then does v4
cleanly supersede v3.

## Files

- `eval_heldout_eps.json` / `eval_heldout_tasks.json` — semantic head per-task
- `best.pt`, `train_log.json`, `stage_medians.json`
- `posbins/` — kill-ablation head evals + log
- compare with `stage/results_v4/` (buggy) and `stage/results_v3/` (baseline)

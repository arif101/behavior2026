# Stage head v4 run — 2026-07-24 (L40S)

Third full run. Implements the three v3 "next steps" plus the spec's kill
ablation: **per-arm attribution**, **real BDDL goal literals**, **real
per-frame predicate evaluation** for p_sat targets, and the
**temporal-position-bin kill ablation** (spec eval 4).

**This is a mixed result — read the whole thing.** The flagship real-ledger
fix works dramatically on toggle-type goals, but the kinematic-predicate
evaluator is buggy and regressed several tasks. v4 does **not** cleanly
supersede v3 yet.

## Setup

Same architecture/scale as v3 (12.69M params, DINOv2 ViT-B/14, 12k steps,
bs 32, bf16), full 10-family split (29,540 train / 10,832 val windows).
The predicate sweep proved infeasible at full scope (episodes are 16k–36k
physics steps at ~1 step/sec; no-render validated but same rate — the
bottleneck is per-step scene-state deserialization, not rendering), so v4
uses **graceful degradation**: 17/40 episodes have real ledger + extractor
per-arm labels; the other 23 fall back to proxy ledger + official-only arms;
real literals and the posbins ablation apply to all 40.

Real-ledger coverage (5 of 10 families): turning_on_radio 4/4,
picking_up_trash 4/4, putting_shoes_on_rack 4/4, tidying_bedroom 4/4,
picking_up_toys 1/4.

Best val stage-acc **0.6564** (v3: 0.6711; lower partly because v4 adds a
TRANSITION class, 36 stages vs 35).

## Headline: the real-ledger fix works on toggle goals

**turning_on_radio ledger F1: 0.000 → 0.9524**, stage-acc 0.589 → 0.800.
The `toggled_on(radio)` predicate is cleanly evaluated (satisfied ~30% of
frames) and the head learns it. The degenerate p_sat case that gave F1=0.000
across v2/v3 is gone. This validates real predicate evaluation for
logical/joint-state goals.

## Regression: kinematic predicate evaluation is broken

`nextto`/`ontop`/`inside` evaluation via OmniGibson `object_states` returns
**all-zero** for the placement tasks:

| task | real p_sat positive rate | ledger F1 v3 → v4 |
|---|---|---|
| turning_on_radio (toggled_on) | 0.29–0.39 ✓ | 0.000 → **0.9524** |
| picking_up_trash (inside) | 0.04–0.21 (sparse) | 0.593 → 0.264 |
| tidying_bedroom (nextto/ontop) | **0.0 ×3** | 0.828 → **0.000** |
| putting_shoes_on_rack (ontop) | **0.0 ×4** | 0.126 → 0.000 |

For tidying_bedroom and putting_shoes_on_rack every relational predicate reads
False on every frame — impossible for successful demos — so the real ledger is
empty, the head predicts ~zero, and F1 against all-zero truth is 0.000. The
real ledger *replaced* v3's segment-completion proxy (bedroom scored 0.83
there), so for kinematic-goal tasks it made ledger F1 **worse**. This is an
evaluator bug, not a head failure — likely `OnTop`/`NextTo`/`Inside` needing a
physics settle/contact refresh after `load_state` playback, or a
reference-binding issue. **Fix before trusting the real ledger on the ~79% of
challenge goals that are kinematic placement.**

## Stage accuracy (semantic head)

| split | v3 | v4 |
|---|---|---|
| held-out episodes (8 tasks) | 0.619 | **0.6205** (≈ v3) |
| held-out tasks (2 tasks) | 0.624 | **0.5394** (down; generalization gap returns) |

Held-out episodes hold at v3 level. Held-out **tasks** dropped and a
generalization gap reappeared. Per-task regressions concentrate in
tidying_bedroom (0.473→0.320) and picking_up_trash (0.449→0.396) — separate
from the ledger issue, attributable to the new TRANSITION class and/or the
per-arm attribution relabeling. Needs isolation (ablate transition class;
diff per-arm vs official-only labels).

## Kill ablation (spec eval 4): semantic wins

Identical head trained on temporal-position bins vs semantic skills:

- Position-bin head predicts its own bins at only **0.119** val / 0.031
  held-out-task (chance ≈ 0.028 over 36 bins) — temporal position is barely
  learnable.
- Semantic head learns skills to **0.62** held-out-eps.
- On the label-space-independent metrics, the bin head is worse: boundary
  recall 0.11–0.69 (mostly ~0.3) and progress MAE 0.22–0.27, vs the semantic
  head's higher recall and 0.13–0.24 MAE. Even radio ledger F1 is 0.608 for
  the bin head vs 0.952 semantic.

**Semantic labeling bought real structure** — the ablation passes.

## Verdict & next steps

1. **Debug kinematic `object_states` eval** (nextto/ontop/inside all-zero) —
   the blocker for the real ledger on placement goals. Prime suspect: state
   needs a physics settle after `load_state`.
2. **Isolate the stage-acc regression** — ablate the TRANSITION class; diff
   per-arm vs official-only arm labels on tidying_bedroom/trash.
3. Then re-run. The radio result shows the mechanism is sound; the rollout
   needs these two fixes before v4 supersedes v3.

## Files

- `eval_heldout_eps.json` / `eval_heldout_tasks.json` — semantic head, per-task
- `best.pt` — semantic checkpoint (val 0.6564); `train_log.json`, `stage_medians.json`
- `posbins/` — kill-ablation head: `train_log.json`, `eval_heldout_tasks_selflabels.json`
  (bins scored on bins), `eval_heldout_{eps,tasks}_semlabels.json` (bins scored on
  semantic labels — comparable ledger/progress/boundary rows), plus `best.pt`
- v3 baseline for comparison: `stage/results_v3/`

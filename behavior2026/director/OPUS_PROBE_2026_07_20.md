# Offline Opus strategy probe — first live results (2026-07-20)

First live run of the deliberative tier's episode-start consult (`harness/opus_consult.py`,
claude-opus-4-8, ~800 tok/consult) against the human demonstrators' actual openings on 12
real episodes (3 per G3 gate task). Situations built by `harness/situations.py` from deep
labels + skill annotations; **no human ground truth ever enters the prompt**.

## Scores (geometry-only briefing: objects_at_start + pairwise distances)

| task | first-target agree | full opening agree | Opus behavior |
|---|---|---|---|
| attach_a_camera_to_a_tripod | 3/3 | 3/3 | camera→tripod, matches human exactly |
| picking_up_trash | 0/3 | 0/3 | dual-arm BATCH cans; human always CARRIES the trash_can |
| thawing_frozen_food | 0/3 | 0/3 | grabs food directly; human opens fridge, lifts PLATES |
| turning_on_radio | 3/3 (target) | 0/3 (skill) | presses in place; human picks radio up first |

Opus is **perfectly consistent across instances** of the same task (same strategy 3/3 each)
— deterministic enough to cache one consult per task family.

## Finding 1 — briefing content, not model capability, is the binding constraint

Enriching ONE thawing briefing with BDDL-style initial state
(`fridge door CLOSED; chicken ON plate_79 INSIDE fridge`) flipped Opus to the exact human
strategy: *move to fridge → open → dual-arm pick up the PLATES → carry to thaw* — including
the non-obvious plate-level manipulation choice it had wrong before. One field, full fix.

→ **OpusBrain briefing MUST include**: containment relations, articulated-door states, and
support relations from the task BDDL initial conditions (all available at episode start, no
privileged eval info — it's the task spec).

## Finding 2 — demonstrator habits are not derivable from state; feed them as priors

Trash: a neutral strategy MENU (shuttle/batch/carry-container/stage) did NOT flip Opus to
carry-the-container; geometry makes batching look locally optimal. But the deep labels show
the human carries the trash_can 2.7–5.7 m in **every** episode — the policy's training
distribution IS carry-container. A director that orders "batch the cans" fights the policy.

→ Include per-task **demo-opening priors** in the briefing (mined from skill annotations —
training data, legitimate at eval). → Treat the Opus plan as a **weak prior**: never veto
the policy's revealed strategy when q-progress is positive; consult again on stall instead.

## Finding 3 — decision schema must speak the skill taxonomy

Opus planned "open the fridge once" in prose but could not express it in `first_actions`
because the schema's skill enum lacked `open door`/`close door` (now fixed). Schema
vocabulary = the ~70-verb annotation taxonomy subset the reflex tier can dispatch.

## Wired conclusions for OpusBrain v1

1. Briefing = task literals + BDDL initial state (containment/doors/support) + object
   positions (belief-seeded) + pairwise distances + demo-opening prior for the task family.
2. One consult per episode start, cached per task family (consistency finding); re-consult
   only on reflex-tier escalation (stall/abandon-cascade), not per tick.
3. Opus output = advisory ordering + macro suggestions; reflex ledger remains the executor.
4. Cost: ~800 tok × (1 + escalations) per episode — negligible vs the 10–30 calls/ep budget.

## Corroborating data from the assembler (same day)

Annotation-derived task targets for `picking_up_trash` = [can_of_soda, **trash_can**] —
the carry-container strategy is visible in `manipulating_object_id` and now supervises
target_points (coverage 93.7% / plausibility 87.0%, better than the old hand task_targets
82.6% / 84.7%).

# BDDL interaction-family census — all 100 challenge tasks (2026-08-13)

Source: meta/tasks.jsonl (behavior-1k/2026-challenge-demos) × goal predicates from
StanfordVL/BEHAVIOR-1K@main bddl3 activity definitions. Per-task detail:
box_artifacts/bddl_census_2026_08_13.json. Predicate→family map in the census script
(session scratch; trivially re-derivable). Counts below merge the 6 first-pass UNMAPPED
labels (onfloor→pick&place ×4, dusty/stained→coat ×2).

| Interaction family | Tasks requiring it | Notes |
|---|---|---|
| pick&place (grasp + place) | **79** | **46 tasks need NOTHING else.** Skill #2, decisively. |
| articulated open/close | 20 | doors/drawers/cabinets — skill #3 candidate |
| coat/wipe/spread | 14 | tool-mediated surface work |
| cook (appliance) | 8 | composite: place + toggle + wait |
| press/toggle | 6 | the pilot family (turning_on_radio here) |
| attach/insert, pour | 4 each | long tail |
| cut, deformable, hang, break | ≤3 each | long tail |

69/100 tasks are single-family. Roadmap implication: after the commit-skill pilot proves
the pattern on press/toggle (6 tasks), grasp/place is the family that moves q — it gates
~79 tasks and fully unlocks 46. Articulated open/close is the clear #3 (20 tasks).
Skill-2 design question flagged: pick&place has TWO terminal events (grasp-initiation,
place-release) — one conditioned skill vs two; decide on VLA place-competence evidence,
not assumption.

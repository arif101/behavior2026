# CONTEXT STATES SPEC — task-0062 `halve_an_egg` (DRAFT v0.1, 2026-08-26)

Status: **PROPOSED — no relabeling, training, or eval has started.** Bars in §6 are frozen the
moment the first training run launches; they never move after data arrives.

## 1. What Step 0 established (see STEP0_GAP_REPORT_62.md)

- Demos are eval-legal: every grasp is a native assisted-grasp contact grasp; robot pose replays
  with 0.0 cm error. The "assist-regime" mismatch the original plan assumed does **not** exist on
  task-62. Replay failures are object-displacement drift over long windows.
- Therefore the context vector's job is NOT to reconcile two physics regimes. Its job is to give the
  VLA the **task-phase and world-state facts the pixels under-determine** (which cycle of a
  multi-object task it is in, what each hand holds, whether the egg has been sliced, how far the
  goal is), so that the 200 demos — which share one 16-skill sequence — become an unambiguous
  supervision signal at every frame, and so that RL-generated clips for the precision families can
  be mixed in with an explicit phase label instead of being averaged against demo behavior.

## 2. c(t) — per-frame context vector (proposed, 16-d)

| # | channel | dims | offline label (training) | live value at serve (LEGAL inputs only) | dropout |
|---|---|---|---|---|---|
| 1 | skill family one-hot | 8 | organizers' `skill_annotation` frame ranges: {navigate, open_door, pick_up_from, place_on, chop, place_in, close_door, other}. Boundaries ±30 frames are label-smoothed. | stage head (existing aux head, currently 4-way; retrained 8-way on these labels). Fallback: last predicted family. | p=0.2 |
| 2 | progress q(t) | 1 | `goal_status` from **per-segment near-anchor replay** (never spike sums, never continuous replay); piecewise-constant in {0,.2,.4,.6,.8,1}. | stage head progress output (already exists). | p=0.2 |
| 3 | composition: post-slice | 1 | 1 for t ≥ slice frame (reward spike +0.4 / replay). | 1 if live progress ≥ 0.4 (the two `real(half)` conjuncts are the first 0.4). | tied to #2 |
| 4 | AG-held per arm | 2 | `_ag_obj_in_hand` from the recorded state (sequential restore stride 10 — pilot ran stride 5, but AG changes only at the ~8 closures per demo, so stride 10 loses nothing) | gripper qpos + closure latch (proxy; error measured offline vs #4 labels before use). | p=0.2 |
| 5 | held-object class per arm | 2×2 | {none, egg/half, knife} from the AG object name (2 bits per arm). | from stage head family + #4 (a pick_up_from(knife) family with AG-held ⇒ knife). | tied |
| — | anchor-quality / replay-drift label | 0 | per-segment replay outcome from the sweep | **not an input** (no live counterpart). Used as a per-frame **sample weight**: segments whose near-anchor replay reproduces the human outcome get weight 1.0, others 0.5; frames inside rollback branches are excluded. | — |
| — | assist regime | 0 | — | **dropped**: no evidence on task-62 (Step 0). Revisit only if a family shows rig-assisted closures in the recordings. | — |

Injection: AdaLN additive term on the action expert via `_embed_cond_extras` (template
`patch_map_adaln.py`): zero-init MLP created after all existing modules, `tanh(x/4)` bounding,
`missing_regex` `.*context_cond.*` in the weight loader, new data key `context` through the
trap-triple (`model.py` field + `from_dict` + `preprocess`; `b1k_policy.py` packing;
`training/config.py` repack) and attached post-flatten in `_preprocess_obs` at serve. Every dataset
in the mix carries the `context` column (RepackTransform KeyError otherwise).

## 3. Relabel procedure — OFFLINE, no simulator (revised 2026-08-27; `task62/relabel_offline.py`)

Finding (2026-08-27): every channel is already in the release; the sim sweep re-derived them worse.
1. Skill family: organizers' `skill_annotation` (unchanged).
2. AG-held + held-object per arm: decoded from the recorded `state` vector. OmniGibson's robot
   `serialize()` appends `[_AG_MAGIC, arm_idx, obj_uuid, link_idx, 14 frame floats, joint_type]` per
   grasping arm; `uuid = float32(md5(name) % 1e8)` resolves against the scene template (halves:
   `half_hard_boiled_egg_231_{0,1}`). Frame-exact. Agreement with the sim labels on the 7 sim-labeled
   demos: 99.3–99.8 % of frames (the residue is the sim's stride sampling). Adds a 4th class
   `other` (fridge door — grasped in 197/200 demos, which relabel_v1 mislabeled as egg).
3. Progress q(t): scoring segments in annotation order (chop +0.4, place-in knife +0.2, place-on half
   +0.2 ×2), each matched to the nearest POSITIVE reward spike of the right size (a +0.4 also satisfies
   a place-on: both halves flipping in one frame); q steps at the spike frame. Negative/flicker
   rewards ignored. Result: 198/200 demos reach q = 1.0 (= the `task_success` set); the 2 non-success
   demos are flagged `usable=False` and excluded. The spike precedes the annotation's segment end by a
   median 97 frames, so timing is tighter than the sim labels (which stamped q at segment end).
   Step 0's "reward not usable as q(t)" holds for the raw cumsum (flickers in 154/200 demos), not
   for positive spikes matched in scoring order. On the 7 sim-labeled demos the sim replay stalled at
   q = 0.8 in 4 where the recording (and task_success) reach 1.0 — the sim was the less reliable source.
4. Post-slice: t ≥ chop spike frame.
5. Anchor-quality sample weight: dropped (sim-only, lowest-value channel). `context_weight` = 1.0.
Cost: ~1 min for all 200 demos on CPU. Output `/root/step0/relabel_offline/ep{raw}.npz` (+ `summary_offline.json`),
consumed by `build_t62_dataset.py --v1 /root/step0/relabel_offline` unchanged.
The sim sweep (`relabel_sweep.sh`, `relabel_v1.py`) was stopped 2026-08-27 02:5x UTC after 7 demos; kept for reference only.

## 4. Data mix (Run-3 / task-62)

- 200 task-62 demos, relabeled, videos from `behavior-1k/2026-challenge-demos` (per-key chunk/file
  columns, NOT data indices — boundary-shard trap), poses from replay. Held out: 19 demos (every
  10th) for stage-head/BC validation.
- RL-factory clips (later, only if Step-0-identified families need them): half→plate,
  knife→sink, second-hand half grasp — strict physics, floor-adjusted deeper anchors
  (random floors from segment−30: 44% / 28% / 33%), success-terminated including fumbling
  prefixes, labeled with the same c(t) (family from the wrapper, q(t) from `goal_status`).

## 5. Ablations that ship with the first training run (Law 1)

- **Liveness**: zero c(t) at eval → success must drop; if it doesn't, c is decorative — stop.
- **Counterfactual**: force family channel to a wrong phase (e.g., "navigate" during a grasp)
  → behavior must change (commit-probe mass, approach distance).
- **Held-out stage-head accuracy** on the 19 held-out demos: family ≥ 85%, progress MAE ≤ 0.1.
  Below that, live c(t) is too noisy to condition on; fall back to dropout-heavy training.

## 6. Pre-registered eval (FROZEN at first training launch)

- Harness: CHALLENGE wrapper (head 720 / wrist 480 + depth), `eval/rgbd_full_res_fixed.py`, NOT the
  224 debug wrapper. Instances 302–320 (301 quarantined). 1 rollout/instance first pass
  (19 × ~100 min ≈ 32 h), 2nd rollout only if the first pass is within 1 instance of a band edge.
- Arms: (A0) random floor per conjunct from the task start — expected 0; (A1) BC baseline =
  same data, no context; (A2) context-VLA, live c(t); (A3) A2 with c(t) zeroed (liveness);
  (A4) A2 with c(t) forced to the demo-phase schedule (oracle-phase upper bound, NOT a submission).
- Bands on A2, mean q over 302–320: **STRONG ≥ 0.60 with ≥ 5 instances at q = 1.0 — the success
  bar, CONFIRMED by the user 2026-08-27; frozen at first training launch** · WEAK 0.30–0.59 · FAIL < 0.30.
- Tripwires: (T1) A2 − A3 < 0.05 → context decorative, report and stop; (T2) held-out family
  accuracy < 85% → live proxy unfit; (T3) ≥ 2 of 8 filmed place successes show the half arriving
  by a sweep/drop rather than a controlled place → add a speed-at-release penalty in the RL
  factory before believing the number.

## 7. What is deliberately NOT in v0.1

- No assist-radius / regime channel (no evidence).
- No object poses or target points in c(t) (illegal at eval; the existing affordance-point route
  stays as-is and is orthogonal).
- No per-instance identity (memorization tripwire from VALIDATION_62).

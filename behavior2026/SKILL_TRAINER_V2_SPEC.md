# SKILL TRAINER V2 SPEC — multi-scene geometric skill factory (2026-08-16)

Provenance: v1 campaign FINAL (stage-1+2 complete, logbook 2026-08-16): 1,091 episodes,
336 successes (31%), 232,555 env steps, 54.0 h @1.20 st/s; sim gate (spec v1 §5) never
reached — stopped at rung 2 of 4; rate DEGRADED across rungs (43%→21%); 13/32 scenes at
zero; 5 predicted-cold chunks burned 5 h for 0/75. Kept assets: RL manufactures the commit
mode (only method that does); warm-start chain compounds (d170→d270→d290 = 0/10-cold →
8/23 → 21/25); 336 banked commit demos; probe/forensics stack. Verified 2026-08-16: radio
yaw is FULLY randomized per scene (±179°, instance files) — a blind policy cannot
disambiguate approach strategy; v1's zeroed-L2 "simplification" is falsified, not pending.

Design rule carried over: architecture task-agnostic; task-specificity lives ONLY in
data/labels/reward. Learner (rlpd_sac.py) unchanged — the harness is what failed.

## 1. Design deltas vs v1 (each traceable to a measured failure)

- D1 ONE multi-scene policy. Round-robin all non-held-out scenes in one training run,
  shared online buffer, one checkpoint lineage. Isaac env-reuse leak → process pool (one
  sim process per scene FILE, cycled; weights+buffers live in the parent). Kills the
  per-scene re-convergence tax the chain evidence says is unnecessary.
  [v1 failure: 50 serial chunks re-learning the same skill]
- D2 UN-BLIND: fill the L2 slot. Wrist-local geometry features in the existing obs slot
  (skill_env_wrapper.py:303 currently zeros it). Training-time source: synthesized from
  privileged sim geometry (radio mesh pose + table plane voxelized to the L2 spec format,
  + noise), validated against real mapper output on ≥100 frames (feature-fidelity check,
  cosine ≥0.9) so the eval-time map can drop in. Keeps render-off (D5) compatible.
  Single-variable arm: blind control retained (A0).
  [v1 failure: yaw-randomized scenes unsolvable blind; 93% of failures diverge >15 cm]
- D3 Episode-denominated budgets: 25 episodes/scene/round, equal for all scenes; keep
  early-stop (rolling ≥0.95 → scene graduates the rung); SEEDING TRIGGER: 0 successes by
  episode 12 → inject demo-replay successes (replay the demo's own actions from the
  restore frame) into the online buffer, then resume RL.
  [v1 failure: step budgets gave cold scenes 10 episodes and healthy ones 50]
- D4 Snapshot restores: per bank entry, run playback-restore + AG re-establish + settle
  validation ONCE, then og.sim.dump_state → npz; all subsequent resets load the validated
  snapshot. Re-validate weekly or on physics-config change.
  [v1 failure: 50–63 s reset tax per episode; d30/d50 banks unusable from transients]
- D5 Render-off during RL (obs never reads RGB); re-render successful trajectories with
  cameras afterwards for flywheel export. Rendering budget goes to the flywheel, not RL.
  [v1 waste: cameras rendered every step for a blind policy]
- D6 Per-state frontier curriculum: per bank entry success-rate tracking; sample entries
  ∝ (1 − rate) with a mastered floor (anti-forgetting, keeps v1's 50/50 lesson); rungs
  become a continuous frontier including harvest-bank boundary states from day 0.
  [v1 failure: rung cliffs (0/20 between −5 and −25 needed a hand-inserted rung)]
- D7 Flight recorder: one JSON per episode (scene, entry, start dist, terminal dist,
  steps, contact steps, success, seed-source, ckpt-id) → logbook ingests directly.
  Stats dumped on EVERY exit path (v1 lost the d260 early-stop tail).
- D8 Reward unchanged for radio (sparse + bounded shaping, hacking guard). Grasp port
  (§3) uses: sparse AG-engaged ∧ lifted ≥5 cm ∧ held ≥15 steps, + potential-based
  shaping on dist(EE→object) then dist(object→lift target); debounced.

- D9 ONE goal-conditioned contact policy (amended 2026-08-16, same day). The grasp port
  is NOT a separate network: a single skill conditioned on [affordance frame (point +
  approach normal) + L2 + gripper state + FAMILY/GOAL code], with EE state expressed
  RELATIVE to the affordance frame so contact mechanics are object-agnostic (press =
  advance along −normal + dwell; grasp = close across it + hold). Conditioning rule
  banked from G3/PHASE2_PLAN §(f): NEVER per-task ID embeddings (measured-dead routing);
  family/goal codes only (≤16), geometry does the rest. Routing is implicit: the gate
  decides WHEN (learned), the conditioning decides WHICH (continuous) — no dispatcher.
  Radio+trash train in ONE buffer (per-task symmetric sampling).
  [v1 limitation: per-task skills are O(tasks); census says 3 families cover 100 tasks]

## 2. Pre-registered bars (FROZEN at first V2 launch; eval on the 8 held-out demos'
   restore states — never trained, same protocol as v1 would have used)

- B1 EFFICIENCY: held-out success ≥60% at rungs 1+2 within ≤75,000 env steps
  (v1 spent 232,555 for 31% on TRAIN scenes; this is ≥2× competence at ⅓ cost).
- B2 COLD CONVERSION: ≥6 of the 11 true-cold scenes (70,110,130,160,220,240,320,340,
  350,370,410) reach ≥3 successes in their 25-episode budget. (d30/d50 excluded —
  bank-validity fix D4 is their remedy; count them separately.)
- B3 L2 ATTRIBUTION: L2-on (D2) beats blind control (A0) on held-out by ≥15 pp
  (one-sided, n≥100 episodes/arm). If it doesn't, un-blinding is NOT load-bearing →
  wrist-depth-patch variant gets one arm, then the input question closes.
- B4 BOUNDARY: ≥50% from harvest-bank boundary states (the states the runtime gate /
  flywheel actually needs) within the same 75k-step budget.
- KILL: if V2 at 150,000 env steps is below v1's held-out equivalent, revert to v1
  recipe for radio and escalate the architecture question to the flywheel-only path
  (skill purely as data factory from v1's existing 336 successes).

AMENDMENT (Arif, 2026-08-17): **the eval-time backstop is REMOVED.** The skill never runs
at evaluation; the shipped system is the VLA alone. On flywheel/probe failure the response
is ITERATE (more harvest data, next Run-3 arm), never deploying a second controller. The
serving-gate machinery (spec v1 §1) is retired; gate signals survive only as diagnostics
(flight recorder) and as reward/curriculum inputs. Geometry-revival arm updated: the
AdaLN geometry source switches from coarse L0/L1 rows to L2 wrist-frame features, which
are non-redundant at contact (gripper occludes cameras; L2 is kinematics-built).

PRIORITY DIRECTIVE (Arif, 2026-08-16): the skill families to prioritize are
**pick_up_from, place_in, place_on** — matching the census (grasp+place in 94/100 goals;
ontop/inside dominate). Press/radio is the proof-of-recipe, not the priority; the D9
goal-code set and curriculum order follow this. Family rewards from BDDL predicates:
pick_up_from = AG-engaged ∧ lifted ∧ held; place_in = inside ∧ released; place_on =
ontop ∧ released ∧ stable (final-state scoring — the boxes topple case). The 4 expansion
tasks exercise exactly these three families.

## 3. Grasp port (starts after V2 smoke, NOT after radio polish)

Task picking_up_trash (task-0001; same scene as radio; task_objects entry exists; data
path verified). New per-task pieces ONLY: target resolver ("can_of_soda" nearest
not-retired), reward (D8), bank builder run on its 200 demos (grasp anchor = first
AG-engage frame; rungs = frames before it). Bar (pre-registered): first ≥10 held-out
grasp successes within 50k env steps. This is the task-agnosticism test of the whole
architecture — if the port needs more than data/labels/reward, the design rule failed.

- B5 REUSE (pre-registered with the port): shared goal-conditioned policy (D9, radio+
  trash one buffer) vs separately-trained controls, transfer measured BOTH directions on
  held-out states. Shared ≥ separate on both tasks → the sub-policy layer collapses to
  one network permanently; shared < separate by >10 pp on either → interference is real,
  fall back to per-family nets and record why.

## 4. Build plan

- d0: throughput matrix (render on/off × 1/2 procs × playback vs snapshot restore) —
  numbers into the logbook; D4 snapshot bank builder + re-validation of d30/d50.
- d0 (parallel, GPU now free): pre-retrain probe baselines on ckpt radio_run2@49999 —
  point_liveness, commit probe (re-confirm 0.00), denoising-variance readout (build),
  key-causality baseline. These gate the flywheel and must predate Run-3.
- d1–d2: V2 driver (process pool, shared buffers, D3/D6 scheduling, D7 recorder);
  L2 synthesis + fidelity check; smoke = 3 scenes (1 cold) reach rung-1 ≥50% overnight.
- d3–d5: full radio V2 run (arms: A0 blind / A1 L2-on / A2 +seeding) vs bars B1–B4.
- d5–d7: grasp bank builder + port smoke (§3).
- Then: flywheel export of all V2+v1 successes → Run-3 per the review addendum ladder.

## 5. Explicitly not in V2

2-env GPU parallelism (VRAM-gated; revisit after render-off measurement), async learner
(measure first), target-anchored action reparam (norm-stats reset; measured arm later),
any VLA-side change (that's Run-3's job), any new reward terms for radio.

## 6. V2.1 — GRIP + PRESS focus (directive: Arif, 2026-08-18; applied at V2-run end)

Round-0 evidence driving this: press−100 = 0/301 with 38% of the round's budget (rung gap
too wide, sampler unordered); 19/38 s3 states are PRE-PICKUP (press−100 already spans the
combined grip+press problem); d70/d110/d240 cold = grip-infidelity class; harvest states
absent from the training rotation (B4 gap).

Changes (single bank+driver update, one variable-set, new run):
- V2.1-a Grip-fidelity restore: bank builder records the demo's true AG contact point and
  in-hand relative pose at each holding frame; restore warps the object to that pose
  before re-establishing the constraint. Retest seeding on d70/d110/d240 after.
- V2.1-b Harvest states into the rotation (54 boundary states, press-in-place band —
  the VLA-visited distribution B4 certifies; family=press, no holding arm).
- V2.1-c RADIO GRASP ANCHORS: new bank entries at grasp-initiation frames (just before
  first target lift; lift_z baseline recorded at build), family=pick_up_from, success =
  AG-engaged ∧ lifted ≥5 cm ∧ held ≥15 steps. Radio becomes the first pick_up_from
  data source; trash port follows with floor-grasp + place_in.
- V2.1-d press−50 rung (bridges the measured −25 → −100 cliff; v1's −5 → −25 lesson).
- V2.1-e Rung-ordered frontier sampling: within a scene, rung k+1 is sampled only after
  rung k's rolling ≥0.5 there (prevents round-0's 38%-into-the-wall spend).
- V2.1-f Keep combined-span s3/pre-pickup episodes (single ToggledOn reward over
  grasp→press — the DP chaining pressure that prices handoff quality implicitly).

Pre-registered V2.1 bars (frozen at its launch): (i) pick_up_from: ≥10 held-out-scene
grasp successes within 50k env steps; (ii) press−100 nonzero in ≥5 scenes; (iii) ≥2 of
{d70,d110,d240} convert post grip fix; B1–B5 continue to adjudicate on their own runs.

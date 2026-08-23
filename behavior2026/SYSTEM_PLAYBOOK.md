# B26 System Playbook — Skill Flywheel for a New Task

Task-agnostic guide to this repo's architecture and RL methods. Written so a fresh
Claude Code session can bring up the full pipeline on a **new task** (any of the 100
BEHAVIOR-2026 tasks) without re-deriving the design. Radio (`task-0000`) is the
reference implementation; nothing below depends on it.

---

## 1. Architecture in one paragraph

The shipped system is the **VLA alone** (π0.5 flow-matching fork, 3B, 32-step chunks
@30Hz; fork adds AdaLN point conditioning + 8×72 foveated map tokens). There is **no
skill network at eval**. The skill (2M MLP, SAC/RLPD) exists as a **data factory**: it
learns contact-level competence via RL under **eval-exact physics**, and its
reward-certified successes are distilled back into the VLA (the flywheel). Division of
labor: the 2M net explores and certifies; the 3B net generalizes.

Why this exists: human demos are collected through a teleop rig whose assists (ranged
"magnetic" grasping via external `_establish_grasp` calls) produce **action streams
that eval physics will not honor** at exactly the contact-commit moments. RL rollouts
under the eval env's own `grasping_mode: assisted` (contact ∧ between-finger raycast ∧
0.3 s closure) are the only physically-honest supervision source for those segments.

## 2. Pipeline stages for a new task

### 2.1 Data prep
- Raw demos: `/root/rawdemos/task-XXXX/episode_*.hdf5` (state (T+1,·) serialized sim
  states; action (T,23); rollback segments stored separately; NO obs streams, NO AG in
  states — verify by searching state rows for the `_AG_MAGIC` sentinel).
- LeRobot reference dataset: per-episode `observation.state` (61-d proprio), videos
  (rgb + depth_linear per camera), `robot2cam_pose`. Build/verify the
  episode_index↔raw_id map physically (`build_episode_map.py` pattern: length
  fingerprint + metalink-distance verification; naive index assumptions BREAK).
- Metalink labels: project the task's BDDL-relevant metalink (button, handle, rim…)
  per frame → `metalink_labels/ep{d}.npz` (`meta_world`, `meta_base`).
- **Arm audit (mandatory, never default)**: derive per-demo which arm acts from the
  action channels. True layout: `[base 0:3, trunk 3:7, arm_L 7:14, grip_L 14,
  arm_R 15:22, grip_R 22]`; grips are ±1; closure = last open→close crossing before
  the manipulation event. Write `arm_audit.json`; it feeds `active_arm`,
  `holding_arm`, and closure frames everywhere. (Radio lesson generalized: hardcoded
  arm defaults infected three metadata fields before auditing.)

### 2.2 Event frames per family
For each skill family the task needs (`press`, `pick_up_from`, `place_in`,
`place_on`), locate per-demo event frames from labels/predicates:
- terminal event frame (predicate flips: ToggledOn / object inside / ontop…)
- grasp closure frame + first-lift frame (object z from labels; LIFT_THRESH ~3 cm)
- reverse-curriculum anchors: quasi-static frames at offsets before the event
  (radio used −5/−10/−25/−50/−100 + a pre-lift grasp anchor `G`).

### 2.3 Banks
- **Start bank** (`build_start_bank_*.py` pattern): per demo × rung: frame, family,
  audited arms, event frames, `lift_z`-style baselines.
- **Snapshot bank** (`build_snapshot_bank.py`): playback-restore each anchor frame,
  establish holds at TRUE current pose (grip-fidelity: coincident anchors, 0-drift),
  settle, `og.sim.dump_state(serialized=True)` → npz + manifest. Snapshots carry AG
  natively.
- **Validation must be DYNAMIC, not static**: a bank entry passes only if (a) it
  restores still (drift/dz thresholds) AND (b) **demo actions replayed from the
  restore reproduce the demo outcome** (the replay-fidelity gate). Static stillness
  checks alone repeatedly passed unusable states.

### 2.4 The wrapper (`skill_env_wrapper_v2.py`)
One booted sim serves every scene (snapshots load in ~1 s). Key mechanisms, all
family-routed and task-parameterized (`target_name_sub`, predicate registry):
- obs (202-d): 5-frame proprio stack + live target vector (recomputed per step from
  the object's CURRENT pose + per-episode DR bias) + affordance confidence + 64-d
  EE-frame L2 occupancy + family one-hot. **Never a task ID.**
- success predicates: the ONLY per-family code. press → predicate flip;
  pick_up_from → AG ∧ lift ∧ **quasi-static** 15-step dwell (dwell counts only while
  object speed < ~0.15 m/s — "held" means STILL, or RL learns to orbit the object);
  place_* → inside/ontop predicates (to implement).
- reset hold-repair: ensure the audited holder holds (coincident establish, guarded
  against double-pin), release stray welds from other arms.
- **hold_act discipline**: the non-active arm tracks a stationary command computed at
  reset — it MUST be refreshed at any arm/goal switch or the holding arm steers back
  to its stale pose, dragging the object.
- grip latch: while holding, the grip command stays closed unless the policy
  deliberately releases (sustained strong-open) — an untrained stochastic grip channel
  otherwise jitters open and OG releases the weld.
- shaping: approach distance (doubled in held-object transport contexts), contact
  bonus, family-specific progress terms (acquisition bonus; potential-based lift
  progress paying only NEW height; stillness cost once lifted; anti-drift anchor on
  held objects during pressing-type phases).

### 2.5 Assist curriculum (constraint annealing)
The demos' rig granted ranged grasp engagement; eval does not. Bridge that gap with a
**curriculum on the constraint**: the wrapper grants AG establish when the policy
closes its gripper within a per-scene `ag_assist_range` (start 0.30 m = rig parity;
pull the object INTO THE FINGER CAGE before welding — weld-at-range leaves a lever
arm that kills lift transmission), anneal −5 cm per consolidation (rolling ≥0.5), a
**sticky rung** at ≤0.10 m (assist requires true finger contact), radius 0 = native
eval AG. **Flywheel export gates on radius-0 episodes only.**

### 2.6 Trainer (`train_skill_v2.py`, RLPD core in `rlpd_sac.py`)
- SAC, 5-critic REDQ min-over-2, UTD 8. Batches: 75% online / 25% **seed buffer**
  when seeds exist (a dedicated buffer with guaranteed share — demo successes drown at
  ~1% otherwise and TD stalls). Retire any prior buffer whose physics no longer match
  the current world; stale replay data actively fights relearning.
- **Seeding** = physical prompts through the buffer: on per-FAMILY starvation
  (N successless episodes of that family in a scene — scene-level triggers get masked
  by the other family's successes), replay the demo's own actions from a playback
  restore (`restore_to_frame`, NEVER a snapshot — snapshots are settled ~90 frames
  past their frame), **closed-loop** (hold each absolute target ≤8 substeps until
  |q−cmd|<0.03 — this env's gains are softer than the rig's; open-loop cuts corners),
  with magnetic establish at the demo's closure command; grasp seeds start pre-closure
  (closure−20). Exclude synthetic bank entries (harvest/bridge) from demo seeding.
- **Frontier sampling + guaranteed cadence**: entries weighted (1−rate)+0.15, press
  rungs gated in offset order; solved feeder skills (grasp) get a forced slot every
  5th episode or the sampler starves them and everything downstream stalls.
- **Chain machinery** (long-horizon = the point): bridge harvest (each grasp success
  donates its end state as a press start entry, cap/scene, gate-exempt); chain
  episodes every 4th in consolidated scenes (goal one-hot switch at the subgoal;
  **the subgoal-success step is stored done=0** so terminal value bootstraps backward
  through the boundary; refresh hold_act + extend phase-2 budget at the switch);
  **chain seeds** re-planted every scene visit (demo replayed straight through both
  phases, goal flipping at closure — the only full-horizon demonstration format).
- Flight recorder: one JSON line/episode with per-mechanism telemetry (`assist_r`,
  `af`, `ag`, `lift`, phase) — design every mechanism observable BEFORE you need it.

### 2.7 Verification instruments
- Frozen bars pre-registered per phase; adjudicate honestly, escalate on tripwires
  (pre-commit the escalation and the threshold before the window opens).
- Probe ladder on the VLA side: commit-mass, point liveness, key causality, map-token
  liveness — frozen dead baselines; every retrain checkpoint must move them.
- **Liveness evals for every input you add** (zero it, measure): inputs the task does
  not DEMAND go unused — richness without demand is not usage.
- Film protocol: third-person camera (`og.sim.viewer_camera`; head cam =
  `zed_link:Camera`, wrists = `*_realsense_link:Camera`); **contact-sheet review of
  every clip before publishing** (sampled stills are blind to motion pathologies);
  keep film scripts in sync with wrapper fixes.

### 2.8 Flywheel export
Export to LeRobot for distillation: **strict-physics episodes only** (assist radius
0), success-terminated **including fumbling prefixes** (recovery arcs are half of
physical competence — don't over-sanitize), with privileged aux targets alongside
(object poses for object-motion prediction, true occupancy for L2). Distill with AWR
weighting + information-forcing objectives (conditional-PMI on spatial keys,
next-state/object-motion heads) under the probe ladder — plain BC on any corpus can
re-collapse.

## 3. Laws (each learned the hard way; do not relearn)

0. **Freeze a RANDOM-POLICY baseline per rung BEFORE any training.** A skill target
   whose random floor breaches the bar is not a skill target (task-62 chop: random
   0.355 — phase withdrawn). No bar means anything without its floor.
1. **Inputs the task doesn't demand go unused.** Add a liveness ablation the day you
   add an input.
2. **Validate futures, not states.** A restored state is only as good as what demo
   actions do FROM it.
3. **Run the replay-fidelity sweep FIRST; it decides the replay-vs-RL mix per task.**
   Segment replay from restored anchors under eval physics: where q holds, replay (and
   replay-under-DR) is a cheap, honest corpus generator; where it collapses (e.g. rig
   assists like radio's magnetic grasps), RL is the only honest source. Continuous
   full-episode replay drifts to q=0 — demos are anchors + locally-valid snippets,
   never policies (task-62 evidence, both directions).
4. **Rollout physics must equal eval physics**; bridge gaps with constraint
   curricula, never by softening the rollout env permanently.
5. **Never default metadata — audit from data.** Arms, holders, closure frames.
6. **Stale replay data fights relearning.** When the world changes, retire buffers
   collected under the old world.
7. **Refresh stationary commands at every role switch.**
8. **Success predicates define the data you export.** "Held" means still; sloppy
   predicates ship sloppy behavior into the VLA.
9. **Make every mechanism observable in the flight recorder before you need it.**
9b. **Guard your own edits**: a sanity_check script (duplicate-definition scan,
   obs_dim guard) runs after every wrapper/trainer patch — string-slice edits can
   silently duplicate blocks with stale copies winning (task-62 incident).
10. **Watch your own videos, as contact sheets.**

## 4. Ops (this box)

Single sim process per box (scale = more boxes). Isaac swallows SIGINT/TERM — kill
with TERM→verify with `ps`→KILL; bracket your pkill patterns AND never put the
process name in the same compound command. `OG_PLAYBACK_REAL_FREQS=1` on every
launcher (physics 120 Hz / control 30 Hz). Boot ~4 min; snapshot load ~1 s; ckpt
saves at scene boundaries — time restarts to them. HF token: `/root/.hf_token`,
file-based only. Backups: git commit → push GitHub (`origin`) + HF bundle
(`arif101/b26-code-backup`) at milestones. Instance 301 is quarantined: never train,
restore, or harvest from it. Held-out demos list lives in the trainer; respect it.

## 5. Bring-up checklist for a new task

1. Fetch raw demos + reference dataset rows for the task; build episode map.
2. Metalink/predicate labels for the task's goal predicate(s).
3. Arm audit from action channels → `arm_audit_<task>.json`.
4. Event frames per family → start bank → snapshot bank → **dynamic validation**
   (replay-fidelity gate) before any training.
5. Register the family predicates in the wrapper (the only new code most tasks need);
   set `target_name_sub` and metalink offset.
6. Smoke: 3 scenes × 25 episodes; verify recorder telemetry (assist fires, holds
   persist, seeds plant) before scaling to all scenes.
7. Pre-register bars + escalation tripwires; arm monitors on the recorder.
8. Chain scheduling from day one if the task composes skills.

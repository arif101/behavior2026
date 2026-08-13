# CONTACT SKILL SPEC v1 — delegated terminal-control policy (2026-08-12)

Provenance: RUN2_EVAL_REPORT.md (0/25 FAIL; commit-probe mass 0.00, diversity collapsed →
MODE ABSENT; selection-class levers support-dead). This is the funded main line: a small
LEARNED policy owning the final ~7cm, trained by sim RL from restore states — the only
method on the board that manufactures commit behavior instead of reweighting it.

SCOPE CORRECTION (2026-08-12): the radio terminal window is the ACQUIRE→MANIPULATE
composite (stage taxonomy: grasp closes ~f1110-1160, lift f1125+, then toggle), NOT a
bare press. The skill's scope = everything from the 7 cm shell to ToggledOn: grasp,
lift, toggle — learned as ONE behavior because the reward (ToggledOn) is terminal and
the splice clips demonstrate the composite. We do NOT hand-decompose it; RL discovers
the sequence. "Press skill" naming retired → COMMIT SKILL. Attempt budget raised
300→600 steps (must cover the demo's acquire+manipulate span; measure exact demo
segment lengths in day-0 and set budget = p95 span × 1.5).
Design discipline: ambitious-design-first (feedback 2026-07-15); learned, not scripted
(feedback 2026-06-16).

AMENDED 2026-08-13 (pre-launch — bars not yet frozen; operator takeover spec review +
restore-fidelity probe findings). Amendments marked [A-2026-08-13] inline: gate stage id +
τ pinned (§1), L2-geometry v1 decision (§2), start-state source pinned + restore validity
check (§4), holdout split rule + gate-fire interpretation precondition + curriculum-stall
kill clause (§5). Probe evidence: box_artifacts/contact_skill_probes/ (restore mm-accurate;
AG constraint does NOT survive og.sim.load_state; held radio slips ~1.2 cm/s).

> **OPEN DESIGN QUESTION [Arif/Claude 2026-08-12, for Suhas, MUST RESOLVE PRE-FREEZE]:**
> gate-stage pin (=2 MANIPULATE) vs skill scope (acquire→manipulate composite). Stage 2
> begins POST-lift in the label taxonomy, but the commit probe's 0.00 mass is measured at
> the PRE-lift band (f1100, gripper-close initiation) — i.e. acquire initiation is the
> missing mode. A stage-2-only gate can only fire after the VLA has grasped+lifted, which
> the campaign says it never does (gripper never closed at 2-3 cm). Proposals: gate on
> stage ∈ {1,2} (ACQUIRE-or-MANIPULATE), or geometry-primary (7 cm shell) with stage ∈
> {1,2} as confirmation. Also reconcile: §1 budget still reads ≤300 while the scope
> correction sets 600 (grasp+lift+toggle span) — one number must win, measured from demo
> segment lengths. Your gate-fire ≥20% precondition already covers the failure mode;
> this note is about not walking into it knowingly.

## 1. System contract (serving)
- Base VLA: **radio_run2@49999** (ours; frozen). Drives episodes at h=32 chunks.
- GATE (per-step, in wrapper): stage_head=terminal ∧ dist(EE, affordance_pt) < 0.07 m ∧
  approach-axis alignment < 20° ∧ aff_conf > τ. Fires → control handoff.
  [A-2026-08-13] "terminal" PINNED = stage class 2 (MANIPULATE) of the 0/1/2/3 vocabulary
  (add_stage_pixel_labels.py). NOT 3 (END): 3 is a post-success proxy (last 4% of frames)
  and would never fire pre-press. τ PINNED = 0.5 (Run-2 telemetry: conf_p50 ran 0.68-0.82
  on live points; 0.5 admits the working band with margin, rejects junk).
- SKILL: per-step control at 30 Hz, budget ≤300 steps/attempt.
- Exit: success-detected | budget-out → rewind 15 cm along approach normal → return
  control to VLA → re-approach; max 3 attempts/episode. Never downgrade mid-attempt.
- Success detection AT EVAL (no privileged predicates): dwell-complete (≥5 consecutive
  contact-consistent steps from gripper effort/velocity profile) ∨ stage-head advance.
  **OPEN ITEM #1 — validate detector offline against sim ToggledOn on training rollouts;
  required precision ≥0.9 before arm-C.**

## 2. Observation / action spaces
Obs (per step, all eval-legal):
- proprio qpos/qvel subset for active arm + torso + gripper, LAST 5 FRAMES stacked
- affordance point in EE frame (3) + confidence (1) — FROZEN at gate-fire (no
  re-targeting mid-attempt; refresh only on retry re-entry)
- L2 wrist-local geometry: pooled occupancy+staleness features from the 0.02 m cube
  (~64-dim pool; exact pooling in impl doc) — wrist depth path legal per 08-09
  adjudication if L2 pooling underperforms
  [A-2026-08-13] v1 DECISION: the 64-dim L2 slot is RESERVED but fed ZEROS both in prior
  data (not derivable from stored clips) and online — identical prior/online distribution,
  no mismatch. Enabling real L2 features later is a single-variable change on a reserved
  slot (no obs-shape change, no buffer rebuild).
Action: joint-velocity deltas, active arm (7) + torso (4) + gripper (1) = 12-dim,
tanh-squashed, same limits as serving contract.

## 3. Network
- Actor: MLP 3×512, LayerNorm, tanh head. Critic: ensemble of 5 MLPs 3×512 + LayerNorm,
  min-over-random-2 targets (REDQ-style). Total ≈2M params.
- Arm 2 (only if frame-stack insufficient): GRU-256 over proprio history.
- NO transformer: flat obs, short horizon, sample efficiency binds.

## 4. Training (RLPD)
- SAC base, symmetric sampling: 50% online replay / 50% prior data per batch.
- Prior data: splice clips (b1k_radio_corrective, HF) converted to skill tuples
  (obs from stored proprio+objpose+camposes; reward labeled from stored success flags).
  Converter: `box_scripts/convert_splices_to_skill_buffer.py` (to build).
- UTD 8 (tune 4–20); γ=0.98 (short horizon); entropy auto-tune.
- Reward: sparse ToggledOn (+1 terminal) + privileged shaping IN TRAINING ONLY, ablatable:
  −0.01·dist(EE,pt) per step, +0.1·contact-made, +dwell-counter progress. Report primary
  results at sparse-only arm too (shaping is a crutch to remove, not a dependency).
- Curriculum (reverse, over restore states): stage 0 = restores ≤5 steps pre-press;
  advance when rolling success ≥70% → 25 → 100 steps pre-press → the 7 cm gate boundary.
  Start-state sources: demo terminal states (train demos) + OUR campaign near-commit
  states (policy-visited distribution — primary) — **NEVER instance 301 restores.**
  [A-2026-08-13] "campaign near-commit states" PINNED: states harvested by running the
  frozen VLA from TRAIN-demo restores (the state-harvest pass), snapshotting in the 10 cm
  shell. EXPLICITLY EXCLUDED: any state from the Run-2 eval campaign — that campaign ran
  on instance 301 (public_test index 0), so its states are 301 states and using them
  violates the held-out rule even though they are "ours".
  [A-2026-08-13] RESTORE VALIDITY CHECK (required; probe finding): og.sim.load_state does
  NOT re-create assisted-grasp constraints — a held object slips ~1.2 cm/s on friction
  alone, and a held-closed gripper never re-engages AG. The env wrapper must, at every
  restore: re-establish AG for the holding arm via robot._establish_grasp (probe-verified
  2026-08-13: post-restore PhysX contact reporting is EMPTY, so the auto-AG path can never
  see a candidate; direct establish holds the object flat at ±0.1 mm), then verify
  AG engaged ∧ |Δ target-object z| < 1 mm over a 30-step window AFTER a 60-step settle
  (a ~3-7 mm attachment-snap transient in the first ~2 s is expected and acceptable —
  probe log probe_ag_fix). States failing the check are DISCARDED (collector
  outcome-filter pattern). This is training-infrastructure repair, not part of the
  learned-vs-scripted boundary.
- Domain randomization: target pose jitter (±2 cm, ±10°), initial EE offset (±3 cm),
  perturbation forces mid-attempt (recovery emerges), affordance-point noise ~ measured
  head error (σ≈2.6 cm ∧ conf-conditioned), physics (friction ±20%).
- Rig: A6000 sim box (sim + learner co-resident; learner is negligible GPU).

## 5. Pre-registered bars (FROZEN at first training launch)
- SIM GATE: ≥80% press success from the full 7 cm-boundary curriculum stage, randomized
  poses, sparse-reward eval mode, ≥200 eval episodes, held-out restore states.
  [A-2026-08-13] "held-out" PINNED: split BY SOURCE DEMO/EPISODE, never by state/tuple —
  states from one demo are near-duplicates; a state-level split leaks and inflates the bar.
  (The prior-buffer converter stores raw_episode_id per tuple to enforce this.)
- ARM-C (integrated, instance 301, n=25, default timeout, same serving stack + gate):
  conversion bands identical to RUN2_EVAL_PREREG for comparability —
  STRONG ≥5/25, WEAK 3–4, FAIL ≤2. Secondary: gate-fire rate, attempts/episode,
  skill-window success rate, rewind counts.
  [A-2026-08-13] INTERPRETATION PRECONDITION (pre-registered): if the gate fires in <20%
  of arm-C episodes, the conversion verdict is diagnostic of the GATE (stage head /
  affordance conf at eval), NOT of the skill — it triggers gate diagnosis, not the kill
  path. Conversion bands stand as a skill verdict only when gate-fire ≥ 20%. (Motivation:
  stage-head held-out accuracy is still PENDING from Run-2 secondaries; an unmeasured
  gate must not silently convert into a skill kill.)
- Kill criterion: sim gate unmet after 3 curriculum-stage-0 redesigns → STOP, escalate
  to data-side rethink (the skill premise itself would then be falsified).
  [A-2026-08-13] ADDED CLAUSE: independent of stage-0, if the curriculum advances no
  stage for 3 consecutive training days, same STOP + escalation (closes the gap where
  stage-0 learns but a middle stage stalls forever with no stop rule).

## 6. The flywheel (weight-level VLA integration path)
Skill successes = on-policy commit demos → converted to LeRobot episodes → Run-3 BC mix.
Verification: rerun the EXACT commit probe (demo 10, f1100) on the Run-3 checkpoint;
success = mass moves off 0.00. Skill's critic doubles as candidate chunk-critic.

## 7. Build plan (~1.5 weeks to arm-C verdict)
1. d0–d1: this spec review; splice→buffer converter; restore-state bank extraction
   (campaign near-commit states + demo terminals; verify restore fidelity via
   probe_restore_fidelity.py).
2. d1–d3: commit-window env wrapper (restore → step → reward → reset) + RLPD loop
   (sbx/jaxrl-class clean implementation) + smoke (stage-0 learns ≥50% in ≤2 h).
3. d3–d7: curriculum training + iterations; daily success-curve reports.
4. d7–d8: success-detector validation (Open Item #1); serving integration (gate in
   wrapper, skill server-side); integration smoke.
5. d8–d10: pre-registered arm-C campaign + report.

## 8. Risks
- Restore-fidelity drift at contact-rich states (probe_restore_fidelity.py first).
- Detector precision <0.9 → arm-C blocked on Open Item #1 (fallback: dwell-only
  detector, accept conservatism).
- Generalization beyond press family deferred to v2 (grasp/pull/turn share recipe,
  new reward defs per family).
- OG throughput lower than assumed → cut UTD, extend d3–d7; kill criterion unchanged.

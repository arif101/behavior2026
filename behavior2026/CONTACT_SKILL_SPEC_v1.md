# CONTACT SKILL SPEC v1 — delegated terminal-control policy (2026-08-12)

Provenance: RUN2_EVAL_REPORT.md (0/25 FAIL; commit-probe mass 0.00, diversity collapsed →
MODE ABSENT; selection-class levers support-dead). This is the funded main line: a small
LEARNED policy owning the final ~7cm, trained by sim RL from restore states — the only
method on the board that manufactures commit behavior instead of reweighting it.
Design discipline: ambitious-design-first (feedback 2026-07-15); learned, not scripted
(feedback 2026-06-16).

## 1. System contract (serving)
- Base VLA: **radio_run2@49999** (ours; frozen). Drives episodes at h=32 chunks.
- GATE (per-step, in wrapper): stage_head=terminal ∧ dist(EE, affordance_pt) < 0.07 m ∧
  approach-axis alignment < 20° ∧ aff_conf > τ. Fires → control handoff.
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
- Domain randomization: target pose jitter (±2 cm, ±10°), initial EE offset (±3 cm),
  perturbation forces mid-attempt (recovery emerges), affordance-point noise ~ measured
  head error (σ≈2.6 cm ∧ conf-conditioned), physics (friction ±20%).
- Rig: A6000 sim box (sim + learner co-resident; learner is negligible GPU).

## 5. Pre-registered bars (FROZEN at first training launch)
- SIM GATE: ≥80% press success from the full 7 cm-boundary curriculum stage, randomized
  poses, sparse-reward eval mode, ≥200 eval episodes, held-out restore states.
- ARM-C (integrated, instance 301, n=25, default timeout, same serving stack + gate):
  conversion bands identical to RUN2_EVAL_PREREG for comparability —
  STRONG ≥5/25, WEAK 3–4, FAIL ≤2. Secondary: gate-fire rate, attempts/episode,
  skill-window success rate, rewind counts.
- Kill criterion: sim gate unmet after 3 curriculum-stage-0 redesigns → STOP, escalate
  to data-side rethink (the skill premise itself would then be falsified).

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

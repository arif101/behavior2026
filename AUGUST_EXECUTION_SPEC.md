# August Execution Spec — BEHAVIOR-2026 Rebuild

**Date:** 2026-07-23 · **Deadline:** Oct 16 · **Basis:** `GATE_REPORT_G3.md`
**One-line goal:** raise the policy above the terminal-primitive competence floor, then make
the surrounding system general and robust.

---

## 0. The two floor-raisers (do these first — nothing else scores without them)

1. **Online RL on the flow policy** — make the policy competent at contact.
2. **A reliable terminal commit primitive** — a grasp/press that actually fires, used first as
   an RL reward signal and a distillation teacher.

Everything below (Opus-in-loop, metacognition, representation switch) makes the system
*general and robust*; these two make it *work at all*. Sequence accordingly.

---

## 1. Diagnosis → plan mapping

| Measured failure | Fix | Section |
|---|---|---|
| Data starvation (30 vs 100–500 eps/task) | scale + diversity data mixture | §3 |
| Terminal actuation (reaches 3–8mm, no fire) | reliable commit primitive → distill | §4 |
| No closed-loop competence (0% success) | online RL with staged rewards | §5 |
| Base orientation (close but wrong side) | Opus deliberative tier in the loop | §6 |
| Genuine reach limits | whole-body reach + honest "unreachable → re-plan" | §4, §6 |
| Near-contact precision decay | representation switch (relative → contact/classification) | §7 |
| Covariate shift (train demo → own drifted states) | SAM optimizer | §3 |

---

## 2. Locked (validated — do not re-litigate)

- **Spine:** π0.5 flow VLA + zero-init AdaLN 3D-point conditioning + small heads on frozen
  DINOv3. Confirmed active.
- **Grounding head** (~1cm, v0.6 recipe). Precision is not the blocker.
- **The interface** (relative EE→target point). Keep; refine near contact only (§7).

---

## 3. Data mixture (diversity-first) + optimizer

**Principle (research, r=0.96–0.99):** generalization scales with environment/object
*diversity*, not raw demo depth. π0.5's own ablation shows the broad co-training mixture is
load-bearing.

- **Scale:** 200 eps/task (we used 30) across **many** tasks (we used 4) — the full corpus,
  not 4 tasks drilled deep. Reuse the assembler; extend the state-decoder path for 40× cheap
  conditioning data where rendering isn't needed.
- **Oversample contact windows:** re-weight the loss toward the grasp/press frames (a sliver
  of each demo — this is why 30 demos' contact signal was too weak). Cheap, high-leverage.
- **Co-training replay:** keep a 2–20% slice of broad/pretraining-style data mixed in
  (prevents narrow-FT degradation; note forgetting was *not* our disease, but the mixture is
  still load-bearing).
- **Optimizer:** swap in **SAM** (Sharpness-Aware Minimization) for action FT — measured
  +60–217% relative on covariate-shift-limited action FT. **A/B it** (§9).

## 4. Terminal commit primitive → distillation

- **Build** a reliable contact-triggered commit (grasp/press): the validated Jacobian servo +
  fix the actuation bug (the finger reaches but the sim's ToggledOn / grasp condition doesn't
  fire — needs sustained real contact along the right approach, and whole-body/torso reach for
  far targets). Director code + fix on HF `director_arm/`.
- **Distill:** run the primitive across scenes → collect successful grasp/press trajectories →
  add to training (corrective RFT / DAgger). The primitive is a **data generator**; the policy
  learns to commit and generalizes to novel objects the script wasn't written for.
- **Data format (PDP recipe, verified):** pair **noisy state ↔ clean corrective action**
  (100% vs 3.36% recovery in the cited ablation). Mandatory.
- **Backstop:** an honest director commit macro (grounding-fed, rules-legal) remains available
  at eval if the learned commit underperforms on a task.

## 5. Online RL on the flow policy (the top lever)

- **Method:** learnable exploration noise in the flow sampler (ReinFlow/SCRIPT-style) + PPO,
  in parallel OmniGibson envs. Cited precedent: 0%→40% real; LIBERO 77→98.
- **Reward = staged/dense (this is what makes it work at 0%):**
  - Sources: **BDDL intermediate predicates** (near/holding/on — each satisfied predicate = a
    reward step) **+ the stage head's progress output** (cofounder-owned; now doubles as the
    reward spec).
  - Requirements on the stage signal: **dense, monotone** (rises toward done), **per-subtask**.
    A 70%-accurate smooth signal beats a 90%-accurate non-monotone one for RL.
  - Shape example (radio): +0.2 base@table, +0.3 gripper<10cm, +0.3 contact, +0.2 toggle.
- **Regularization:** KL-anchor to the BC policy (proximal anchor — mandatory; InterPrior's
  prior-preservation env split is a validated alternative to A/B).
- **Then re-enable AWR** as a *refinement* pass once rollouts contain a mix of success/failure
  (it was demoted only because it can't bootstrap from 0%).

## 6. Deliberative Opus tier — into the runtime loop

- **Promote from offline probe to in-loop.** Opus owns strategy: base placement / approach
  side, object sequencing, recovery — fed by the perception heads (honest scene state), output
  consumed by the reflex tier as executable targets. Proven this week: correct edge selection
  from geometry alone.
- **Cadence:** event-driven (episode start + on metacognition trigger), ~10–30 calls/ep with
  strict timeout → programmed fallback. Not per-tick.
- **Metacognition trigger:** compose the sensors we already have (stall + grounding confidence
  + stage progress) into "my strategy is failing → re-plan," firing the deliberative tier
  instead of a fixed macro. Real metacognitive recovery.

## 7. Conditioning refinement (near contact)

- Add a **representation switch**: stop conditioning the gripper-close on a shrinking relative
  displacement (ill-conditioned toward zero — the 2cm cliff in the noise sweep). Trigger the
  close on a **contact / "in-grasp" classification** instead.
- Support the switch with **belief-head temporal fusion** (Kalman over the approach) +
  **foveated close-range crops** to hold ~1cm under the closed-loop viewpoint distribution.

## 8. Build order (phased)

- **Phase A (weeks 1–2): raise the floor.** Data assembly (diversity + oversampling) → BC
  retrain (+SAM A/B) → commit primitive built + distilled → **first nonzero honest success**.
- **Phase B (weeks 2–4): online RL.** Staged rewards wired (BDDL + stage head) → RL loop on
  OmniGibson fleet → climb from nonzero toward competitive; AWR refinement pass.
- **Phase C (weeks 4–6): system.** Opus-in-loop + metacognition trigger + honest director;
  representation switch; grounding v0.6 scaled to full task set.
- **Phase D (weeks 6+): harden + submit.** Full-matrix eval, latent SE(3) probe (mechanistic
  appendix), M4 submission package (Docker serve, 24GB fit), leaderboard dry-run.

## 9. A/B schedule (what to isolate)

| Test | Question | Cheap? |
|---|---|---|
| SAM vs AdamW | +60–217% claim hold on our setup? | yes (optimizer swap) |
| Contact-oversampling on/off | does re-weighting contact frames lift grasp? | yes |
| Diversity (many tasks) vs depth (few×deep) | power-law confirmed for us? | medium |
| RL staged-reward vs sparse | does shaping bootstrap off 0%? | core |
| KL-anchor vs prior-preservation split | which stabilizes flow RL? | medium |
| Learned commit vs honest director macro | ship which? | eval-time |

## 10. Hardware & risk

- **Compute:** RL rollout fleet = **RT-core GPUs** (4090/L40S/RTX-PRO class; A100 *cannot* run
  OmniGibson) + an H100-class trainer. This is the schedule risk — provision early.
  Acceptance gate (banked playbook): `proc-gpus == dev-nodes`, then `vulkaninfo` shows the GPU;
  missing-GLVND is a 30s fix; PI-futex needs the `nopi.so` shim; Isaac-on-Blackwell works.
- **Risks:** (1) RL sample efficiency on a 3B flow policy at scale (mitigate: staged rewards +
  KL anchor + start from a competent BC base, not scratch); (2) sim compute for the RL fleet;
  (3) the commit primitive's actuation bug must be genuinely fixed before it can teach.
- **Floor if RL underdelivers:** BC-scale + distilled commit + honest director is the
  lower-risk fallback and still a real, self-contained submission.

---

## 11. Cofounder handoff (stage head)

The stage head is now on the critical path as the **RL reward spec**, not just a director
input. Re-target the Jul 25 labels review to also check **reward-usability**: dense, monotone,
per-subtask progress, validated on **policy-rollout (closed-loop) states**, not only human
demo frames. Continue scaling to more tasks (aligns with diversity-first). Keep training.

# BEHAVIOR-2026 PHASE-2 PLAN — FINAL (Synthesis Chair, 2026-07-07)

**Stance:** Execution-realist spine (Plan C), carrying Plan A's metric-mechanics and triage weapons and Plan B's causal-evidence experimental design. One human, ~7–8 real improve-measure cycles, every gate power-specced, every bet with a dated kill and a paper-floor demotion path. Target: hidden-set q 0.25–0.31 (path: broad base ~0.15 + director/parking/governor +0.03–0.05 + data engine +0.04–0.08 + consolidation +0.01–0.03); winning bar 0.35 acknowledged as reach — our committed floor is top-5-competitive plus a publishable contribution that survives even double-negative gate outcomes.

---

## 1. HOW THE PLAN SURVIVES THE THREE CRITIQUES' FATAL FLAWS

| Flaw (source) | Resolution in this plan |
|---|---|
| 25-adapter bank = winner's curse + task-ID memorization at scale (Crit-A #1, #6) | Adapter bank KILLED. Routing survives only as an end-stage 3–6 cluster-specialist study: dev-only selection, ≥+0.05 margin over 20 paired instances, shrinkage toward the global base, default-to-base, public-vs-dev divergence as an uncontaminated canary (selection never touches public 0-9). |
| Eval economics mispriced 2–4x; submissions unbudgeted (Crit-A #2, Crit-B #2, Crit-C) | Sim priced at Isaac-capable real-VM rates ($1.2–1.9/h, vulkaninfo acceptance gate per our playbook); every leaderboard submission itemized (~135–250 GPU-h each, 3 planned); fixed eval currencies (smoke 10 / sprint 120 / full-dev 400 rollouts); fleet = 2 always-on + burst-to-4 real VMs with orchestration built in W5, quota/reservation actions in W1–W2. |
| Corrective-DAgger rests on unvalidated mid-episode state restore (Crit-A #3, Crit-C #3) | W1–W2 restore spike is a BLOCKING gate at G2.5 (Jul 17): load HDF5 state → settle → step → verify observables/contacts/renders (stale-obs bug class explicitly tested). No corrective spend commits before it passes. Fallback if it fails: live-rollout teacher-takeover DAgger only. |
| One-human schedule fiction (Crit-A #4, Crit-B #4, Crit-C) | ≤2 major deliverables per week; explicit capacity drop-order (drop first: probe studies → breadth expansion → metacog dispatch → routing study → corrective R2; never drop: factory, director, submission backstops); G3 moved Jul 24→Jul 31; first submission Aug 21 with Aug 28 submit-whatever-exists backstop; W15 untouchable. |
| Grounding Δq quoted at oracle value; no accuracy gate before the big train (Crit-A #5, Crit-C #5) | Honest grounding priced at ≤½ oracle lift in all projections; portfolio-wide OFFLINE pointer benchmark (held-out demo frames, all tasks, zero rollouts) is a blocking pre-train gate Aug 4; honest+oracle dual modes maintained forever; per-task oracle-at-train-only demotion for tasks failing the benchmark. |
| No base-competence gate before G3; all-arms-zero risk (Crit-B #1) | G2.5 gate (Jul 17): any nonzero q or nonzero flipped literals on one top-feasibility single-task ckpt before any multi-arm spend. G3 kill criteria explicitly cover "all arms ~0" (→ debug base competence, G3 slides 1 wk, breadth pre-cut absorbs it). |
| Director load-bearing but unscheduled (Crit-B #3) | Director is workstream WS2 from W1: BDDL parser, max-over-goal-options enumerator, per-stage target namer, inference-time stage estimator. Circular dependency broken: director v0 runs on predicate estimates from decoded proprio+grounding (coarse), stage head refines later. |
| Predicate labels ≠ week-1 CPU artifact (Crit-B #4) | Scoped: KINEMATIC predicates only first (ontop/inside/nextto from poses = 79% of literal mass); non-kinematic via bounded sim-replay budget or excluded from AWR v1; Larchenko timestamp-uniform stages as proven one-day fallback. |
| 720p committed blind (Crit-B #5) | 1-day feasibility spike (train memory, vision-token count ×3 cams, 24GB serving latency with grounding stack) at G2.5, BEFORE any G3 arm trains; G3 arms run at the same resolution the G4 recipe will use. |
| Held-out tasks donate hidden points (Crit-B #6) | Hold-outs cut 8→6, transfer table measured at G4+W11, tasks FOLDED BACK into the W12 refresh train before freeze. |
| Train-narrow portfolio math broken; no target score (Crit-C #1) | TRAIN BROAD, ITERATE NARROW: base trains on all ~90 feasible tasks' demos (incl. 24 new-scene tasks — fixes zero-shot-harvest hole); iteration/eval portfolio = 40 value tasks. Target score stated above. |
| Gate thresholds below noise (Crit-C #2) | Every gate has a pre-registered power spec (below). G3 demands large effects (swap-following ≥60%, n≥50; paired Δq ≥+10pp); corrective gate = ≥+0.03 paired full-dev vs MATCHED control, not +0.01 sprint. |
| Single-shot base-train assumption (Crit-C #4) | Explicit 600 H100-h second-base-train reserve line; G4 slip plan pre-cuts breadth (WS8), never the data engine (WS6). |
| Contribution double-jeopardy, no floor (Crit-C #6) | Pre-committed paper floor: rigorous controlled negative — honest-vs-oracle grounding gap at 720p household scale + causal swap-test methodology + corrective-vs-matched-control table. Experiment logs collected AT gates as blocking deliverables, not on "idle cycles." |

**Best ideas kept (provenance):** 623-D decoder as W1 keystone (all); offline BDDL scorer + 100-task value table (A); parking/never-unsatisfy/cheapest-option/no-progress-retry director (A); anti-fall governor with fall-rate AND completion-time A/B (A+C-fix); staged eval currencies (A/C); predicate-transition stages in q-units (all); depth-for-grounding-only, coarse-to-fine 2D pointer (all); RFT-first then corrective with matched-budget control arm (B, non-negotiable); teacher gate ≥60%/50 real failure states + retreat-reapproach as never-empty fallback (B); dead-tail zero-spend (all); good-vs-fatal memorization split (A, at defensible scale); point token over AdaLN for addressability (all); split hygiene with decontaminated calibration — metacog calibrates on self-generated privileged-randomized instances, never dev 10-19 (B+fix); two Docker rehearsals (C); dual-use rollout factory (C); park-safely default policy for untrained tasks (Crit-B); camera/FK label stack, grasp-phase detector, norm-stats job, submission dry-run, fall/timeout auto-detection as named work items (Crit-C); W15 untouchable buffer, spend tripwires, pre-designed fallbacks (C).

---

## 2. DESIGN DECISIONS a–g — RESOLVED

**(a) Point insertion: dedicated grounded-point prefix token(s)** — source-object + destination-container per active literal, Fourier-encoded 3D in robot-base frame + stage-target flag; re-ground at every receding-horizon replan (~1–2s), force re-ground on director stage transitions, freeze during gripper-close (grasp-phase detector = gripper command + proprio). AdaLN = bounded G3 ablation arm on 3 tasks only.
*Rationale:* a token is an addressable, substitutable locus — swap test, J-lens saliency, and our existing prefix-KV probe infra all come free; minimal diff to pi0.5.
*Dissent recorded:* AdaLN may condition more strongly (Plan B); if it wins by >+10pp at G3 we ship both and report the comparison as a finding.

**(b) Stage supervision: BDDL-predicate-transition auto-stages from the 623-D decoder, kinematic predicates first** — stage = index into per-task flip sequence + carried-object flag; Larchenko timestamp-uniform buckets as the proven fallback if the decoder or predicate evaluators slip past W2.
*Rationale:* stages in q-units align the stage head, AWR advantage, metacog conditioning, and director sequencing under ONE label source; free from public sim states; kills the missing-annotation dependency.
*Dissent:* timestamp stages are simpler and 2025-proven; kept as fallback and ablation baseline, not primary.

**(c) Grounding head: learned coarse-to-fine 2D heatmap pointer (frozen DINOv2-class features, full-frame ~256 → crop refine, target-conditioned on the director-named BDDL object) + depth unprojection to base-frame 3D.** Labels free from privileged poses projected via camera intrinsics/extrinsics + head/torso FK (explicit W2 work item). Honest AND oracle modes maintained forever; portfolio-wide offline benchmark is a blocking pre-train gate. No direct 3D regression; no depth in the policy trunk.
*Rationale:* matches our validated LIBERO recipe, visually debuggable frame-by-frame (diagnosis-speed moat), synthesizes Comet's pair of findings (resolution doubles success → crop-refine; naive depth hurts → depth stays in grounding); the measured honest-vs-oracle gap is a headline paper number — the exact punt in 2606.27663.
*Dissent:* direct 3D regression avoids depth-sensor noise; rejected for debuggability and label simplicity.

**(d) Recovery recipe order: (1) zero-training serving governor (W1–2), validated with fall-rate AND completion-time A/B and re-validated on the properly-trained G4 base before tuning locks; (2) scripted retreat-reapproach (object-agnostic, near-guaranteed, the never-empty corrective class); (3) brace from detected tip-onset; (4) regrasp-after-drop last.** All learned recoveries gated on the state-restore spike + ≥60%-on-50-real-failure-states teacher gate. Re-rank by the G4 failure census.
*Rationale:* the governor may kill most fall mass for free, and the fall evidence is ONE undertrained single-task ckpt — spend training budget where the census says the mass is; retreat-reapproach insures the corrective dataset can never be empty.
*Dissent:* Plan B ordered brace first (fall = observed class #1); recorded — the census decides at W7.

**(e) Portfolio: train broad, iterate narrow.** Base trains on all ~90 feasible tasks' demos (excludes 5 particle tasks — permanent zero spend — and ~5 infeasible); iteration/eval portfolio = 40 tasks (11 single-literal quick wins + 29 best known-carryover pick-place); 6 carryover tasks held out of training for the transfer table, folded back into the W12 refresh before freeze.
*Rationale:* hidden rank correlates with nonzero-q COUNT; demos for all tasks are free; 40-task training caps the final score below even Larchenko's 0.26 (Crit-C math); iterating on 40 respects the 7–8-cycle budget.
*Dissent:* Plan A's 40-task train is cheaper and deeper per task; rejected on portfolio arithmetic.

**(f) Conditioning: language prompt + grounded-point token; NO per-task ID embeddings, ever.** Task-ID exists only as a G3 baseline arm. Gated fallback for cross-task interference: 8–16 skill-FAMILY embeddings (never per-task). End-stage 3–6 cluster-specialist routing allowed under the winner's-curse guardrails in §1.
*Rationale:* splits good memorization (task identity, legal, carried by language + routing at coarse granularity) from fatal memorization (positions — our pi0.5 swap-0.17 finding — carried by the point); preserves held-out-task transfer, the user-locked anti-memorization pillar, and the paper claim.
*Dissent:* Plan A's full adapter bank attacks the 0.345→0.611 gap directly and is legal; rejected — that oracle number is a selection-overfit upper bound and the bank contradicts the user's rejected-warm-start principle.

**(g) Data engine: AWR success-replay round 0 FIRST (W7, monetizes already-paid factory rollouts; expect a fraction of NVIDIA's +0.12 given our lower base — priced accordingly), then corrective-DAgger R1 (W9) WITH a matched-budget success-replay control arm, then R2 or RFT-2 per the Sep 15 verdict.** Prerequisites: state-restore spike (G2.5) + teacher gates (Sep 1). Residual RL: banned permanently (off-manifold evidence).
*Rationale:* replay stabilizes the base while teacher machinery matures, its rollouts ARE the failure harvest, and the ordering makes the paper's central control arm nearly free; the control arm is what stops reviewers reducing claim 2 to "you added data."
*Dissent:* Plan A's 3-round 60/40 mix without control; rejected — uncontrolled, and a slipped corrective engine would take the whole axis down.

---

## 3. SCHEDULE Jul 7 → Oct 16 (gates bold; GPU budgets per week)

- **W1 Jul 7–13:** 623-D decoder written + validated vs known geometry (CRITICAL PATH); mid-episode state-restore spike started; box up by script; governor v0; value table v0 from 2025 per-task scores; sim-VM quota + H100 reservation actions. (~40 dev GPU-h)
- **W2 Jul 14–20:** **G2.5 GATE Jul 17:** (i) restore-spike verdict [gates WS6], (ii) first nonzero-q/nonzero-flip single-task ckpt, (iii) 720p/24GB feasibility spike verdict [fixes G3/G4 resolution]. Camera+FK label stack; point-label pipeline; director v0 (parser, goal-option enumerator, target namer); data staging starts (~2.4TB incl. broad portfolio). (~60 GPU-h)
- **W3 Jul 21–27:** pointer head trained; offline pointer benchmark v0; G3 arms launched (4 arms × 4–5 tasks, ~120 H100-h); kinematic predicate-stage labels; governor fall-rate + completion-time A/B. (~120 H100-h + 60 sim-h)
- **W4 Jul 28–Aug 3:** G3 eval on burst fleet (paired held-out instances, ~500 rollouts + 50-relocation swap test + conflict test); **G3 DECISION Jul 31** (power spec: swap-following ≥60% n≥50; paired Δq ≥+10pp vs language-only); broad norm-stats job; staging complete. (~200 sim-h)
- **W5 Aug 4–10:** **PRE-TRAIN GATE Aug 4:** portfolio-wide offline pointer benchmark (e.g. ≥70% frames in-mask, <7cm 3D on held-out frames; failing tasks demoted to oracle-at-train-only). Broad base train LAUNCH (~90 tasks, ~1,150 H100-h, resolution per spike); factory build + fall/timeout auto-detection. (~1,150 H100-h begins)
- **W6 Aug 11–17:** train continues; per-ckpt sprint evals; **Docker 24GB rehearsal #1 Aug 15 (blocking)**; submission-mechanics dry-run; factory always-on (2 boxes). (~150 sim-h)
- **W7 Aug 18–24:** **G4 GATE:** full-dev eval on 40-task iteration portfolio (≥0.08 mean dev q, else debug loop pre-cutting breadth not corrective); **FIRST SUBMISSION Aug 21** (backstop rule Aug 28); failure census → recovery re-rank; RFT round-0 launch (~150 H100-h). (~200 sim-h)
- **W8 Aug 25–31:** RFT-0 eval; retreat-reapproach + brace teachers built; **teacher gate Sep 1** (≥60% on 50 harvested failure states each); held-out transfer table measured. (~200 sim-h + 50 H100-h)
- **W9 Sep 1–7:** corrective collection at scale (~550 sim-h) if gates passed; DAgger R1 train (300 H100-h) + matched control (150 H100-h).
- **W10 Sep 8–14:** R1 two-arm paired full-dev eval; **SUBMISSION #2 Sep 12**; metacog head trained on factory logs (conformal, calibrated on self-generated privileged-randomized instances). (~300 sim-h)
- **W11 Sep 15–21:** **CORRECTIVE KILL CHECK Sep 15** (≥+0.03 paired full-dev vs matched control); DAgger R2 OR RFT-2 + breadth per verdict (~300 H100-h); dispatch A/B on fall-prone tasks; hold-out fold-back staged. (~200 sim-h)
- **W12 Sep 22–28:** **METACOG GATE Sep 22** (≥30% fall reduction vs governor-only, else monitoring-only/paper-only); fold-back refresh train (~250 H100-h); **SUBMISSION #3 Sep 26**; **spend tripwire check Sep 26**. (~200 sim-h)
- **W13 Sep 29–Oct 5:** **architecture freeze Sep 29**; cluster-specialist routing study (3–6 ckpts, dev-only, ≥+0.05 margin, shrinkage to base); **Docker rehearsal #2 Oct 1**; per-task q<0.05 freeze; paper-critical eval reruns on frozen candidates. (~300 sim-h)
- **W14 Oct 6–12:** **FREEZE Oct 10:** final ckpt selection via full-dev; final Docker (grounding head + director + governor + park-safely default for untrained tasks); double dry-run; full HF artifact backup. (~150 sim-h)
- **W15 Oct 13–16:** untouchable buffer — submit/verify only; no model changes; paper writing continues past deadline.

**Budget (honest pricing):** H100 ~3,020 h @ $2.5 = $7.5k (G3 120 / base 1,150 / SECOND-TRAIN RESERVE 600 / RFT-0 150 / R1 300 + control 150 / R2-or-RFT2 300 / fold-back refresh 250). Isaac-capable sim VMs ~3,380 h @ ~$1.4 = $4.7k (gates ~280 / factory ~1,700 / corrective collection ~550 / 3 submissions ~500 / freeze ~350). Storage/egress/CPU $0.7k. **Planned $12.9k; contingency to $15k; hard cap $16k.**

**Descope ladder (dated):**
1. **Aug 28:** no live submission → cut iteration portfolio to 25 tasks, freeze feature work, submit best existing ckpt; nothing proceeds until a public number exists.
2. **Sep 15:** corrective <+0.03 vs control → stop teacher development; all H100 to RFT-2 + breadth; paper reports the controlled negative.
3. **Sep 22:** metacog <30% fall reduction → governor-only ships; metacog = monitoring/paper-only.
4. **Sep 26:** spend >$10k with public <0.15 → minimal tier: one sim box, portfolio frozen, director/parking/governor polish only (these cannot overfit hidden instances).
5. **Oct 1:** Docker rehearsal #2 fails → descope inference (pointer res, optional heads) before any other work.
6. **Permanent:** 5 particle tasks never attempted; residual RL never; second base train comes out of the reserve line, paid by cutting WS8 breadth, never WS6.

---

## 4. RESEARCH CONTRIBUTION

**Claim:** *On BEHAVIOR-2026, honestly-grounded relative-3D-point conditioning (learned 2D pointer + depth unprojection, labels free from privileged sim state, no task-identity embeddings) causally carries position generalization that task-ID conditioning structurally cannot — and physics-executed corrective recovery data, distilled via predicate-delta-weighted DAgger, is a distinct data axis that beats matched-budget success replay; we quantify the honest-vs-oracle grounding gap at 720p household scale that prior work (2606.27663) assumed away.*

**Minimum defensible experiment set (all gate-blocking deliverables, logs banked at the gate):**
1. G3 four-arm comparison (task-ID / language-only / language+point-honest / language+point-oracle), paired held-out instances, pre-registered power spec.
2. Causal conditioning swap test (n≥50 relocations) + language-vs-point conflict test.
3. Honest-vs-oracle grounding gap: portfolio-wide offline pointer benchmark + on-policy paired delta.
4. Held-out-task transfer table (6 tasks; the row task-ID cannot fill), then fold-back.
5. Corrective-DAgger vs matched-budget success-replay control, paired full-dev.
6. Conformal metacog calibration/coverage + net-positive dispatch A/B (falls).
7. Layer-wise prefix-KV geometric probes (object 6-DoF decodability) across conditioning arms and pre/post corrective training.

**Floor (pre-committed):** if both headline gates fire negative, the paper is the controlled negative — the measured honest-vs-oracle gap plus the swap-test causality methodology plus the corrective-vs-control table. Diagnostic negatives are bankable; this satisfies "publishable independent of placement."

---

## 5. LESSONS-LEARNED REGISTER (for the write-up)

1. Hidden rank correlates with BREADTH (nonzero-q on 43/42/15 tasks for 2025 top-3) — many-nonzero beats few-perfect; a missing rollout is 0.
2. Final-state-only scoring with achieved-then-lost=0 and init-true denominator inflation (34/100 tasks) makes end-state parking, never-unsatisfy, and cheapest-goal-option pure free q that no training run matches per dollar.
3. ~79% of goal literals are kinematic placement on ~60 tasks; 5 particle tasks scored 0 for everyone — one grounded pick-place motor carries the metric mass and the dead tail gets zero GPU-minutes forever.
4. Falls are the observed failure-class #1 (our G2 video: task-directed approach, then tip-over despite the institutionalized 250kg base) — but the evidence is one undertrained checkpoint; measure the census before locking recovery priorities.
5. The 0.611 best-of-checkpoints "oracle" is a selection-overfit upper bound, not expected gain: selection on 20 noisy instances (per-instance q std ~0.4) is a winner's-curse machine — route away from a global base only on wide margins with dev-only selection.
6. Oracle grounding ≠ honest grounding: the +32-46pp external result used oracle points, and our own recognition-wall history (honest bind 0.40 vs oracle 0.97) says price honest lift at ≤half and measure the gap as the finding.
7. Eval is the binding currency: ~20 min/rollout, full self-eval 350–450 GPU-h, and Isaac requires real Vulkan VMs (RunPod-class GPUs cannot run it — verified) — so eval is sold in fixed currencies and every gate has a pre-registered power spec, or its kill criteria fire on noise.
8. Comet's triple: 720p doubled success on some tasks (spike feasibility before committing), naive depth-into-policy HURT (depth for grounding only), temporal ensembling scored 0 (receding horizon only).
9. Mid-episode sim-state restore is an unvalidated load-bearing primitive — our stale-obs-after-set_state history produced weeks of silent 0.00 pipelines; spike it before committing a single corrective GPU-hour.
10. Good vs fatal memorization: task identity is legal and given at eval; POSITION memorization is fatal (pi0.5 swap 0.17) — carry positions with a grounded point token, carry identity with language, and never with per-task embeddings.
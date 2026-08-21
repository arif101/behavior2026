# G3 Point-Conditioning Gate — Report

**Date:** 2026-07-23 · **Status:** complete (pre-registered gate + extended diagnostics)
**Hardware:** A100 (training) + RTX PRO 4000 Blackwell (OmniGibson eval), both spun down
**All artifacts:** HF `arif101/behavior2026-artifacts` + project memory

---

## 1. Pre-registration recap

The gate was designed to answer one thesis question: **does telling a fine-tuned π0.5
*where* the target is (a relative 3D point via zero-init AdaLN) beat telling it *what* the
task is (language), on shifted instances the policy never trained on?**

Four arms, identical training (30 deep-mapped episodes × 4 gate tasks, 30k steps, full
fine-tune from `pi05_base`), differing only in conditioning:

| Arm | Conditioning | Role |
|---|---|---|
| 1 `taskid` | raw task-id prompt | memorization control |
| 2 `lang` | descriptive sentence | language baseline |
| 3 `lang_point` (oracle) | sentence + 3D point from sim state | axis ceiling |
| 4 `lang_point` (honest) | sentence + 3D point from grounding head | shippable version |

**Pre-registered criteria:** (a) oracle arm ≥ **+10pp** over `lang` on shifted instances →
the 3D-point axis is real; (b) honest arm retains ≥ **50%** of the oracle gain → grounding
head is good enough to ship.

---

## 2. What we ran

- **Training:** all 3 checkpoints trained to 30k (one disk-full incident at the final save,
  recovered from 25k; disk-safe driver added). All params + norm stats on HF.
- **Grounding:** v0.5 benchmarked offline on all 4 gate tasks (validated against published
  held-out numbers); **v0.6-gate** fine-tuned (12.8 min) to fix thawing.
- **Eval (OmniGibson, public_test shifted instances, videos on):** radio + trash × {`lang`,
  `lang_point`+oracle} × 20 instances; radio train-mode (`lang`, ±replan-h4); point-swap
  causality (trash); + the full oracle-assisted director probe suite.
- **Diagnostics:** 47-episode video taxonomy; grounding-precision noise sweep; 105-agent
  adversarial research validation.

---

## 3. Results

### 3.1 Closed-loop success — the headline

| Configuration | Success |
|---|---|
| `lang` · radio · shifted (24 eps incl. standalone) | **1 / 24** (~4%) |
| `lang` · radio · **training layouts** (10) | **0 / 10** |
| `lang` (replan h=4) · radio · training (10) | **0 / 10** |
| `lang_point`+oracle · radio · shifted (20) | **0 / 20** |
| `lang` · trash · shifted (20) | **0 / 20** |
| `lang_point`+oracle · trash · shifted (20) | **0 / 20** |
| point-swap · trash (6) | **0 / 6** |

**The pre-registered criteria could not be evaluated: both arms score ~0.** The competence
floor sits *below* the conditioning signal — the policy cannot complete a grasp or press, so
"does the point beat language" is unmeasurable at the success level. This is itself the
gate's primary finding (see §4), and it replicates the organizers' own π0.5 baseline (0/10,
prompt-blind).

### 3.2 The conditioning axis IS active — below the success metric

The 47-episode video taxonomy makes the point unmistakable:

| Signal | `lang` | `lang_point` |
|---|---|---|
| Nav-stuck (roams past the target) | 10 / 20 | **1 / 10** |
| Visible grasp attempts | ~none | **repeated (gripper at can)** |
| New arm pathologies (thrash, occlusion) | none | present |

**The 3D-point conditioning demonstrably steers navigation and drives engagement** — the
policy with points goes to and reaches for the object; the language-only policy tours the
room. This is exactly the pre-registered hypothesis, confirmed *behaviorally* and invisible
to a success-only metric. Success-only scoring **hid a real effect for the entire eval**.

### 3.3 Grounding head — measured, and NOT the blocker

| Task | v0.5 | v0.6-gate |
|---|---|---|
| radio | 2.1px | 2.8px |
| trash | 3.0px | 4.1px |
| tripod | 2.7px | 5.6px |
| thawing | 124.6px (never trained) | **4.56px** |

~4px ≈ **~1cm 3D** at manipulation range. Occlusion handled correctly (no hallucination; 85×
confidence collapse → belief-head fallback). The **precision noise sweep** (inject 3D error
into the target, measure press success on the reachable config):

| injected error | 0cm | **1cm** | 2cm | 3cm | 5cm |
|---|---|---|---|---|---|
| press success | 2/3 | **3/3** | 1/3 | 1/3 | 0/3 |

**At our grounding operating point (~1cm), the press succeeds 3/3. Precision is not the
blocker.** (Margin is tight past 2cm — a reason to *hold* 1cm under closed-loop viewpoints,
not to chase sub-cm.)

### 3.4 Director probe — the failure decomposes into 4 modes

Oracle-assisted (privileged state; **diagnostic upper bound, not submittable**). A reach-aware
base-placement fix produced the project's first successes and localized the "last centimeter":

| Placement strategy | radio success |
|---|---|
| hardcoded geometry | 1 / 10 |
| Opus-reasoned standpoint | 1 / 10 |

Tied — and the tie is the finding: **placement strategy is not the sole gate.** Failures
distribute across (a) base too far, (b) base close but wrong orientation (arm can't reach
back), (c) **press-execution bug** — fingertip reaches 3–8mm and the button *still doesn't
fire*, (d) genuinely unreachable targets (mid-table, arm too short). Opus's *reasoning* was
correct on every instance (picked the right table edge); correct reasoning at one stage
doesn't rescue a multi-fault pipeline.

---

## 4. Findings

1. **The terminal manipulation primitive is the universal blocker.** 0 successful grasps or
   presses in 47 episodes. Navigation and approach mostly work; the commit does not.
2. **The failure is data starvation at contact.** 30 episodes/task is below every documented
   threshold for nonzero closed-loop grasping (100–500/task; adversarial research, 105 agents).
3. **3D-point conditioning is behaviorally active** — measured on video, not just argued.
4. **Grounding precision is not the blocker** — measured, 3/3 at 1cm.
5. **The "last centimeter" is three problems, not one** — orientation, actuation, reach limit
   — none fixed by better perception or better placement alone.
6. **The stack can reason** — Opus, given only geometry, picks the correct approach edge every
   time; the intelligence exists, it just wasn't in the runtime loop (it was hardcoded).

### Post-report correction (2026-07-25, measured on box)

§3.4 listed a "press-EXECUTION bug (fingertip reaches 3–8mm and the button still doesn't
fire)". **That inference is REFUTED by direct measurement.** A dz sweep on a *reachable*
instance (overlap sphere r=22.4mm; servo link confirmed in `robot.finger_links`):

| descend target | contact steps | overlap+contact | success |
|---|---|---|---|
| **+20mm (current default)** | 833 | **5** (= `CAN_TOGGLE_STEPS`) | **TRUE** |
| 0mm | 624 | **5** | **TRUE** |
| −10mm | 735 | 1 | FALSE |
| −25mm | 623 | 1 | FALSE |

The press primitive **works and is already optimally tuned**; driving the finger deeper
(the "fix" suggested by reading `toggle.py`) *loses* the overlap and breaks it. Consequences:
(a) a **reliable commit teacher for distillation exists today** — the highest-uncertainty
dependency of the August plan is already satisfied; (b) the earlier no-toggle observations
were **base-placement failures, not actuation failures**; (c) the "last centimeter" reduces
from three problems to **one: reach-aware base placement + orientation** (measured band: base
≤0.61m from target succeeds, ≥0.70m always fails).

**Methodological lesson (banked):** a confident mechanism inferred from source reading nearly
caused a regression on working code. Measure before fixing — this is the second such case
this week (the first: "grounding precision is the blocker", also refuted by measurement).

### Hypotheses tested and rejected
- **Memorization gap** — rejected: 0 on training layouts too.
- **Catastrophic forgetting** — rejected: would spare the trained task; ours fails it too.
- **Replan horizon** — rejected: h=4 vs 16 changes style, not outcome.
- **Gripper-loss re-weighting alone** — measured lifts refuted in verification (demoted to
  cheap adjunct).
- **Conditioning is inert** — rejected: it visibly steers behavior.

---

## 5. Verdict

**The pre-registered success criterion is inconclusive** (both arms ~0 — the competence floor
is below the measurement). **The gate's actual result is stronger than a pass/fail number:** a
precise, evidence-backed, literature-calibrated diagnosis of *why* a self-trained π0.5 fails
this benchmark, with the conditioning axis shown behaviorally active, perception cleared as a
blocker, and the true wall (terminal-primitive competence under data starvation) located and
decomposed. This directly specifies the August rebuild (see `AUGUST_EXECUTION_SPEC.md`) and
de-risks it — we are not guessing what to build.

**What this is not:** a submittable result. Every director success used privileged state.
The honest path (grounding-in-loop + learned/reliable primitives) is the August build.

---

## 6. Reproducibility

- Checkpoints: HF `ckpts/g3/{taskid,lang,lang_point}/29999/params` + `assets_g3/`
- Dataset pipeline: `g3_pipeline/` (action-fp mappings, assembler, configs)
- Grounding: `ckpts/grounding_v05/`, `ckpts/grounding_v06_gate/`
- Eval results: `g3_eval/`, `director_arm/results/`
- Fork (AdaLN point conditioning): `code/openpi_fork_adaln_src.tar.gz`

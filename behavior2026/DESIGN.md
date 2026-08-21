# BEHAVIOR 2026 — Our-Model Design Doc
**Date:** 2026-07-05 · **Deadline:** Oct 16, 2026 · **Status:** v1 draft
**Decision (user-locked):** our own model — fine-tuned from the commodity `pi05_base` core with OUR architecture from step one. No warm-start of third-party challenge weights. Larchenko 2025 = ideas + reference implementation only (Apache-2.0, cited); his checkpoint stays on disk solely as a carryover-task benchmark to beat.

---

## 1. Why 26% is the ceiling today (failure model that drives every design choice)

- Episodes are ~15k steps (~20 min). BC policies compound per-step error to near-zero over that horizon; **one unrecovered failure kills the episode**. 2025 winner's answer: hand-coded correction tables. NVIDIA's answer: post-hoc RFT (0.224→0.345, checkpoint-oracle **0.611**). The 0.26→0.61 gap is a *no-recovery, no-self-knowledge* gap — not a backbone-capacity gap.
- 2025 winner conditions on **task-ID embeddings** (no perception in the conditioning path) → zero transfer to the 50 new tasks; hidden-instance ranking punishes memorization.
- The q-metric pays partial credit on **final state only**: achieved-then-lost = 0; breadth across 100 tasks ≈ hidden-set rank (2025: nonzero-q task counts 43/42/15 for ranks 1/2/3).

## 2. Architecture — four additions to `pi05_b1k` (ranked by expected Δq)

### A. Corrective data engine + recovery skills (data axis — the headline)
- Success-replay RFT + **corrective-DAgger from failure states** using privileged scripted teachers (sim state legal at train time) + **AWR-through-the-loader** (advantage = predicate-delta; sampling weights, no off-manifold perturbation).
- Doctrine (hard-won on LIBERO, independently validated by ReActor @ SIGGRAPH'26): **physics-executed teacher data only** — no kinematic conversion; no dwell states; no sub-resolution decision features; no saturated control regimes.
- Optional scorer: frozen motion-prior as reusable rollout filter (SMP recipe, SIGGRAPH'26) — prototype only if cheap.

### B. Grounded relative-3D-point conditioning (anti-memorization axis)
- Replace task-ID-style conditioning with **depth-grounded relative 3D target points** injected into the action expert (AdaLN or dedicated token), re-grounded per stage.
- Train-time labels: privileged sim object poses (legal). Inference: depth + learned grounding head (our LIBERO grounding lineage).
- Mechanism validated externally (+32–46pp on π0.5/GR00T with *oracle* grounding, arXiv 2606.27663 — they punted on honest grounding; we own it).

### C. Conformal metacognition head + recovery dispatch
- Second-order head predicting **P(failure | stage)**, conformally calibrated on our own eval-fleet rollouts; on trigger → dispatch trained recovery behavior (retract-reapproach, re-grasp, re-ground), NOT hand-coded rules.
- This is our ε-research line made concrete; nobody in the competition has it.

### D. Stage head + predicate-aware task director (planning axis, ~zero GPU)
- Stage-prediction aux head w/ voting (adopt from Larchenko, cite) — anchors long-horizon progress.
- Director layer reads the (known-at-eval) BDDL goal: sequences flippable literals, cheapest goal-option selection, no-progress retry, **end-state parking** (final-state-only metric), anti-fall embodiment YAML.

### Adopted training/inference techniques (cited, orthogonal to memorization)
Correlated-noise flow matching; multi-sample FM (15/step); delta actions + per-timestep normalization; soft inpainting (30/26/4); receding horizon (temporal ensembling scored 0.00 in 2025); absolute joints; 448² head-camera res A/B (Comet: 2× on some tasks; gate on 24GB inference budget).

## 3. Concrete file-level diffs (wensi-ai/openpi fork @ 01177e0 — same commit Larchenko pins)

| # | File | Change | Status |
|---|------|--------|--------|
| 1 | `src/openpi/training/config.py` | `pi05_b1k_ours` entry: monolithic public dataset root + lerobot `episode_filter` per task | **DONE** (on box, `.bak` kept) |
| 2 | `scripts/compute_norm_stats.py` path | norm stats over our task set via the new config | **RUNNING** (gate G1) |
| 3 | `src/openpi/models/pi0.py` (or subclass) | 3D-point conditioning token/AdaLN into action expert; stage + metacog aux heads | next |
| 4 | `src/openpi/configs/robots/b1k.py` | unchanged (61-D proprio layout verified vs 2026 eval at serve gate) | verified |
| 5 | `src/openpi/shared/eval_b1k_wrapper.py` | receding horizon exists; add inpainting + director hooks + metacog trigger | later |
| 6 | new `src/openpi/b1k_ours/` | grounding head, director, recovery-skill policies, corrective-data collectors | scaffold next |

## 4. Data plan
- Public dataset = ONE monolithic LeRobot-v3 repo (docs' per-task filter is dead). Per-task subsetting solved two ways: surgical file download (episode→chunk mapping; radio task = 200 eps / 12 files / 2.1 GB) + `episode_filter` at load time.
- v1 task set: fat-middle pick-place (79% of all goal literals are kinematic placement) + single-literal quick wins; abandon the particle/cleaning dead tail (0 for everyone in 2025).
- Norm stats / FAST tokenizer / Cholesky (for correlated noise): recompute per task-set on 2026 data.

## 5. Gates & timeline
| Gate | What | Where | Target |
|------|------|-------|--------|
| G1 | Monolithic loader + norm stats e2e | 5090 box | now |
| G2 | Single-task fine-tune (radio) from `pi05_base` → serve → eval ≥ baseline pi0.5 ckpt behavior | 5090 (LoRA/low-mem if VRAM-bound) | ~Jul 12 |
| G3 | Grounded-point injection A/B on 3–5 tasks (the decisive experiment: does honest grounding beat task-ID conditioning on held-out instances?) | 5090 + short H100 burst | ~Jul 24 |
| G4 | Multi-task run (v1 task set) + eval fleet + first leaderboard submission | H100 burst + 4090/5090 eval fleet | ~mid-Aug |
| G5 | Corrective-DAgger/AWR round 1 + metacog + recovery | fleet | Sep |
| — | FREEZE + official 1000-rollout eval + submit | | ~Oct 10 |

## 6. Risks (honest)
1. **Training compute**: full π0.5 fine-tune wants H100-class; 5090 (32GB) likely LoRA/low-mem only → G2 may run at reduced fidelity; budget the H100 burst early.
2. **Grounding-head quality on 720² household scenes** — harder than LIBERO tabletop; mitigation: privileged-label pretraining + depth (RGBD is given at eval).
3. **No warm-start = no free carryover q**: our first leaderboard number arrives ~2–3 weeks later than a warm-started banker would. Accepted trade for ownership.
4. **Sim throughput** for corrective rounds + self-eval (~350–450 GPU-h per full eval) — jobqueue fleet is a first-class deliverable, not an afterthought.
5. Metric mechanics must be enforced in the director from day one (parking, cheapest-option) — free points, easy to forget.

## Eval observation legality — ADJUDICATED 2026-08-09

Raised by SPATIAL_INTEL_RESEARCH_2026_08_09.md §2.1 (apparent contradiction between this
doc's "RGBD is given at eval" and the depth-illegal premise circulating since the 08-08
survey). Verified against behavior.stanford.edu/challenge/evaluation.html on 2026-08-09:

- **Allowed at eval: "RGB + depth + proprioception."** Depth images ARE legal policy inputs.
- **Prohibited: "ground-truth segmentation, object state, target object pose, full-scene
  point cloud, robot global pose, or other simulator-only privileged information."**

Consequences: the live map + affordance point consuming eval depth are fully legal
(Run-1b "all-legal serving" claim stands). The 2026-08-08 survey's blanket
"depth/point-cloud policy inputs illegal" premise is WRONG for sensor depth (right for
GT/full-scene point clouds); its discard reasoning for depth-input methods (PointVLA,
3D Diffuser Actor, BridgeVLA, SpatialVLA, QDepth-style depth experts on real depth, etc.)
is void on the legality axis and those items stand or fall on cost/evidence only.

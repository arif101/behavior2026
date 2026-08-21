# Graphics/animation sweep for B26 (2026-07-21, wf_041eb865 + follow-up wf_b30608d2, 209 agents)

Sweep 1 missed the live SIGGRAPH 2026 program (user caught it — conference running Jul
21-23); sweep 2 reached it via the ACM schedule system (sess104/sess105) + arXiv
journal-refs + a partial Ke-Sen index fetch (index still filling mid-conference).

## From the ACTUAL SIGGRAPH 2026 program (follow-up sweep)

- **GPC** (Peng group, 2606.29148, IN PROGRAM) — FSQ skill-token generative pretraining
  for motor control with **EMERGENT perturbation recovery** (no dedicated recovery
  mechanism). The one genuinely new program item for us. Tension to resolve before W3:
  does recovery emerge from scale, or need our dedicated corrective loop? (Their recovery
  numbers not adversarially verified; read before finalizing RFT design. Note we rejected
  FSQ for our LIBERO motor — different context, action-head vs skill-token pretraining.)
- **SMP** confirmed in-program; MotionBricks/ARDY/MUSIC/MOCHI = program's other motion
  papers, low transfer value. Awards (5 best + 10 HM): zero manipulation-relevant.
- **SCRIPT** (2605.22894, venue unconfirmed) — RL post-training of a TRUE flow-matching
  policy: learnable noise injected into Euler sampling (ReinFlow-style sigma-net over
  flow-time+state) + PPO, 128 parallel envs. = the concrete mechanism template for
  RFT-ing pi0.5's flow head. Their own ablation: RLHR gains modest (FID 0.203→0.164) —
  consistent with our plan: corrective-DAgger/AWR carries the bulk, RLHR is polish.
- **InterPrior** (CVPR 2026 Highlight, NOT SIGGRAPH) — 3-stage distill→perturb-augment→
  RL-finetune with **prior-preservation env split** (subset of parallel envs keeps the
  distillation objective during RL) = validated ALTERNATIVE to our proximal anchor;
  68.8% on 3-subgoal chains (+39.7pp over MaskedMimic). Cheap A/B inside W3.
- **XDiffuser** (2605.16863) — plan-first-diffuse-later: classical planner as connectivity
  oracle guiding denoising. Same extrinsic-guidance family as OmniGuide; director-computed
  plan → sampler guidance is the analogue.
- RQ3 (appearance randomization w/ measured perception gains): STILL zero surviving claims
  (absence over ~6% of program; index incomplete). One refuted claim properly killed
  (2604.07984 headline 0-3; its Table-3 robustness numbers survived 3-0).

## Ranking update: ADD GPC-read + SCRIPT-RLHR-template + InterPrior-env-split A/B.
## Nothing displaces the sweep-1 top-5 below.

## Ranked adoptions (impact × buildability)

1. **OmniGuide (arXiv 2603.10052, UPenn preprint)** — inference-time steering of
   flow-matching VLAs: add gradient of a differentiable 3D energy field (Gaussian
   attractors / SDF repellers, backprop through differentiable FK) to the flow VELOCITY at
   each denoising step. Demonstrated ON π0.5 (real: 24.2→92.4% success; sim ablation:
   denoising-guidance alone ≈ +20pp). **Fits our recovery dispatch exactly**: corrected 3D
   target from grounding/belief → attractor, no retraining. Caveats: sim numbers used
   ORACLE perception on a curated suite; ~2× latency (15 Hz vs our 13.5 FPS budget —
   borderline); needs differentiable R1Pro FK; unreviewed.
   ⚠ Open (their words + ours): interaction with a policy ALREADY point-conditioned via
   AdaLN is unmeasured — complementary channel or conflict? We are positioned to answer
   this (G3 + guidance A/B).
2. **PDP (SIGGRAPH Asia 2024) data-construction ablation** — corrective data must pair
   **NOISY state ↔ CLEAN corrective action**: 100% perturbation-recovery vs 3.36%
   (clean/clean) vs 59.5% (noisy/noisy). Near-zero cost — it is a formatting decision for
   our physics-snapshot corrective-RFT data. Adopt verbatim in W3.
3. **PARC (SIGGRAPH 2025, Peng group)** — generate → physics-correct → retrain flywheel:
   tracker 27%→68% over 4 iterations, no new human data. Validates our snapshot
   replay/hard-mining loop shape; expect monotonic-but-diminishing returns.
4. **Contact-map intermediates (CMT NeurIPS 2025; BimArt CVPR 2025)** — object-centric
   contact maps generalize to unseen categories far better than direct pose gen (74.14% vs
   39.16%). BUT BimArt's own ablation: contact prior ALONE worsens penetration
   (9.45→20.27%); headline needs test-time optimization we cannot afford at 13.5 FPS.
   → use as AUX SUPERVISION / grasp-place VERIFICATION (stage-head candidate), never as
   online refinement.
5. **SMP (arXiv 2512.03028, Peng group)** — frozen motion-diffusion prior as RL reward
   (getup-from-arbitrary-fallen 0.998). For us: optional frozen action-prior regularizer
   inside AWR — keep proximal anchor primary.

## Negative results that scope our design

- **CondMDI (SIGGRAPH 2024)**: inpainting-style keyframe imputation on a model NOT trained
  for it FAILS (ignored or discontinuous jumps) → naive action-chunk editing on π0.5 is
  ruled out; gradient guidance or trained-in conditioning are the only viable eval-time
  paths.
- **Guided Action Flow (arXiv 2607.02092)**: learned Q-critic guidance works single-task
  (+14pp) but held-out generalization collapses (+2.5pp = 1 success in 40) → for our
  timescale, ANALYTIC energy fields (OmniGuide-style) beat learned guidance critics.

## How this lands in our plan

- **This week**: point-CFG knob (null-embed guidance, free) + OmniGuide-style velocity
  guidance prototype in the serve wrapper (needs differentiable R1Pro FK — JAX, ~1-2 days).
  Both become eval-time arms IF raw G3 arm-3 lift is small (weak-reliance mitigation).
- **W3 (Aug)**: PDP noisy/clean formatting (mandatory), PARC flywheel shape, SMP optional.
- **Stage head**: contact-map aux-supervision idea → discuss with cofounder after Aug 8 gate.
- **Follow-up**: re-sweep RQ4/RQ5 when SIGGRAPH 2026 program indexes.

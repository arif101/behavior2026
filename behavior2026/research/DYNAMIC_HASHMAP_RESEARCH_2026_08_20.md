# Dynamic-hashmap research (2026-08-20)

**Scope.** Ten deep-read dossiers across five axes (retrieval, equivariance/adaptation, conditioning liveness, privileged 4D latents, test-time steering), plus twelve sweep-only candidates. **Headline result of the verification pass: 10 of 10 deep-read dossiers landed on `watch`. Two were argued `adopt_now` by the reader (FRS, RL²-VLA) and both were downgraded on verification. Zero `adopt_now` survived.** No paper in this set is strong enough to buy a slot in the one remaining retrain on its own evidence. What the set *does* buy is a sharpened diagnosis, three cheap instruments, and a named list of things not to fund.

Short names used throughout: [DARP] 2606.09758, [EquiContact] 2507.10961, [HEP] 2502.05728, [Cross-View] 2608.06965, [Pri4R] 2603.01549, [Mech-Study] 2603.19233, [OmniGuide] 2603.10052, [FRS] 2606.13675, [RL²-VLA] 2607.26991, [RtS] 2605.10094.

---

## 1. Verdict on the framing

**The hashmap model is supported — as the default outcome of BC on a flow VLA, not as a pathology of our wiring.**

- [Mech-Study], 394k+ episodes across six VLAs: pi0.5 cross-task activation injection, n=1,968 pairs → **2.6% destination success, 99.6% source-dominant trajectories**; X-VLA n=3,150 → **0% / 99.8%**. Authors' conclusion: fine-tuned VLAs execute *spatially bound motor programs tied to scene coordinates*, not task abstractions.
- Same paper, the result that matters more: a layer-17 linear classifier reads prompt category at **99.3%** while behavior is statistically prompt-invariant (**F(4,3391)=1.23, p=0.247**). *Encoded-but-not-used is the baseline state of a BC-tuned pi0.5*, with no auxiliary channel involved.
- [RtS] is the field building this deliberately: keys are the base VLA's own patch tokens, cosine-matched at γ=0.9992, values are stored successful segments. It works, slightly (§3).

**Where the framing breaks, and this is the load-bearing correction.** The hashmap metaphor predicts *wrong value retrieved* — a mode-**selection** error, implying mass on alternatives. Our measurement is stronger and categorically different: `RUN2_EVAL_REPORT.md` records **commit_mass = 0.00 at both 0.10 and 0.20 rad thresholds, p10–p90 spread 0.004 rad, decision key = MODE ABSENT** ("confident lock on a non-commit mode, not indecision"). That is a hashmap with a **missing value**, not a wrong one.

This single distinction disqualifies a family. The [DARP] verifier states it cleanly: DARP is a variance reducer and a first-order local adapter (Thms. 1–2 are Laplacian smoothing bounds); aggregating *k* commit-free neighbors yields a commit-free action. **Smoothing a bank with a hole does not fill the hole.** [FRS] and [RL²-VLA] deliberately chose base policies *off* the eval distribution "to allow room for improvement" — diffuse priors with latent mass to select among. [RL²-VLA]'s illustrated failure mode is an *oscillating* policy (Figs. 11–12: "oscillations above the spoon"); ours is confident collapse. **No paper in this set measured anything in our regime.**

**Open measurement, cheap.** Three dossiers built arguments on "93% of failures end >15cm away ⇒ approach is a mode-selection error with live alternatives." That is an inference. The [FRS] verifier flagged it. Nobody has run the commit-probe harness at *approach*-phase states. See §6/A2 — 2 hours, and if approach is also collapsed, all of §4 reduces to one line.

**Second correction, from [Mech-Study]:** expert-pathway injection produces *active wrong reaching* (231–337 steps); VLM-pathway injection produces *passive stalling* to the 520-step limit. Our failure signature is the former. If that dissociation transfers, our VLM half is arguably intact and the intervention belongs in the expert conditioning path — but the pi0.5 dissociation there is qualitative, the "2x" quantity comes from a different model (SmolVLA, 15.8 vs 9.0), and grade is C.

---

## 2. The liveness problem

### Why injected geometric keys die

Appearance is a sufficient statistic for the action on every in-distribution row, so the geometric channel is redundant and gradient descent never routes through it. Four independent replications, different codebases:

| Source | Injected-channel result | Same paper, geometry elsewhere |
|---|---|---|
| [Pri4R] Table IV | P_t as backbone input tokens: **+0.2 avg, Pick-and-Place −7.8** | P_t as *prediction target only*: **+13.2** |
| [Cross-View] App. G | 3 feature-level injections on **pi0.5**: canonical cross-attn tokens **15.3% vs 16.8% floor**; gated residual — *"the gate remained nearly closed"*; forcing it open **collapsed nominal rollouts** | authors abandoned injection, regularized the action output instead; winning method carries **no geometric channel at all** |
| [HEP] Fig. 6 "No FT" (= key as side-channel conditioning) | 0.78 vs 0.94 full; **non-equivariant** arm 0.60 vs 0.70 | frame transfer (key = coordinate origin) |
| [Mech-Study] | 99.3% decodable / p=0.247 behavioral effect | — |

Our own evidence is stronger than any of theirs: the pre-registered causality ablation **fired the "decorative" flag** — masked points, no degradation, right-wrist approach *deeper* masked (**0.011 m**) than live (**0.030 m**). The channel is not merely uninformative; ablating it costs nothing.

**Note the direction of [Cross-View] App. G.** It is a documented failure of exactly the intervention class Run-3 is pre-registered to attempt, plus an ID-collapse warning when you force the auxiliary route open. It is exculpatory evidence against the Run-3 framing, not support for it.

### Ranked measured fixes

**1 — Make the label something appearance cannot predict (data).** The strongest argument in the set, and it is ours, not a paper's. With byte-identical RGB/wrist/proprio and two different correct chunks *a, a′*, the appearance-only predictor's best response is the mean, at an irreducible per-row flow-matching loss of **‖a−a′‖²/4**. Reading the point is the *only* escape. This is a closed-form, falsifiable, **rollout-free** training-time readout computable from pair metadata on held-out pairs. It holds under plain FM before any consistency term is added. [Cross-View] is the δ_u = 0 special case; the residual generalization is L_EQ = ‖(v_p − v_0) − (a − a′)‖² with shared (t,ε).

Two corrections that must travel with it:
- **Sign.** "Flip the consistency loss to reward divergence" is wrong. Negating λ gives coefficient (1−4λ) on ‖δ‖², unbounded below past λ=0.25 — the model buys free loss by exploding view-specific residuals. You *target* a residual, you never maximize one.
- **The serving-regime objection (verifier, and it is serious).** At eval the affordance head is RGB→3D, so the point is a deterministic function of the image and *can never contradict it*, while every counterfactual pair is defined by point contradicting image. Training obedience to a regime that never occurs at eval buys a live channel whose marginal information content at serving is bounded by **affordance-head error alone (2.6 cm nominal)**. Mitigation: draw perturbations from the head's *realized* error distribution (σ≈2.6 cm, tail to ~10 cm) rather than 20 cm adversarial displacements only, and keep a substantial fraction of correct-point rows. This bounds the honest ceiling and should be measured, not assumed.

**2 — Make the label a function of the key (structural).** Verifiers gutted both candidates at our substrate. [EquiContact]'s headline (CompACT 19/20 ID → **0/10** unseen translation vs 20/20 for GCEV) conflates frame choice with the *presence of a goal channel* — CompACT receives no g_ref at all — and we are already on the goal-channel side and dead anyway. Its Assumption 1 fails in its own Table IV (wiping **0/5, 0/5** at extreme transforms, with the authors reporting the policy tracking aluminum-extrusion slots instead of the target line — an appearance shortcut, in the paper claiming to forbid them). The proposed "action-frame only" arm is **published negative**: State-free Policy (2509.18644) holds relative-EEF actions fixed on pi0 and reports **0% height / 6% horizontal** generalization; the lever that worked there was *deleting* proprio and global cameras — unavailable to us with 100 language-conditioned tasks and a mobile base. [HEP]'s guarantee requires the entire input be coordinatized; you cannot subtract a 3D point from RGB tokens, and partial canonicalization *is* the No-FT arm. In-house counter-evidence: we already run a wrist-local foveated map and the channel died anyway.

**3 — Constrict the competing route.** Untested at our scale, but the only levers that are free inside a retrain we already pay for. Sweep-only, so evidence grade unknown: [sweep] StableVLA/IB-Adapter (ICML 2026, <10M-param adapter throttling visual information reaching the action head — makes geometry the cheapest remaining path to low loss); [sweep] AC-VLA (2607.15714, instantiated **on pi0.5**, state-conditioned asymmetric masking that suppresses wrist view specifically during closed-gripper phases — our stage head at 0.998 acc supplies the phase gate); [sweep] Flatness/SAM (2606.23641, optimizer swap only, no data, no architecture). Recommend deep-read triage (§6/A6), **not** adoption on sweep-note evidence.

**4 — Geometry as prediction target.** [Pri4R] polarity result is real but the marginal question is unmeasured: Run-2 **already ships depth_aux** (`box_scripts/patch_run2_config.py:81`), so the question is point-tracks-vs-depth on pi0.5. Scaling Table III's depth share (+8.3 of +13.2, OpenVLA-OFT only) onto pi0.5's total (+4.1) implies **~+1.5 marginal**, single run, no error bars anywhere in RoboCasa. Its own mechanism claim is oversold ~4x: robot-points-only reproduces +10.7 of +13.2, and robot-point future displacement is approximately FK of the action chunk — a dense task-space reparameterization of the action loss, not "world dynamics." And it never ablates z_t (§8).

---

## 3. Soft keys — retrieval routes

**[DARP].** Split it. The reparameterization insight (arm 7: s_q vs Δs with s* present in both — informationally equivalent, yet the absolute form collapses) is the one genuinely interesting control. The verifier showed the analogy to us is false: in DARP the neighbor tuple (s*, a*) is a *necessary* input, so no competing shortcut exists; ours is a goal channel losing a shortcut competition. Retrieval half is unadoptable: best **k≈500 on a 4,200-point corpus** (~12% of the data per query), small k *loses to BC*, distance is a decayed trajectory window (single-frame is the documented losing config), and the paper's own closest analogue to a VLA token interface — REGENT — is its **worst** baseline (below random on Ant). Table 11 is the buried lede: honest validation-loss model selection collapses Hopper BC 2313→507 and Stack 0.47→0.12; only the ratio (~1.6–2.6×) survives.

**[RtS] — the literal dynamic hashmap, on a frozen policy.** LIBERO-10 pi0.5 **92.4 → 94.4** = 2.0pp = **1.75 binomial SE**, Fisher p=0.25. **No comparison in the paper reaches p<0.05 unpaired**; 5/5 real tasks point the same direction (sign test p=0.03), so: small and probably real. Reported seed stds (±0.2) are ~5× *tighter* than the single-run binomial SE (1.14pp) — coherent only if seeds share initial states.

Three reasons it is not adoptable here, in order:

1. **Protocol.** Memory is per-task, built from successes *within* the eval. Our denominator is all 100 tasks, unattempted = hard 0, run1b conversion 1/31. If the graded run is ~one episode per task, memory is empty at every scored step and the method is a no-op by construction. **Check the rules doc — minutes, and it gates the whole branch (§6/A0).**
2. **Aliasing inversion.** Identical visuals → cosine ≈1.0 → high s̃; neighbors agree → low σ_DTW → high confidence → t₀→t_min → **steers hard toward the wrong action**. Confidence is anti-correlated with correctness exactly on the counterfactual pairs we can manufacture. The paper concedes the mechanism (Limitation 4). Table 5: dirty memory scores **87.6 vs base 92.4** — downside 2–3× the upside.
3. **No shuffled-key control anywhere**, and **93.6 of the 94.4** is delivered by plain top-1 warm start. This team, having just probe-verified its own channel dead at <1% action delta, should not accept an injected-signal claim without the control it applied to itself.

**Serving cost, honestly.** "1.10×" excludes the VLAC critic forwards. Key dimensionality is never stated; worked through for our stack (~64 pooled tokens × ~2048 × 3 views ≈ 400k dims/entry) a 5k-entry index is **~4 GB fp16** on a 24 GB box already hosting OmniGibson. Mandatory PCA the paper silently skips.

**Two things worth taking from this axis, both free:**
- **Progress-peak prefix truncation** ([RtS] Table 5, worth **+2.6pp on its own**, independent of retrieval): keep only the prefix up to the max-progress timestep, discard post-success redundancy and regressions. Our stage head already computes the signal. Adopt into the Run-3 mix and the RL success bank at zero cost.
- **The retrieval key as an instrument** (§6/A1): compute k_t on counterfactual pairs and measure the fraction above γ=0.9992 whose correct actions diverge. That is a quantified appearance-aliasing rate — the strongest possible justification for a liveness bar, and a targeting map for pair-generation budget. Hours, no training.

---

## 4. Value adaptation — equivariance, frames, test-time steering

**Equivariance/frames.** Covered in §2/fix-2. Net: [EquiContact]'s isolating ablation measures goal conditioning, not frame choice; [HEP]'s transferable number for a non-equivariant backbone is **No-Equi 0.70 vs No-Equi-No-FT 0.60** — +10 points, ~1–2σ per task, **zero seeds anywhere in the paper**, 6 ablation tasks of which 2 are saturated at 0.96–1.00. [HEP] is also 5-year-old prior art on the recentering itself (C2F-ARM 2021, RVT-2 zoom-in 2024) and is *behind* that prior art on the axis that decides our case: it trains the low level on GT keypose and **never measures coarse-anchor error tolerance**. Our anchor is 2.6 cm against a 2.2 cm target.

**Test-time steering — three mechanisms, three structural verdicts.**

| Method | Can it add action mass? | Blocking fact |
|---|---|---|
| [OmniGuide] energy gradient through differentiable FK | No | **dL/d(gripper) ≡ 0** for any EE-position energy — indices 14 and 22 on R1Pro receive identically zero gradient. Cannot emit a grasp-close. Sim semantic result uses **GT poses**; the one clean dose-response is at λ_C=0.02 while the semantic attractor runs at λ_S=**5.0** (250× larger, unswept); σ_S never given a numeric value; no wrong-attractor control. |
| [FRS] flow reversal warm-start | Yes, in principle | **Oracle ceiling**: handed chunks from a 92.8%-success expert, FRS delivers **33.7%** on the hard split — the channel destroys ~2/3 of whatever competence you feed it. Corollary that kills the obvious use: *if your reference action is good, execute it, don't steer with it.* Routing RLPD actions through FRS is strictly lossy. Also: 15-task DSBC and DSRL splits are **defined as** "tasks where FRS gained ≥10 pts", then evaluated on the same rollouts that did the selecting. |
| [RL²-VLA] velocity composition v = w·v_VLA + (1−w)·v_RL | **Yes — the only family that injects rather than selects** | Every pi0.5 number is measured on top of **8 rephrases × 5 samples**, including Compose-Always. There is **no single-prompt pi0.5 datapoint anywhere**. Compose-Adaptive's fallback branch is Rephrase, which with fixed BEHAVIOR task specs collapses to vanilla 14.3 — a configuration with zero datapoints. Isolated composition effect: **+2.0, p≈0.4, n=3 seeds**. Blending w≈0.5 against a *confidently wrong* field yields a convex combination valid for neither. |

Run-2's own decision already recorded the structural point (report item 1): **all selection-class levers are support-dead at this state class** — mode-sticky decoding, best-of-K, DSRL, chunk-critic reranking. Nothing to select among 25 identical non-commit samples. Composition and warm-start are the only two that can add mass, and both blend against a collapsed field with no measurement in that regime.

**Real OOD evidence in our regime: none.** That is the honest one-line answer to §4's question.

---

## 5. Run-3 design implications

**Data mixture.**
1. Counterfactual pairs (sim restore, same visuals, perturbed point, different correct action) with perturbations drawn from the affordance head's realized error distribution (σ≈2.6 cm, tail to ~10 cm), plus correct-point rows. **Mix ratio is the unanswered variable** — [Cross-View] used 100% pairs (338,575) and explicitly names pair-count and step-budget sweeps as unresolved. A thin slice can be absorbed as label noise at the mean-prediction penalty; that is precisely what the loss-floor check detects.
2. RL-skill successes, **prefix-truncated at the progress peak** ([RtS] Table 5).
3. Retain failed rollout segments (already the Run-2 decision — now provably the majority behavior class).
4. **Pair-consistent spatial augmentation as a hard rule.** Worth +4.0pp in the only paper that measured it, and for our pairs the point channel is the *sole* discriminative content, so any independently-sampled crop injects pure noise into exactly the term meant to revive the channel. Photometric aug may stay independent.

**Losses.** Plain FM on both rows carries the loss-floor argument. The residual term L_EQ = ‖(v_p − v_0) − (a − a′)‖² with shared (t,ε), K=2, λ≈0.10, t~Beta(2,3), bilateral gradients is the sharpener, not the mechanism. **Mandatory control: the deranged-pair arm** — derange which (p′, a′) attaches to which row *inside the residual term only*, leaving both FM targets row-local and correct. Without it, a positive result is uninterpretable (noise inflation reads the same as liveness). [Cross-View]'s own shuffled control collapses to 25.8% camera / 50.8% ID, i.e. **wrong pairs are actively toxic, not merely useless** — given the b1k_radio_map qvel incident (229df05), the state-hash / action-chunk-hash validity gate is load-bearing, not hygiene.

**Conditioning routes.** Δ-form (p_target − p_ee) is *already what we ship* (`eval/oracle_point_wrapper.py`, per `scripts/b1k/add_target_points.py`) and it is dead — so re-expressing the point in the EE frame is information-preserving up to a learnable rotation and is **not a lever**. The remaining untried structural levers, all ~free relative to the retrain: constrict the appearance route (IB-Adapter-class; phase-conditioned wrist masking gated by the stage head), geometry-as-prediction-target (partially shipped), optimizer flatness. All three unvalidated at our scale.

**The pre-registered liveness bar — restate it, explicitly and dated.** Three defects:
1. **Denominator.** Action-delta is fragile: [Mech-Study] shows 99.3% decodability with dead behavior, and near-unity action cosine (0.999) coexisting with 15–19pp success deltas.
2. **Tautology risk.** Under any Δ-label or frame-transfer variant the bar passes by arithmetic construction ([HEP] dossier's own warning). Same for an analytically-imposed additive guidance term.
3. **No ceiling.** Every liveness number so far is uncalibrated.

Proposed replacement, extending rather than discarding `probes/point_liveness.py` (whose decision key already separates DEAD / PRESENT-BUT-NOT-SPATIALLY-SELECTIVE / LIVE):
- **Tier 1 encode** — decodability of the point from expert activations. *Expected to pass even when dead; must not satisfy the bar alone.*
- **Tier 2 necessity** — project the point subspace out, measure Δsuccess and Δterminal-distance. **Hook the MLP sublayer, not the full residual stream** — that exact bug inflated half of [Mech-Study]'s appendix (corrected hooks → p=0.975).
- **Tier 3 sufficiency** — on save/restore counterfactual pairs, **cm of EE displacement toward the perturbed point**, plus commit-mass, plus the analytic FM-loss-floor crossing. Denominated in cm and rad, not action-delta.
- **Floor** = Run-2 checkpoint. **Ceiling** = the same assay on the RLPD terminal skill (point-supervised by construction).

Governance note: this is a pre-registration amendment made *after* Run-2 results are in hand. Date it, state the reason (tautology + missing ceiling), and freeze it before the retrain — otherwise it is post-hoc criterion selection.

**Non-negotiable framing.** A live channel is necessary, not sufficient. Run-2's ablation showed masking the points cost nothing; reviving them buys nothing unless the commit mode exists. That remains the RL skill's job, and its cold scenes sit at **0.00 across ~90 episodes** with the recommended fix (frontier demotion to never-trained stage 0) **unrun** as of the 08-15 mid-pass note.

---

## 6. Pre-Oct-10 actionables, ranked by EV/cost

| # | Action | Cost | Smallest experiment / kill criterion |
|---|---|---|---|
| **A0** | **Rules check**: trials per task in the graded run; legality of cross-episode memory writes | **minutes** | Read the rules doc. If ~1 episode/task, the entire retrieval branch is a no-op by construction and is discarded today. |
| **A1** | **Appearance-aliasing probe** | 4–8 eng-h, no training | Implement only the [RtS] key extractor (patch tokens → 2×2 pool → concat → L2). On counterfactual pairs measure cos(k) distribution vs same-state repeats vs null; report fraction above **0.9992** whose correct actions diverge, and that divergence in cm. **Cannot fail to inform**: high aliasing = quantified justification for the liveness bar + targeting map for pair budget; low = the dead channel matters less than we think. |
| **A2** | **Approach-phase diversity measurement** | ~2 h, reuses commit-probe harness | Run the 25-sample chunk probe at states from failure rollouts that ended >15 cm away. If diversity is *also* collapsed, §4 collapses entirely and three dossiers' central fit arguments are void. If it is multimodal, composition-class steering becomes worth a second look. |
| **A3** | **Loss-floor pilot** (the decision experiment) | 1–2 eng-days + <1 A100-day (~$50–150) | Warm-start from Run-2, mix a counterfactual-pair slice, ~1–2k steps. **Primary readout is analytic and rollout-free**: does held-out counterfactual-row FM loss fall below ‖a−a′‖²/4? Plus deranged-pair control, plus the Tier-3 assay. **Kill: if matched training cannot cross the floor in 2k steps at the intended mix ratio, the channel is not recoverable by this route and Run-3 does not spend on it.** Sweep mix ratio second. |
| **A4** | **Liveness-bar ceiling calibration** | hours | Run the A/B/C assay (real / +20 cm displaced / zeros, noise fixed per sample) on the RLPD skill. Without a measured ceiling every liveness number is uncalibrated. |
| **A5** | **Progress-peak prefix truncation** | ~0 | Apply to the RL success bank and any eval-rollout replay feeding Run-3. Stage head emits the signal. |
| **A6** | **Triage reads only**: Flatness/SAM, IB-Adapter, AC-VLA wrist-phase masking; plus CofactVLA (2608.04396) and Track4Action (2608.03727), surfaced by verifiers and never read | ~half day each | These are the only candidate Run-3 arms that cost a config line. Do **not** adopt on sweep-note evidence; two verifiers noted CofactVLA targets our exact "vision-override" pathology in flow-matching VLAs inside a *single forward pass* — no paired corpus, no ~2× trunk cost. |

**Explicitly not funded**: retrieval index at serving; activation steering at serving ([Mech-Study]: pi0.5 expert steering is the most fragile of six models, −84pp at −3×, recovery 0% in all three variants — though note three concurrent papers report positive VLA steering, so treat this as *unfunded*, not *refuted*); SAE work (424 SAEs, 4.3 TB, corrected-hook result p=0.975); differentiable-FK guidance; action-space change plus differential IK for 23-DoF dual-arm + torso + base (redundancy resolution, not IK); DARP retrieval inside the training loop.

---

## 7. Watch list + discard pile

**Watch (all ten deep-reads; re-consult only on a fired trigger):**
- **[DARP]** — keep the Δ-vs-absolute parameterization control as an experimental template; retrieval unadoptable (k≈500 on 4.2k points, REGENT is its own worst baseline).
- **[EquiContact]** — keep only the g_ref noise-injection recipe at σ matched to 2.6 cm, as a V2-skill config flag.
- **[HEP]** — re-consult if we ever produce the coarse-anchor tolerance curve it never measured.
- **[Cross-View]** — keep the algebra (loss floor, targeted residual, pair-consistent aug, shuffled control); App. G is evidence *against* the Run-3 framing.
- **[Pri4R]** — re-consult only with a z_t-liveness control and a depth-aux-vs-point-tracks comparison on pi0.5.
- **[Mech-Study]** — instrumentation only (three-tier assay, MLP-sublayer hooks); not a method.
- **[OmniGuide]** — structurally cannot emit commit; revisit only if approach-only gating is ever wanted.
- **[FRS]** — the oracle ceiling (92.8% expert → 33.7%) is the fact worth remembering; kills the RLPD-through-FRS idea.
- **[RL²-VLA]** — the only mass-*injecting* mechanism; revisit iff A2 shows live approach diversity **and** the single-prompt cell is ever measured by someone.
- **[RtS]** — gated by A0; take the truncation rule and the key extractor, discard the method.

**Discard now:** serving-side retrieval indices; eval-time activation steering; SAEs; sample-and-rank / best-of-K / mode-sticky decoding at commit states (support-dead, already recorded in Run-2's decision); re-expressing the point in the EE frame as a "revival lever" (already shipped, already dead); reimplementing HEP's escnn stack; DARP's REGENT-style retrieval tokens on the pi0.5 prompt.

---

## 8. Evidence-quality audit

**Aggregate.** 10/10 deep-reads → `watch`. 2/2 `adopt_now` claims downgraded. Evidence grades: one B+ ([Cross-View], [FRS], [RL²-VLA], [Pri4R] at B), the rest C. Every one of the ten has at least one of: zero training seeds, a max-over-config headline, or a missing control that the team's own probe discipline would have required.

**Max-over-config / mis-transcribed headlines, by name:**
- [FRS] "up to 95 pts absolute" = single-task max. Matching aggregate: **4.5 → 11.1** on the 62-task hard split = **+6.6**. ~14× gap.
- [Pri4R] "+10% LIBERO-Long / +40% RoboCasa" = OpenVLA-OFT. On **pi0.5: +3.8 and +4.1**. On pi0, LIBERO-Long **85.7 → 85.6 = exactly zero**.
- [RL²-VLA] "+17.3" and "+19.4" = best of 3 and best of 4 tasks respectively.
- [EquiContact] "13–18 demos per task" = 13–18 **prompts** per task. The demo count is reported nowhere — paper, appendix, or project site.
- [HEP] "one-shot 80% vs 5%" = n=20, one real task, one toy car. The decision-relevant number is No-Equi 0.70 vs 0.60.
- [DARP] "Peg 17→52" is the **vision** table; state-based Peg is 46→62.
- [Pri4R] reports the same OpenVLA-OFT LIBERO-Long baseline as **83.0 / 85.5±0.2 / 89.2** in three tables — a 6.2-pt spread against ±0.2 claimed error bars, internally impossible. The abstract quotes the largest resulting gain.
- [RtS] "1.10× latency" excludes the VLAC critic forwards entirely.

**Missing controls, by name:**
- **No shuffled-key / random-retrieval control anywhere in [RtS]** — so "retrieval content is informative" is unestablished, and 93.6 of 94.4 is top-1 warm start.
- **No cross-seed injection null in [Mech-Study]** — the condition is *defined* in Sec 3.2 and its results appear nowhere. That is the null distribution for the entire 99.6%/99.8% displacement family, which is itself only a rank comparison cos(traj,src) > cos(traj,dst) that a frozen trajectory wins by default.
- **No z_t control in [Pri4R]** — never zeroed, permuted, or stop-gradded the shared embedding the whole mechanism claim rests on, and most of 1024 points in a robot-centered cube are static (dP=0) and trivially predictable from the head's own P_t input. That is the aux-leak shape this team already wrote a rule against.
- **No verifier-free ablation in [RL²-VLA]** — every number is (injected diversity × trained 1B–7B verifier), never decomposed.
- **No attractor-error sensitivity, no wrong-attractor control, and no numeric σ_S in [OmniGuide]**.
- **No observation-frame vs action-frame decomposition in [EquiContact]** — the two are changed together, and we already ship the observation half.
- **No visual/R3M-regime ablation in [DARP]** — the regime where it claims larger gains and where we live. Fig. 6 is state-based Stack only, as an unlabeled bar chart with no numeric table.

**Seeds.** [HEP]: zero, everywhere, one run per cell. [EquiContact]: zero, n≤20 real trials. [Pri4R]: zero across RoboCasa Tables II and S4. [DARP]: grep for "seed" over the full HTML returns **zero hits**; all CIs are over rollouts of a single policy (Hopper 3545.57 ± 3.54 across 100 MuJoCo rollouts is not credible). [RL²-VLA] and [Cross-View] do report 3 seeds; [Cross-View]'s per-seed camera values are cleanly non-overlapping, which is why it grades B.

**Self-retractions in place.** [Mech-Study] §F.3–G.7 used full-layer residual hooks that break skip connections; with corrected MLP-targeted hooks, concept ablation gives **p=0.975, mean Δ = +3.3%** — invalidating the kill-switch narrative, feature-3259, step-0 criticality, and the temporal profile. Main-text Fig. 4's 28–92% zero-effect range never states its hook protocol. [Cross-View] App. G is a self-documented failure of its own first-choice intervention class.

**Statistical pattern worth naming.** [RtS]: no comparison in the paper reaches p<0.05 unpaired, including the flagship (Fisher p=0.25 on 462/500 vs 472/500). [RL²-VLA]: RL-vs-BC is +2.0 against ±3.0 std at n=3. [DARP] Table 11: validation-loss selection collapses the absolutes 4×. **The consistent shape across this literature is small real effects reported at max-over-config, with the model-selection protocol doing more work than the mechanism.**

**Our own inference discipline.** Three dossiers treated "93% of failures end >15 cm away" as evidence of live multimodality at approach. It is an inference from an aggregate distance statistic, and A2 costs two hours. Do not let it propagate into the Run-3 pre-registration unmeasured.

---

## Calendar fit

**Hard constraints.** Freeze ~Oct 10, submit Oct 16 — but the team's own plan freezes the **submission entry ~Sep 20**, i.e. **31 days from today**. One retrain slot. One 24 GB RT box, currently running the stage-2 RLPD curriculum with cold scenes at 0.00 and the frontier-demotion fix unrun. Eval budget (~1 A5000-hour per full rollout on the contended box) is the binding constraint, not the trainer. `BDDL_CENSUS` (c33a470): grasp/place gates 79 tasks; skill #2 unlocks 46. q counts all 100 tasks at episode end; unattempted = hard 0.

**Rule: nothing in this report may consume RT-box hours the RL skill needs.** A0, A1, A2, A4, A5 are minutes-to-hours of CPU/inference and fit inside week 1 without touching the skill's queue. A6 is reading. **A3 is the only item that rents an A100**, and it must return its go/no-go *before* the Run-3 mix freezes.

| Window | Research arm | Funded main line (concurrent) |
|---|---|---|
| **08-20 → 08-26** | A0 (today), A1, A2, A4, A5; A6 triage reads | Frontier demotion to stage 0 on cold scenes; first-success bootstrap |
| **08-27 → 09-02** | A3 loss-floor pilot + deranged-pair control; amend and freeze the three-tier liveness bar | Stage-2 continuation; harvest successes, prefix-truncated |
| **~09-03** | **Run-3 mix decision.** Only four inputs move it: A1's aliasing rate, A2's approach-diversity number, A3's floor-crossing, and the skill's cold-scene first-success rate | — |
| **09-05 → 09-15** | Run-3 retrain (the one slot) | Skill training continues on RT box |
| **09-20** | Submission entry freeze | — |
| **09-20 → 10-16** | Eval only — the binding budget | — |

**The decision that should be pre-committed now.** If A3 fails to cross the loss floor *and* cold scenes still do not convert, the highest-EV use of the one retrain is **breadth — expansion-task recipe replication — not channel revival**, because unattempted tasks score a hard 0 across all 100. That is already the team's stated priority order (`SPATIAL_INTEL_RESEARCH_2026_08_09.md`, Weeks 6–8: *breadth > depth-aux attribution > anything promoted*), and **nothing in these ten dossiers is strong enough to displace it.** Ten `watch` verdicts, zero surviving `adopt_now`, and the single strongest mechanism in the set — the ‖a−a′‖²/4 loss floor on manufactured counterfactual pairs — is one we can derive and test ourselves for under $150, without citing any of them.
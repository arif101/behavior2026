# PwC deep research sweep (2026-08-16)

Six parallel skeptic-verified dossiers via the pwc CLI (huggingface/pwc-cli, installed at
~/.local/bin/pwc) + arXiv full texts: Openpi Comet (2512.10071), VLA-Corrector (2607.01804),
Qwen-RobotManip (2606.17846), the 4D/WAM crop (X-WAM 2604.26694, MVISTA-4D 2602.09878,
ABot-M0.5 2607.00678, ImageWAM 2606.19531 + skims), RL-on-VLA (RL²-VLA 2607.26991, SRPO
2511.15605), and a 30-query breadth sweep. Discipline: headlines verified against papers'
own tables; max-over-config flagged; missing ablations named; grades A-C. Context: run
same-day as the probe-baseline triptych (commit mass 0.00 / liveness DEAD / key-causality
IGNORES at both bands — the Run-2 ckpt is a pure visual-context policy).

## 1. Landscape verdict

The field has converged on our two theses from independent directions. (1) Our dead-channel
finding is now a *named* phenomenon — BayesianVLA (ICML 2026) calls it "Information
Collapse": conditioning predictable from pixels loses conditional MI with actions and the
policy degenerates to vision-only — with a training objective to prevent it. (2) Our
flywheel is a published, cited blueprint — PLD (ICLR 2026, 61 cites): frozen VLA + residual
RL specialists at failure states + distribution-aware replay + distill back. We are not
betting on a thesis; we are replicating a validated pattern with a better simulator.
Meanwhile the WAM camp's own ablations replicated the test-time-imagination null (MVISTA's
skip-the-backprop variant: −0.1 to −0.5pp), and its best citizen (ImageWAM) wins by
deleting the video mechanism. No-WAM verdict: strengthened.

## 2. Adopt now (ranked by evidence × leverage / cost)

1. **Serving/training resolution 224² → 720/480 (Comet).** Measured 0.30→0.60 on
   turning_on_radio, same π0.5 base, same benchmark — the largest single lift in either
   2025 solution. Our uniform 224² downsample (button ≈ 5px) is the exact gap. → Run-3
   input arm #1: native-res fine-tune; affordance-centered foveation stays as the
   compute-matched variant. First step: verify the fork's resize path (hours). [B, same-task]
2. **Reward-weighted (AWR-style) distillation loss in the flywheel (SRPO offline).**
   Progress-weighted advantages over successes AND failures instead of success-only
   cloning: 17.3→88.7 LIBERO-Long offline; +20–27pp absolute on 6.7–13%-base real tasks
   from ~50 trajectories/task — the best published evidence for distillation from OUR
   regime. Cost: a loss change; our ground-truth rewards already exist. [A−/B]
3. **Anti-collapse objective for the key channel (BayesianVLA) + counterfactual point
   labels (CAST).** Dual-branch conditional-PMI objective needs no new data; CAST
   validates the counterfactual-pairs plan. → Run-3 conditioning arm alongside the
   asymmetric-dropout/point-decode package. [B+; deep-read before implementation]
4. **PLD's distribution-aware replay** for flywheel data curation (hybrid rollouts biased
   toward base-policy-visited states). Deep-read queued; steal the replay design. [B+]
5. **Probe control at 20/32 denoise steps (Qwen-RobotManip).** Their conditioning was
   invisible at 4 Euler steps, paid at 10–20; ours sampled at 10. Rerun liveness/commit
   probes at 20/32 — eval-only, hours. If still dead at 32, diagnosis is bulletproof. Also:
   their structured-schema prompt BEAT soft conditioning (65.9 vs 61.2, with no-cond 62.7)
   → a structured-token point-injection arm; NOTE the direct conflict with the banked
   AdaLN-77.5-vs-text-38.5 result — per-arm probes adjudicate, not priors.
6. **LVM monitor (VLA-Corrector) as the uncertainty trigger.** 75% of their gain is
   detect-and-truncate; OGG is within-support (authors' own words) — skip it. Offline
   smallest experiment: train the 40M latent-delta MLP on our logged rollouts, AUROC of
   deviation-score vs episode failure on replay, bar ≥0.75 (~1 eng-day + GPU-hours).
   Candidate REPLACEMENT for the unbuilt denoising-variance readout at lower integration
   cost. Portable today for free: median+MAD online normalization of our gate thresholds.
7. **RL²-VLA latent-informativeness kill-test (1 h).** Dump action-expert latents e_t at
   the 25 frozen probe states ± point displacement; if variance ≈ 0, that is a 4th
   independent memorization-bank measurement and prunes all latent-conditioned sidecars.
   Only if latents live: optional 1–2-day arm (latent-conditioned QAM/BC head, velocity
   composition w∈{0,0.25,0.5} at the commit step, ground-truth selection). MEDIUM
   support-death risk, kill test built in. Does NOT replace distillation.
8. **Comet config confirmations + assets.** We already match every measured cliff
   (full-chunk receding horizon, absolute joints, 30 Hz, proprio-in — each 0.30→0.00 if
   wrong): our serving config is exonerated; resolution and data are the live deltas.
   Depth-as-policy-input HURT them (0.30→0.20) — consistent with our dead point channel.
   Their public comet-1.5k RFT dataset + pi05 checkpoints: CHECK 2026 RULES, could
   replace weeks of rollouts. Checkpoint-union headroom (their 0.611 union vs 0.345
   served): per-task checkpoint routing if rules allow.

## 3. Watch (triggers named)

- Endpoint-state prediction aux on own latents, toggle-off branch (ImageWAM+X-WAM refined
  spec) — Run-4 candidate, ~2–3 A100-days, after flywheel arms report.
- Per-subspace FM-loss + gradient-interference telemetry (ABot lesson, diagnostics-only) —
  fold into Run-3 telemetry now (zero cost); split loss weights only if measured swamping.
- π_RL (2510.25889, open source): exact log-likelihood RL for flow-π0.5, 320 parallel
  envs. Revisit if a rollout fleet is ever rented; post-deadline candidate.
- RTC + training-time RTC (PI lineage) — only if we move off full-chunk serving; Legato's
  seam diagnosis (spurious multimodal switching) matches our 5–14× measurement.
- MolmoB0T procedural demo generation — alternative cold-scene data source.
- HABC viability critic (rewards committing under uncertainty) — skill-reward idea.
- GIAVA / PEEK / AVR / FOVI — foveation design variants for the input arm.
- EBench (2606.18239) — per-capability diagnostic harness template.
- Self-Correcting VLA (2602.21633) — VLA-Corrector's nearest competitor.

## 4. Discard

- WAM builds, all variants (verdict strengthened by proponents' own ablations; nothing in
  the crop runs beside Isaac in 24 GB; video-prior confound unbroken in every paper).
- OGG gradient guidance (within-support, 25% of gain at 2.12× replan cost).
- Backbone switch to Qwen-RobotManip (weights explicitly unreleased; mixed per-axis wins).
- SRPO online arm (needs in-batch successes; structurally dead at 0–3% base; full-param
  GRPO on 7B at ~10⁵ episodes — years on our box).
- "Emergent error recovery" imports (one anecdote, grade C).

## 5. Evidence audit & corrections to our own records

- Common defects across the crop: no seeds/CIs (except VLA-Corrector real-robot and
  RL²-VLA), latency hardware unreported in 3 of 4 WAMs, self-built benchmark suites,
  max-over-config headlines (ImageWAM 83.1, QRM "Context", RL²'s +17.3 = max-over-tasks
  vs strong baseline).
- CORRECTION: DESIGN.md's "Comet 448² head-camera A/B" does not exist in any version of
  their paper — the real A/B is 224²→720/480. Purge the 448² note.
- NUANCE our "RL on 3B flow unproven" line: π_RL demonstrates it open-source at 320-env
  fleet scale; infeasible on one box, no longer unproven in the field.
- Meta: 30+ queries found NO 2026 publication on reverse-curriculum-from-restore-states at
  scale — our campaign method is under-published, not superseded. Paper opportunity.

## 6. Plan deltas (folded into V2/Run-3)

Run-3 arm list, priority order under single-variable discipline: (A) corrective-pair
distillation with AWR weighting [items 2+4] + FACT schedule (base recipe, not an arm);
(B) resolution 720/480 [1]; (C) conditioning-revival package: BayesianVLA-PMI +
counterfactual points + asymmetric dropout + point-decode aux [3]; (D) structured-token
point injection [5]. Pre-Run-3 (this week, no slot): probe controls at 20/32 steps [5],
LVM offline AUROC [6], latent kill-test [7], resize-path verify [1], rules check on Comet
assets [8]. Telemetry additions: per-subspace gradient logging [watch-2]. Skill/V2 spec:
unchanged (sweep found nothing superseding RLPD-from-restore-states; PLD/CAP validate it).

# Recovery Design (v1, 2026-07-16) — verified by deep-research sweep wf_382e086f (104 agents)

**Verdict: recovery is NOT a separate learned head.** It is three mechanisms at three layers we
already own. Every measured 2026 result on pi0.5-class flow VLAs supports this decomposition;
no surviving evidence supports a failure-triggered separate recovery policy (the one apparent
counterexample, CR-DAgger's residual head, is confounded with a force/wrench modality the
baselines lack).

## The three layers

### L1 — Recovery *skills* live in the base policy (corrective data engine — THE biggest lever)

Measured: RePO-VLA 20%→75% adversarial (pi0.5, arXiv 2605.09410); IG-RFT 18.8%→40% offline
→85% with gated interventions on 1731–2931-step real tasks (2602.20715); FlowPRO 93–99% real
bimanual (2606.05468). All keep the flow action head unchanged.

Recipe (all three ingredients required):
- **Advantage-weighted flow-matching RFT**: `exp(A/β)`-weighted FM regression loss, critic
  gradient stopped before the VLM trunk, chunk structure intact (IG-RFT).
- **Proximal anchor to the reference policy** — non-negotiable: unregularized Flow-DPO
  collapses pi0.5 to 13% (FlowPRO ablation).
- **Correction data from gated takeover**: metacog/critic-value STAGNATION triggers a
  sim-privileged expert takeover (IG-RFT's human blueprint, with sim expert substituting);
  store low-value pre-takeover states + corrective trajectory, mix ~1:1 with demos.
- **Value-conditioning token** (RePO-VLA): train on success/recovery/failure with a value
  scalar, serve at v=1.0 → within-chunk micro-recovery with NO online detector. Slots
  directly into our existing AdaLN conditioning pathway beside z_point/z_stage.

### L2 — Recovery *decisions* live in the director (programmed macros)

The entire measured test-time-recovery literature lives at this layer, over FROZEN policies:
B2FF milestone rewind +8.2pp autonomous / +17.7pp oracle, largest on LIBERO-Long 55.8→87.3
(2606.09258); VLA-Corrector chunk-truncate+replan +15.65pp on pi0.5 with FEWER policy calls
(2607.01804).

Macros (priority order): (1) **truncate-chunk + replan** at detection — never perturb
mid-chunk; (2) **re-ground**: re-query grounding head / re-open belief hypothesis set
(inflate Σ, drop P(exists) prior); (3) **retreat-to-nominal + retry** with per-goal-literal
retry budget; (4) **stage rollback** keyed to the ledger (Larchenko hysteresis rules).
Bound retries and use hysteresis: FAR's +17.6pp headline shrinks to +5.6pp once
retry-budget-matched — the budget is not the method.

### L3 — Recovery *triggering* is the metacog head (calibrated gate)

- SAFE-style probe (2506.09937, NeurIPS-25): 1–2 layer MLP/LSTM over pi0.5 internal features,
  2.3M params, 0.73ms — trivially inside the 24GB budget. Train on sim failure rollouts
  (free for us). Functional conformal band → rollout-level FPR ≤ α; **α is the explicit,
  reportable intervention budget**.
- Our extra features: stage_age_ratio, belief entropy, grounding confidence, stage-head
  distribution entropy.
- logpZO (FAIL-Detect, 2503.08558) = zero-failure-data fallback only — it overfits tasks in
  multitask settings (SAFE measured), wrong choice for 100 tasks.
- Plan for the **detection penalty**: realistic triggering costs ~9.5pp avg / 24pp on long
  suites vs oracle timing (B2FF) — always report autonomous-trigger numbers, and expect
  detection quality to matter MORE as horizon grows.

## Quantified pitfalls (each has a measured cost)

| pitfall | cost | fix |
|---|---|---|
| Naive fine-tune on correction data | −30pp below base (CR-DAgger) | AWR weighting + proximal anchor |
| Unregularized preference RL on flow | collapse to 13% (FlowPRO) | RPRO proximal anchor |
| Perturbation during contact phases | −15pp (IG-RFT ablation) | **stage head gates DART noise: free-space/approach only** |
| Take-over relabeling | ~55% vs ~100% for delta corrections (CR-DAgger) | gentle on-policy corrections; don't interrupt base execution |
| Uncontrolled retries counted as method | FAR +17.6 → +5.6 retry-matched | per-literal retry budget + hysteresis |
| Mid-chunk perturbation artifacts | — | truncate chunk + replan (reduces policy calls, VLA-Corrector) |

## Novelty position (medium confidence — absence-of-evidence across verified set)

The **closed calibrated detect→dispatch→recover loop is UNOCCUPIED**: SAFE/FAIL-Detect stop
at detection (SAFE explicitly defers behavior improvement to future work); RePO-VLA/FlowPRO
skip detection entirely; B2FF (closest occupant) has a single-signal uncalibrated trigger, no
belief state, no stage ledger, no partial-credit scoring, and loses 24pp on long horizons
under realistic detection — exactly the regime BEHAVIOR scores. Only ONE datapoint exists in
the literature on detection-quality→net-gain; charting that frontier (net gain vs conformal α)
is itself a contribution. Reviewer will say already-done: conformal FPR control, retreat-retry
macros, corrective RFT. Ours: composing them under hysteresis arbitration over per-arm stage +
belief heads at 10k–140k-step horizons, with the α-frontier measured.

## Build order & calendar fit

1. **Now–Aug**: L2 macros land incrementally in the director (truncate+replan first — cheap,
   frozen-policy, measured). Stage rollback waits on cofounder head (Aug 8).
2. **Sep (corrective engine, post-G3, eval fleet up)**: L1 — stagnation-gated sim-expert
   takeover collection → AWR flow RFT with proximal anchor; value token added to AdaLN.
3. **Sep 22 metacog gate**: L3 — SAFE-style conformal probe + stage/belief features; report
   net-gain-vs-α curve. Ablation: detection-free (v=1.0 only) vs gated dispatch — the
   literature has never run this comparison.

Cite/beat: RePO-VLA 2605.09410, IG-RFT 2602.20715, FlowPRO 2606.05468, CR-DAgger 2506.16685,
B2FF 2606.09258, VLA-Corrector 2607.01804, FAR 2607.01111, SAFE 2506.09937, FAIL-Detect
2503.08558.

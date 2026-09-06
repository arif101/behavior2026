# Deep research 2026-08-03 — 4D latents, initiation wall, torso, scaling (104 agents; 23/25 claims confirmed)

_Recovered 2026-09-06 from the operator's session notes (originally memory-only); the verified-claim summary as recorded that day._

**Deep research 2026-08-03** on 4D latents + initiation wall + torso + scaling. 23 confirmed / 2 refuted claims. [[project-run1-launched-2026-07-30]]

**RQ1 — INJECTION ROUTES (5 independent confirmations of our α-stall):**
- Concat/input-token = weakest route everywhere: GLaD early-fusion 84.0 vs hidden-state distillation 94.1; DIPOLE concat 0.69–0.87 vs cross-attn+dropout 0.94–1.00 (early fusion collapses to 0.00); VLA-4D concat 94.2 vs attention 97.9; **ATM (RSS24, peer-reviewed): early-fusion-ONLY 5.33% vs late-fusion-near-action-head 72.83%** → direct support for our AdaLN record as the primary delivery.
- Our α-stall has names: **"modality collapse"** (DIPOLE) and **"prompt-induced action shortcut"** (3DThinkVLA — phenomenon confirmed 2-1; their ablation TABLES refuted 0-3, don't trust).
- **G3VLA (June 2026): aux head decoding ray-coords + log-depth from the policy's OWN features, GT-depth supervised: pi0 84.6→88.1** — WE HAVE GT DEPTH; validates the planned depth-distillation aux with numbers. Also: ray embeddings as inputs = largest single component (−2.0 when removed).
- **DIPOLE fix: modality-wise dropout p=0.2** (mask RGB vs geometry branch per step) — cross-attention alone does NOT prevent collapse; add map-vs-image dropout to our anti-shortcut suite.
- Route must match consumer: G3VLA injection helps pi0 (+3.5) but is null on GR00T's cross-attn two-tower.

**RQ2 — INITIATION WALL (ranked, all high-confidence):**
1. **RFCL (ICLR24): reverse-curriculum RL from DEMO STATE RESETS** — only method solving precision contact tasks (PegInsertion/PlugCharger) from ≤5 demos where BC-from-demos = 0%. We own the simulator + HDF5PlaybackWrapper 0.5mm state playback = infra exists TODAY. Caveat: their results state-based; visual flow-VLA transfer is extrapolation.
2. **RaC: rewind-to-in-distribution + corrective segment; 1:1–1:2 recovery:corrective FRAME RATIO is load-bearing**; flow-matching policy (our family); 78.3% @5h vs ALOHA-Unleashed 75% @89h; beats HG-DAgger. Our scaffold_collect implements exactly this.
3. **ResFiT: residual off-policy RL on FROZEN BC base, sparse binary reward: 14→64% in ~134 rollouts (~15 min robot time)**; parameterization-agnostic (chunk/flow OK by design, untested empirically).

**RQ3 — TORSO: WB-VIMA (CoRL25, Stanford, BEHAVIOR robot family/Galaxea R1): autoregressive whole-body decoding (base→torso→arms conditioning order) is load-bearing — up to 53% real-robot drop without; +45% sim ablation.** Our flow expert emits all 23 dims jointly = exactly the un-conditioned structure they show fails on low targets. Run-3 candidate: torso-conditioned decoding; near-term: torso-targeted kicks + corrective data.

**BOTTOM LINE (research-ranked by q-gain/GPU-hour):** (1) initiation wall via RFCL state-reset curriculum + RaC recovery data (+ResFiT capper); (2) whole-body torso decoding; (3) re-route map delivery: GT-depth distillation aux + AdaLN conditioning; **STOP engineering prefix map tokens for visible tasks** (keep K=8+anti-sink as-is; memory's demonstrated venue = out-of-vision tasks per SOMA). Map data structure stays (feeds AdaLN + affordance + eval memory); its prefix-token DELIVERY is de-prioritized.

# Dossier: Temporal Forcing (arXiv 2608.30643, Ding et al., posted 08-31) — read 2026-08-20

4D representation alignment for VLAs. Post-dates our dynamic-hashmap sweep by 11 days;
directly extends its §2 liveness ranking with a THIRD measured mechanism.

MECHANISM: history pathway (K=7 frames -> frozen DINOv2+Q-Former gist tokens -> causal
transformer -> 16 tokens -> ZERO-INIT gated cross-attn into backbone) + training-only
alignment of the pathway's PRE-GATE latents to StreamVGGT causal-4D targets (state +
change + readout terms; change term cancels window-constant content). 4D model + heads
deleted at inference; +13ms latency.

KEY NUMBERS: LIBERO 96.6->98.8; RoboTwin 53.5->62.8 (Handover Block 0->44 — occlusion);
real hidden-placement full-task 20.0->43.3 with isolated stages UNCHANGED (75->78/90).

THE ABLATION GOLD (Table 2 + Fig 6, controlled 10k protocol):
- Pathway w/ action loss only: gate 0.6e-3, inert, avg -2.5 BELOW base (= our Run-2
  dead channel, independently replicated WITH gate telemetry).
- w/ temporal alignment: gate 13.4e-3 (22x), gate-off at inference costs 5.4pp — live.
- Supervision elsewhere-but-not-on-pathway (row 4): 83.7, WORST — gate opens then
  collapses. Run-3 warning: supervise the injected pathway ITSELF.
- Framewise-3D targets ACTIVELY hurt the history pathway (aliasing lives in the target:
  temporal contrast 0.101 vs 0.491 causal). Temporally consistent targets load-bearing.
- History-blind predictor control: change-term 0.54 floor vs model 0.21.

WHAT WE ADOPT:
1. Gate-magnitude telemetry as standard training instrumentation (V2 skill + Run-3) —
   reference signatures: flat=inert / transient-collapse=abandoned / open-hold=adopted.
2. Run-3 liveness design: alignment supervision ON the injected pathway (pre-gate,
   readout-term trick) COMPOSES with counterfactual pairs (hashmap report fix #1) —
   content-creation + demand-creation. Our targets can beat StreamVGGT: GT sim state,
   GT depth, our own 4D map (staleness channel) are legal training-time targets.
3. Supersedes parked HAMLET-lite as the multi-task-era memory arm (better evidence,
   same cost class; HAMLET beaten 98.8 vs 97.6 in-paper).

CAVEATS: base is StarVLA-OFT (Qwen3-VL-4B, L1 MLP head), NOT flow-matching pi0.5 —
transfer to our expert untested; no seeds/CIs; no reproduction yet; code unreleased.
Does NOTHING for commit-mode absence (their isolated stages unchanged) — RL skill
remains the terminal-value manufacturer. Grade B+ (best ablation discipline this cycle).

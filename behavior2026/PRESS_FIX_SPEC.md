# Press-fix retrain spec — button target + stage-gate + temporal history

Evidence (Run-3 n=25): data-only ablation plateaus. A0 grasp 0/25 (hard floor); A4 (+factory
+episodes) ~2-3/25 real film-verified grasps but sub-threshold; 0 task success all arms. Root
causes of no-press: (1) the button is never in the policy's conditioning (no target in obs/data/eval),
(2) Markovian oscillation (grasp -> release -> re-grasp; the policy can't tell from the current
frame that it already grasped). Grasp is repaired-but-weak; press is architecturally blocked.

Architecture facts (confirmed in openpi_fork): PaliGemma backbone -> prefix KV (images+lang+state
+injected target_points). Separate gemma_300m ACTION EXPERT attends into that prefix KV to denoise
the 32-step action chunk. KV is rebuilt per-call from the CURRENT frame, not carried -> Markovian.
A `stage_head` exists but is a PROGRESS REGRESSOR (stage = episode progress in [0,1], predicted from
the current frame) — a clock the action expert does NOT condition on. Dead for sequencing.

## The three fixes (do together — they are one mechanism, not a ladder)

### F1 — Button/toggle affordance target  [PERCEPTUAL PREREQUISITE]
The action expert can only act on what's in the prefix KV. Put the button there.
- Sim knows the toggle link (radio.states[ToggledOn].link) — auto-generate a PRESS target point
  (toggle-link position + approach normal) with no hand labeling.
- Add a second target channel: target_points already carries 2 rows (L/R EE displacements to the
  RADIO). Add a press-target row/channel = displacement to the TOGGLE LINK.
- Apply in BOTH: (a) the manufactured training obs (re-pack the parquet with the press target),
  (b) the AffordanceMapFullRes eval wrapper. Policy must SEE the press target in training to use it.
- Needs the sim/factory box (freed by stopping A2).

### F2 — Real stage-gate  [reuse the stage_head hook, make it functional]
Turn the dead progress clock into a behavioral switch the action expert conditions on.
- Relabel: replace scalar progress with a CRISP discrete stage {approach, grasp, transport, press}
  in the parquet 'stage' column (derivable from grasp-weld state + ee-button distance in the
  manufactured episodes — auto-labelable).
- Wire it as INPUT, not just an aux output: feed the stage as a token in the prefix (so the action
  expert attends to it) AND keep the aux prediction loss (governing law: a channel with no
  supervising loss stays dead). Stage gates which target the policy follows: grasp-target in grasp
  phase -> button-target in press phase.
- Code change in pi0.py / the b1k policy; no new box.

### F3 — Temporal history pathway (Temporal Forcing)  [DE-MARKOVIANIZE]
Let the action expert attend over MULTIPLE frames' KV, not just the current one — this is exactly
"KV across steps". Fixes the oscillation (policy remembers it already grasped) and is the substrate
for the self-improvement flywheel (a policy that tracks its own progress + history completes more
autonomously -> generates more honest full-task episodes -> better data).
- Zero-init GATED history pathway: retain last K frames' prefix KV; the action expert attends to
  them through a gate initialized at 0 (starts as a no-op, does NOT break the pretrained Markovian
  weights) — the model LEARNS to use history where it helps.
- 4D alignment loss supervises temporal/spatial consistency of the cross-frame features (the channel
  gets a supervising loss so it doesn't stay dead).
- Code change; heavier than F1/F2 but on the same retrain.

## Plan
1. A4 done -> stop eval (A2 skipped, restartable later for the formal record). [in progress]
2. F1 button-target data + wrapper (sim box, ~today) ; F2/F3 model code (parallel, no box).
3. Retrain from Run-2 init on the idle A100 (add_stage_labels + button target + gated history).
4. Re-eval on the button-aware harness. Success = a hand drives <5cm to the toggle and turns it on.
5. Strengthen grasp in the same mix (grasp is only ~10%; it caps success even once press works).

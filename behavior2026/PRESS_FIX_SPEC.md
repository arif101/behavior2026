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

## Full-stack build (one retrain, per user 2026-09-12) — status
Target: stage-cond + progress-cond + temporal forcing + granular stage-dependent targets, ONE retrain.
- [x] F2 stage_conditioning: config flag ON + b1k_policy derives stage_tokens from 'stage' (was zeros=no-op). CODE READY.
- [x] progress_conditioning: NEW flag; sinusoidal scalar -> zero-init MLP -> adaRMS (mirrors point/stage); progress already plumbed by the stage_head branch. CODE READY (config pi05_radio_press has it on).
- [ ] Granular targets: grasp-stage -> radio centroid (from captured objpose_radio), press-stage -> togglebutton metalink. Switch target_points by stage in the data repack + eval wrapper. (Optional upgrade: a grasp affordance if the radio lacks a grasp metalink — needs a sim check.) DATA/CPU + wrapper.
- [ ] Temporal forcing (the crux, multi-day): 
      * Observation: add history field (last K frames' prefix features/tokens) + serve-side ring buffer.
      * pi0.py: retain past prefix KV; add a ZERO-INIT GATED cross-frame attention pathway so the action
        expert attends to history KV alongside the current-frame kv_cache (line ~593). Zero-init gate =>
        no-op at start, safe graft onto the pretrained (A4) checkpoint; model LEARNS to use history.
      * 4D alignment loss: supervise temporal/spatial consistency of the cross-frame features.
      * compute_loss + sample_actions: thread history through both train and serve.
      NOTE: this is real transformer surgery + needs its own smoke + likely a short validation run;
      it is the item most likely to need iteration. A100 stays DOWN until the whole stack smoke-tests.
- [ ] Combined config: add temporal-forcing + granular-target flags to pi05_radio_press when built.
- [ ] Smoke-test each component (build model, dummy forward, verify zero-init no-op + finite) BEFORE the retrain.

## FOLDED IN (2026-09-12): approach-clip re-manufacture — attacks grasp Bottleneck 1
Weak-grasp root cause (A4 n=25): 12/25 reach <=0.18m but only 2 grasp (17% execution), and 9/25
STALL >0.45m (approach fails). Two bottlenecks: (1) unreliable approach [void approach clips + far
affordance error], (2) poor grasp execution [button-as-grasp-target -> fixed by granular target].
- Fix applied: factory_approach_cap_v10.py now LOCKS torso joint 4 (action dim 6, A_TORSO pos 3) at 0
  after each captured trunk command/hold -> 10-DOF demo convention (arm7 + torso1-3). Backup:
  factory_approach_cap_v10.py.pre_joint4fix. This kills the A5 divergence (degenerate-std joint 4).
- TODO (sim): re-run the factory to regenerate the 12 approach clips (now 10-DOF), convert with the
  granular stage-dependent targets (grasp->centroid, press->togglebutton metalink) + stage/progress
  labels, un-quarantine, add to the mix. Then the single retrain trains on approach+factory+episodes+map.

## Radio metalinks (settles grasp target): only 'togglebutton' exists (no grasp metalink).
=> grasp-stage target = radio CENTROID (from objpose_radio); press-stage = togglebutton (radio+P_OFF).

## 2026-09-13 — approach data: torso joint 4 stays LOCKED; the reach comes from the BASE (v11)
- Joint 4 is exactly 0 in all 430k demo frames (the demos flex joints 1-3 only), so the policy never needs
  it; the old approach clips used it only because the factory's 11-DOF servo found reach there. Keeping the
  lock, the reach shortfall is covered the way the humans covered it: `factory_approach_cap_v11.py` drives
  the holonomic base along the pull line to a 15 cm standoff (BCAL in-place calibration, closed loop,
  honesty-tracked, aborts if the radio moves), runs the unchanged staging/servo/closure/weld, then drives
  back to the demo's stance with the radio in hand before the demo's transport replay. The drive IS part of
  the captured clip (base actions + base_pose recorded).
- ROOT CAUSE of "the base cannot be driven in playback": the OmniGibson playback wrapper hard-codes 1000 Hz
  stepping; `patch_playback_freqs.py` (honours OG_PLAYBACK_REAL_FREQS=1) was NOT applied on the new box.
  Measured: demo action replay moved the base 0.009 m over 40 frames vs 0.252 m recorded; after the patch
  0.257 m (exact). Every restore-and-control script assumes the patch; simbox_bringup_v6.sh now applies it.
  The 09-13 snapshot bank was built WITHOUT it (settle checks time-dilated) and is being rebuilt.
- d10 (pull 1.04 m, previously SKIP_REACH): v11 drive 1.04 m in 131 steps, radio displacement 0.000,
  hand->grasp-pose gap 0.99 -> 0.135 m at the standoff. Sweep yield pending.

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

## 2026-09-14 — v11 sweep #3 (ORIENT-FIRST): why the 09-13 resweep yielded 1/38
- Sweep #2 (v11, orient at the 10 cm staging point) saved only d20. Films: every base phase was clean
  (drive-up, corridor retreat/advance, drive-back), but the ORIENT phase — rotating the wrist to the certified
  grasp attitude while parked 10 cm from the grasp point — swept the gripper through the handle and flipped
  the radio before the approach began (d110), and that phase was NOT honesty-tracked, so the clip was only
  caught later by the weld gate. Back-resting / heavily rotated demos (d10, d160 ORIENT abort at 147 deg)
  are a separate, still-open limitation of the body-frame canonical grasp.
- Fix (`box_scripts/rebuild/make_v11_orient.py`, post-processes the generated v11): ORIENT runs FIRST at
  the 20 cm base-landing point where the hand is clear, with honesty tracking + RADIO_TOUCHED abort at
  1.2 cm inside the rotation loop; then STAGE to 10 cm; RESTAGE shortened 60 -> 30 steps and now
  honesty-tracked too (it was missed by the servo's tag filter). Phase order is now
  DRIVE -> RETREAT (if near) -> ORIENT -> STAGE -> RESTAGE -> APPROACH -> PUSH/ALIGN -> weld -> DRIVE_BACK.
- Sweep #3 launched 19:18 UTC on the RTX box (two sims, ~16 h); previous outputs archived under
  `/root/sweep_archive/`. Success bar: >= 12 honest clips (the original recipe's yield) — below that the
  attitude-aware grasp becomes the blocking item, not the approach.

## 2026-09-14 — ROOT CAUSE of the "reach shortfall", the base drive-up and the back-resting radios: the TEMPLATE-POSE restore bug
- All 38 factory demos start with the radio UPRIGHT on the table (same 3 deg tilt in every demo; decoded from
  frame 0 of the raw hdf5), at a per-demo sampled position/yaw. No demo begins with the radio on its back.
- OmniGibson's recorder stores only AWAKE objects per frame; the radio is asleep until the human touches it,
  so it is absent from every pre-grasp frame. `restore_to_frame` (reverse_curriculum_collect.py) loaded only
  state[t], never state[0], so the radio stayed at the SCENE-FILE pose = the task template's fixed pose
  (3.464, 4.886, 0.534) while the robot was posed for the demo's sampled spot. Proof: the factory's "pull"
  (radio at t_pre vs t_post) equals the template-vs-sampled offset within 2 cm on 37/38 demos (median 4 mm),
  and "rig ROTATED the radio N deg" equals the template-vs-sampled yaw. Offsets 5 cm .. 1.12 m, yaw 1..178 deg.
- Consequences: the "pull line" the base drove along was the template offset (d10's 1.04 m drive); SKIP_REACH,
  the torso-joint-4 reach debate and the original recipe's "high-yaw instances did not generalize" were this
  bug (its 12 successes were the demos whose sampled pose happened to sit near the template). The
  back-resting radios were OUR restore: for small-offset demos (d40 8 cm, d160 15 cm, d20 36 cm) the template
  radio landed against the restored robot's hand and physics kicked it over before the honesty reference was
  taken (inferred from films + offsets; the template mismatch itself is measured).
- FIX (commit with this note): `restore_to_frame` now loads state[0], then every frame <= t whose state
  size exceeds the robot-only baseline (an awake object), then state[t]. Verified
  (`box_scripts/rebuild/probe_restore_radio.py`): d20/d40/d10 restore the radio at the sampled pose to
  0.000 m, upright (3.0 deg), hand->radio 0.11-0.14 m, zero motion over 30 settle steps; 3-5 s per restore.
  Every restore-and-control script imports this function (factories, snapshot bank, skill_env_wrapper),
  so the snapshot bank and the HF `b1k_radio_approach` (A5) clips were built on the template pose too.
- Sweep #4 (v11 ORIENT-FIRST + fixed restore) launched 20:13 UTC; sweep #3's partial outputs archived under
  `/root/sweep_archive/*_templatebug_partial/`. With the radio in the right place the v11 phases degrade to:
  no DRIVE (gap < 0.40), BASE_RETREAT 0.25 m, BASE_STAGE to the 20 cm landing point, ORIENT, STAGE, APPROACH.
- **v12 verified on d20 (21:23 UTC):** with the fixed restore the v11 corridor ("back off along the tines",
  body frame) pointed INTO the radio on an upright radio (tine axis is world-up: 0.42, 0.12, 0.90) -> d20 STAGE
  hit at it=4, 46 cm displacement. v12 corridor = the human's own approach line (grasp pose -> hand at the
  restored pre-grasp frame): d20 pre-contact displacement 2.2 mm / 1.1 deg (STRICT), first contact APPROACH:4,
  weld intact, carried to the post pose within 1.5 cm, transport replayed, clip saved (494 obs steps, 2589 cmds).
  Sweep #5 (v12) launched 21:23 UTC over the 38 factory demos.
- Blast radius of the template-pose bug, checked source by source: human demos (organizer obs; pose labels from
  OG's SEQUENTIAL playback, which loads frame 0) — clean. Grasp+transport factory and the 58 complete episodes
  restore at closure+6, where the radio is already awake in 38/38 demos — clean; the policy-relay 3/3 stands.
  Only pre-contact restores were wrong: the A5 approach clips (rebuilding), snapshot-bank pre-grasp entries
  (RL skill), and the Run-2 corrective corpus's pre-grasp starts (retired).
- **v13 = own grasp (22:12-22:36 UTC).** Sweep #5 (v12, d20-canonical grasp) failed d40 (12-deg approach, palm shoved
  the body 9 cm, no finger contact) and d50 (ORIENT 62 deg short: d20's hand attitude is unreachable on a radio
  yawed 135 deg). The canonical grasp existed only because the template bug made each demo's own grasp look
  non-rest. v13 uses the demo's own certified grasp (rel_p/rel_R/fingers at closure+t0off; --grasp canon keeps
  the old path): d40 and d50 both saved with 0.0000 m / 0.0 deg pre-contact displacement, fingers within 2 mm of
  the human's, welded, carried, transported. Sweep #6 (v13) launched 22:36 UTC with d10/d20 (v12) + d40/d50
  (v13) kept: 4/4 strict-honest so far.

## 2026-09-15 — Sweep #6 result: 20/38 clips (19 strict, 1 relaxed) vs the >=12 bar; conversion at the strict bar
- Yield by factory variant (all on the fixed restore): v12 d10/d20; v13 d40/d50; v13b d60(relaxed)/d80/d160-d210/d250/d270;
  v13d (since 04:17 UTC) d260/d310/d330/d340/d370/d390. Failures: 11 RADIO_TOUCHED (mostly pre-v13d: reached the
  position with 11-22 deg wrist error and twisted the radio), 6 APPROACH_FAILED (v13d stops short instead of
  twisting: d300/d380 attitude not within 0.10 rad, d350/d400/d410 stalled 13-55 cm out = reach), 1 WELD_FAILED
  (d70, pre-v13c PUSH). PRE 'pull' <= 2.7 cm on all 38 -> the restore fix held.
- Conversion (`approach_convert_v11.sh`, BAR=strict): 19 renders, 8,900 obs steps -> /root/b1k_radio_approach_v2
  -> depth labels -> sample weights -> HF arif101/b26-radio-manufactured:b1k_radio_approach_v2 (+ clip cmds/meta).
- Next: re-run the 18 failed demos under v13d (11 of them never saw it); the reach-stall class (d350/d400/d410) may
  need the base to close the gap (BASE_APPROACH triggers only when the position was not reached and best < 0.10).

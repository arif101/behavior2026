# Cold-scene discriminator analysis (Arif/Claude, 2026-08-14)

Data: skill_start_bank.json (stage-1 anchors) × raw demo state vectors at those frames
(31 train scenes; demo 10 was accidentally included and labeled cold — treat n as 30+1
contaminated; effect small). Labels from the stage-1 final report bands.

RULED OUT (uniform across warm/cold):
- Handedness: active_arm=left, holding_arm=left in ALL 40 demos.
- Anchor span: stage2→press median 100 steps in every band.
- (Suhas already ruled out button height: 0.744 vs 0.738 m.)

POSTURE SIGNAL — REAL BUT PARTIAL:
- Assumption-free per-dim scan of the 623-dim sim state at the stage-1 anchor:
  best-5-dim composite separates warm/cold at AUC 0.77.
- Top dims cluster in a coherent block (31-164) with PAIRED duplicates carrying
  identical stats (73≡148, 68≡124, 81≡152) — same physical quantity serialized twice;
  looks like the robot joint block (qpos + target/mirror). Suhas: map indices to the
  state layout you already decoded for the wrapper.
- Counter-examples exist (WARM demo 300 scores like a cold scene; cold 160 scores warm),
  so posture is PART of the story, not all of it. Remaining candidate: radio/button
  ORIENTATION (press-axis direction relative to base) — not testable from the state
  scan without the layout map; metalink_labels has radio world pose per frame if you
  want the direct test.
- Full script + per-demo scores: cold_scene_scan.py alongside this file.

IMPLICATION FOR STAGE 2: supports the episode-count chunk cap (cold scenes needing
practice get it) + suggests start-state augmentation: jitter restore postures toward
the warm-cluster manifold on cold scenes' early rungs (curriculum-legal, training-side).

# Stage-2 mid-pass note (Arif/Claude, Sat 22:15 UTC — for the pass report)

Read at 18/32 chunks. Two observations for the readout, one recommendation.

1. DOSE CAVEAT: the volume experiment under-delivered by ~2x. Chunk cap = 25 eps OR
   4,500 steps; cold-scene failures burn 300 steps each -> cold chunks terminate at
   15 eps (measured: every 0.00 scene shows eps=15). Cold scenes got ~1.5x stage-1
   volume, not 2-3x. Same step-cap bias class as last pass, one layer down. If the
   report claims "volume doesn't fix cold scenes," qualify the dose.

2. THE VERDICT HOLDS ANYWAY: zero successes across ~90 cold-cohort episodes means no
   reward signal at all — RL cannot bootstrap from pure failure at any volume. Cold
   scenes don't need more episodes at -25/-10; they need their FIRST success.

3. RECOMMENDATION: note that cold scenes have NEVER trained at stage 0 (press-5) —
   the full passes started at -10; stage 0 only ever ran as the demo-10 smoke (75% in
   31 min). The staged frontier driver (box_scripts/skill_frontier_driver.py) demotes
   exactly there: cold scenes -> stage 0 for first-success bootstrapping, warm scenes
   continue climbing, per-scene promotion on evidence. Second lever if any scene fails
   even at -5: posture-targeted start augmentation (AUC-0.77 finding).

Warm-side watch item: demo 190 dropped 0.75 (s1) -> 0.25 (s2); if it drops further on
the next rung, check whether its s1 mastery rode a narrow posture funnel.

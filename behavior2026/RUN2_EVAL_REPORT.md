# RUN-2 EVAL REPORT — per RUN2_EVAL_PREREG.md (frozen 2026-08-08)

Campaign 2026-08-10/11, A6000 sim box (38.147.83.29:28534), ckpt radio_run2@49999
(arif101/b26-run2-params), A2 serving arm (affordance points + live FoveatedMap geometry →
AdaLN, prefix map tokens zeroed), instance 301 (public_test index 0), default timeout
(3,225 steps). Zero deviations from the pre-registration; zero serving/eval errors across
30 episodes + probe.

## Primary
**0/25 conversions — FAIL band** (bar: ≤2/25 indistinguishable from run1b's 1/31).

## Approach telemetry (not a prereg metric; recorded for mechanism)
- Closest EE-target approach across the campaign: **left 0.021 m, right 0.030 m** —
  contact range against a 2.2 cm target. Run1b's signature was a ~0.35 m standoff.
- Affordance point injected on 75% of steps (60,315/80,601), conf_p50 0.68–0.82.
- Rollout 2: right wrist worked the 5–10 cm shell for ~1,200 steps (cyclic
  approach–withdraw–re-approach; per-chunk arm-command magnitude flat at 2.31±0.006).
- VERDICT: the Run-2 data recipe (corrective + RaC + splice + aux stack) moved the
  approach distribution decisively and the commit distribution not at all.

## Causality ablation (n=5, points masked, map live)
**No degradation — the "decorative" flag FIRES** (prereg guard). Masked arm: 0/5
conversions (equal), approach depth L 0.086 / R **0.011 m** (≥ A2's depth). The
affordance-point route shows no eval-time causal effect on approach or conversion in the
Run-2 serving context. Caveats: n=5; map_geo conditioning stayed live (this indicts the
point route specifically, not geometry conditioning wholesale); approach depth is the only
discriminating metric available at 0-conversion.

## Commit-mode probe (root-cause metric; identical script/demo/frame as July baseline)
Demo 10 @ f1100, 25 fresh-connection chunk samples on the frozen obs:
- trunk-drop min-excursion **p10 −0.004 / p50 −0.001 / p90 0.000 rad** (flat)
- **commit_mass = 0.00** at both 0.10 and 0.20 rad thresholds (baseline: 0.00)
- **diversity: collapsed** (p10–p90 spread 0.004 rad)
Probe's pre-registered decision key: mass ≈ 0 AND low diversity → **MODE ABSENT** (not
resampling-killed). Confident lock on a non-commit mode, not indecision.

## Secondaries
- Stage-head held-out accuracy: PENDING (trainer-side eval over map episodes 180-199).
- Stall rate vs run1b banked logs: PENDING (campaign action log archived,
  eval_run2_primary/action_log_end_offset.txt; run1b logs in foveated backup campaigns/).
- Depth-aux training-curve check: L1 curve met threshold during training; per
  SPATIAL_INTEL_RESEARCH_2026_08_09.md §2.6 this readout is NOT interpretable as evidence
  about precision behavior (label misalignment caveat, logged 08-09).
- Negative control (held-out BC MSE): PENDING.

## Decision (per the frozen scale-up rule)
Primary FAIL + commit-probe = 0 → **RECONSIDER RECIPE.** The splice-corpus premise (0.03 m
initiation demos seed commit mass under BC) is rejected as tested. Consequences:
1. ALL selection-class levers are support-dead at this state class (mode-sticky decoding,
   best-of-K, DSRL, chunk-critic reranking) — nothing to select among 25 identical
   non-commit samples. Arm-B v1 (decoding mode) DOWNGRADED accordingly.
2. Funded main line: **learned sim-RL contact skill** (delegated press primitive from
   restore-band states; needs no base successes — manufactures its own via sim reward).
3. Training-side complement (Run-3): FACT logit-normal noise schedule (mechanistic
   candidate for WHY BC failed to instill the mode from splice demos at τ<0.2 precision).
4. Data side: retain failed rollout segments (now provably the majority behavior class);
   splice-weighting revisit is no longer a credible standalone fix.
5. Expansion DATA prep proceeds regardless (per prereg; the gate governed recipe
   replication, which is now moot in its Run-2 form).

## Artifacts
- /root/eval_run2_primary (25 jsons + videos + wrapper_stats_final.json + action-log offsets)
- /root/eval_run2_ablation (5 jsons + wrapper_stats_ablation.json)
- /root/commit_probe.log (verdict lines; teardown crash post-measurement is cosmetic)
- Map/foveation renders: map_live.mp4 + montage (rollout-0 window)

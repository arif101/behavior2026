# RUN-2 EVAL PRE-REGISTRATION — frozen 2026-08-08

Registered while training runs (launched 18:10 UTC, no eval result of any Run-2 checkpoint
has been observed). Per the G1 lesson (early separation at n=5/5 regressed to null at full
n), nothing below changes after results start arriving.

## Serving config (primary arm, "A2")
A5000 harness, same wrapper stack as G1 arm A, plus: affordance-head points → AdaLN;
LIVE FoveatedMap geometry → map_geo AdaLN route; prefix map tokens ALL-ZERO (frozen route,
null-token convention — the configuration class that holds the only legal conversion).
Final checkpoint (step 49999). Norm stats = the Run-2 mix stats shipped with the params.

## Primary metric
Conversion rate (episodes ending q=1.0) on turning_on_radio instance 301, **n=25 rollouts**,
default timeout. Baseline: run1b arm A = 1/31 pooled (the 3-10%/rollout rare-mode band).
- **STRONG success: ≥5/25 (20%).** (Fisher exact vs 1/31, one-sided.)
- **WEAK signal: 3-4/25.**
- **FAIL: ≤2/25** (indistinguishable from the baseline band at this n).

## Secondary metrics (mechanistic; each maps to one component)
1. **Commit-mode probe** (root-cause metric): re-run the distribution probe at demo
   near-commit states (restore band f1100-f1600, OG_PLAYBACK_REAL_FREQS=1, 25 samples).
   Baseline 0/25 chunks with sustained trunk descent. Success = mass > 0.
2. **Stage head**: held-out 4-way frame accuracy on map episodes 180-199 ≥ 80%.
3. **Stall rate**: fraction of primary rollouts firing the v1 stall detectors within the
   first 1500 steps, vs the run1b campaign rate (computed from banked action logs).
4. **Depth aux**: training-curve check only — depth L1 < 0.15 log-m by 20k steps.

## Guards
- **Causality ablation** (n=5): serve A2 with points masked (null token). Expected:
  degradation vs A2. If no degradation, the conditioning is decorative — flag loudly.
- **Negative control**: held-out BC action MSE on episodes 180-199 within +10% of run1b
  (no base-task regression from the mix/aux stack).

## Scale-up decision rule (gates the RECIPE replication to the 4 expansion tasks)
- PROCEED (replicate recipe per task) if primary ≥ WEAK **and** commit-probe > 0.
- HOLD + diagnose if primary FAIL but commit-probe > 0 (data worked; serving/selection
  issue — chunk-critic becomes the next lever, not more data).
- RECONSIDER recipe if both fail (the splice-corpus premise itself is then in question).
- Expansion DATA prep proceeds regardless (already signed off) — this gate governs
  training-recipe replication, not data work.

## Hygiene
- First eval runs on the FINAL checkpoint only; any mid-checkpoint eval happens after and
  is labeled exploratory.
- Budget: 25 + 5 + 25-probe ≈ 30 A5000-hours.
- Everything above was fixed before the first rollout; deviations get logged in this file
  with timestamps, never silently.

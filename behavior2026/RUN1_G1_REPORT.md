# G1 REPORT — Run 1b campaign (2026-08-03/04, per RUN1_EVAL_PROTOCOL.md + Amendment 1)

31 rollouts, ckpt radio_map_run1b@36559, A5000 24GB, all-legal serving (affordance points +
online 3-cam map + dead-reckoned odometry; zero oracle signals). All per-run archives + videos
+ RaC clips backed up to `arif101/b26-foveated-backup-20260730` (campaigns/run1b_g1_campaign,
rac_clips_v1).

## Headline

**FIRST FULLY LEGAL SUCCESS: arm A run 7, q = 1.0, 1,906 steps** — approach, grasp-descent,
press, complete, with affordance-only conditioning. The initiation wall is **probabilistic,
not absolute**: the commit mode exists in run1b's distribution at a low per-rollout rate.

## Pre-registered table

| arm | n | conversions | formations (<0.30m/50) | dmin med | wristL med |
|-----|---|------------|------------------------|----------|-----------|
| A (all-zero tokens) | 9 | **1** | 1 | 0.543 | 80.0° |
| B (full map) | 11 | 0 | 0 | 0.502 | 85.4° |
| B0 (geometry-only map) | 3* | 0 | 1 | **0.337** | **29.2°** |
| C (full map + kick) | 6 | 0 | 1 | 0.498 | 83.4° |

*B0 n=3 with stats (n=4 results; one stats file lost to the logger surgery).

## Decision rules, answered

- **G1 PRIMARY — FAIL as pre-registered.** Formation(B) − formation(A) = 0/11 − 1/9 ≤ 0
  (needed ≥ +25pp). Wrist median(B) = 85.4° (needed < 60°). Both prongs fail.
- **Map decomposition — INVERTED vs expectation, n-limited**: B0 (geometry-only) posts the
  best approach median (0.337) and the two best wrist runs (28–29°); B (full map) never
  converted in 11. Tentative read: the GEOMETRY channels may help approach while the map's
  TARGET channel adds nothing beyond AdaLN (or mildly conflicts with it). n=3 forbids
  conclusions; noted for Run 2 design, not acted on as fact.
- **Kick read — collection-only, as pre-committed**: C ≈ B on conversions (0/6). BUT the
  scaffold works as a data engine: 6/6 interventions per episode, re-entry→progress on
  nearly every kick (runs 3–6: 6/6 clips kept), and C run 6 reached **0.114 m** — the
  campaign's nearest near-miss. **27 outcome-filtered RaC clips collected.**
- **Early-stop check**: the early B-vs-A dmin separation (p≈0.05 at n=5/5) regressed with n
  (B med 0.502 vs A 0.543 at full n) — a live lesson in why the pre-registered n mattered.

## Mechanism findings

1. **Conversion is a rare mode (~3–10%/rollout pooled), now legal.** Rate is compatible with
   the oracle-era band (1/6, 1/5). Formation→conversion ran 1/3 this campaign.
2. **Approach is solved across ALL arms** (every rollout descends 2.2m → 0.3–0.6m; the old
   checkpoint's parking/freezing is extinct). Run-1 training (oversampling + metalink AdaLN)
   did this arm-independently.
3. **Torso (full-vector logs, 11 runs)**: failures now dive to −1.45..−1.54 routinely (old
   ckpt: −1.0); one B run hit −1.80 (the old converting signature) without converting →
   torso depth is **necessary-not-sufficient**; the residual gap is the final coordinated
   descent, not raw posture.
4. **Online map fidelity**: self-corrects from stale 1.6m fixes to 0.18–0.23m held accuracy
   (≈ its theoretical floor: metalink-vs-center offset + odometry drift). Write-and-refine
   works; run-3-style conf collapses were bridged in-episode.
5. **q scoring pays nothing for proximity**: every 0.35m approach scored 0.0 — conversion
   rate is the ONLY leaderboard-relevant number on this task.
6. **24GB submission-class fit: PROVEN** across 31 full episodes (server+Isaac+DINO+map,
   zero skipped frames).

## What G1's failure does and does not mean

The pre-registered architecture bet (prefix map tokens → formation gain) is **not supported
on this task** — consistent with the deep-research verdict (concat = weakest route; memory
redundant at 90% visibility). It does NOT kill the map: the map's value case was always
memory-demanding scenarios (untested until multi-task), its fidelity is proven, and its
geometry channels show a (noisy) positive signal. Per the pre-committed fallback ordering:
corrective data (Run 2) > critic-selection > aux rebalance, with prefix tokens FROZEN.

## Run 2 mandate (updated by this campaign)

- **Amplify the rare mode**: base mix + 27 RaC clips (grow to ~100 via collection runs) +
  reverse-curriculum initiation clips from demo-state resets + the A7 success episode (gold).
  Composition per RaC paper: recovery:corrective ≈ 1:1–1:2.
- **Map→AdaLN motor path** (ATM 72.83-vs-5.33 route evidence) + GT-depth aux (G3VLA
  84.6→88.1) + modality dropout (DIPOLE). Prefix K=8 unchanged.
- **Serving config for Aug-12 leaderboard**: arm-A-style (all-zero tokens) is provisional —
  it holds the only conversion and proximity scores zero. Revisit if Run 2's B-arm converts.

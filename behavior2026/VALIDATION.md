# Validation Matrix — BEHAVIOR-2026 (v1, 2026-07-10)

Every milestone has: a **pre-registered decision rule** (written before the experiment runs), a
**power floor** (enough rollouts to see the effect — 5 rollouts cannot distinguish 0% from 10%;
0.9^5 = 59%), an **instrumented metric** (battery/proximity numbers, never eyeballed video), and a
**fallback** (what we do if the gate fails). Verdicts get banked with numbers, including negatives.

## Cross-cutting assays (run at every training milestone)
- **Trunk-health probe**: `probes/probe_bbox_decode.py` text-decode on the fine-tuned ckpt.
  Healthy = coherent VQA-style answers survive; garbage tokens = protection failed (we measured
  this exact failure on the organizers' baseline).
- **Prompt-sensitivity assay**: run matched vs mismatched prompts on 5 instances; a protected
  model must NOT be prompt-blind (organizers' ckpt: quarter-activity curves identical to 0.02
  across prompts = failure signature).
- **Behavioral battery trend** (fixed probe set, proximity-gated): engagement rate, verified
  target attempts, min-target-dist, spiral rate. These move BEFORE q does — q is a cliff metric.
- **Baseline anchor**: any surprising number gets compared against the organizers' reference
  checkpoint on the same instances before we believe it.

## M1 — Grounding pilot (now → ~Jul 14, A40 box)
- **Label validity**: visual overlay spot-check (10 frames/episode) PLUS quantitative gate —
  depth-at-projected-pixel matches projected depth within 5cm on ≥95% of non-occluded frames
  (catches extrinsics/frame-convention bugs mechanically).
- **Head validity**: median pixel error on held-out episodes AND on a shifted-instance split
  (train instances A, test instances B). Decision: shifted-instance error < 2× on-distribution
  error → green-light 300GB box + full-scale dataset. Fallback: fine-tuned SAM 3.

## M2 — Pointer benchmark at scale (Aug 4)
- Per-task table across all 100 tasks: within-object-mask rate + px/3D error, held-out instances.
- Gate: within-mask ≥80% on a supermajority of tasks; per-task failures enumerated (no silent
  truncation — a task with no labels is a listed gap, not an omission).
- **Eval-context verification**: run the grounder live during a few evals; T1 renderer overlays
  its heatmap on the video. Catches demo-vs-eval context drift that offline benchmarks miss.

## M3 — G3 point-conditioning gate (Jul 31, H100)
- 4 arms (task-ID / language-only / language+honest-point / language+oracle-point), ≥50 rollouts
  per arm on SHIFTED instances, SAM in all arms, replay arm as protection baseline.
- Primary: Δq AND Δ(proximity-verified engagement) oracle vs none. Decision ladder:
  - oracle ≥ +10pp → axis validated;
  - honest captures ≥50% of oracle gain → grounder is good enough, proceed;
  - oracle works but honest doesn't → fix the grounder, not the policy;
  - oracle fails → the conditioning interface is wrong (re-examine AdaLN wiring) or the
    failure is not where we think — run batteries before pivoting.
- Trunk-health + prompt-sensitivity assays on every arm's checkpoint.

## M4 — First full submission (Aug 21) — THE validity anchor
- **Mechanical**: organizers' Docker verification passes (24GB GPU, timeout compliance).
  Submitting 8 weeks before deadline exists precisely to de-risk this.
- **External calibration**: leaderboard q on public instances vs our internal predictions for the
  same instances. Any divergence = our harness differs from theirs → fix before trusting any
  internal number again.
- **Value-table check**: predicted per-task q vs leaderboard breakdown; re-fit eval economics.

## M5 — Corrective engine (teacher gate Sep 1, kill-check Sep 15)
- Teacher gate: privileged teacher recovers ≥50% from restored mid-episode failure states on gate
  tasks — below that, corrective data has no signal to distill.
- Kill-check (pre-registered): corrective-DAgger arm ≥ +0.03 q vs MATCHED-BUDGET control
  (same data quantity of plain success rollouts), ≥50 rollouts/arm. Fail → descope to
  success-rollout RFT (Success-Conditioning theory says that cannot degrade).
- TD-calibration head: Brier/AUROC vs episode outcomes on held-out rollouts + reliability diagram.

## M6 — Metacognition gate (Sep 22)
- ≥30% spiral-rate reduction (battery-measured) at matched success rate.
- Dispatch net-positive: q with dispatch ≥ q without, ≥50 rollouts.
- Conformal coverage: empirical failure-coverage within the declared α band on held-out data
  (stage-keyed; AC-RAC per-action validity for dispatch decisions).
- Baseline to beat: DTW sampling-disagreement gate (VLA-ATTC).

## M7 — Freeze (Oct 10)
- Dress rehearsal on dev instances 10–19 (never trained on, never tuned on).
- Full assay regression: trunk-health, prompt-sensitivity, grounder benchmark, battery trends.
- Submit best-of-validated configuration; the leaderboard history (Aug 21, Sep, Oct) is the
  progress curve that proves the trajectory.

## Standing rules (learned this week, enforced)
1. Decision rules are written BEFORE experiments run.
2. No conclusion from eyeballed frames — two visual "smoking guns" were retracted this week;
   instrumented metrics or it didn't happen.
3. Cheap decisive pilot before any multi-day compute commit (replay pilot killed a re-render
   plan in 40 minutes).
4. Oracle numbers are ceilings, never claims; honest numbers are the submission.
5. Every gate verdict — pass or fail — gets banked with its numbers.

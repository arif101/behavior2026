# STEP 0 — Demo ↔ eval-env gap, task-0062 `halve_an_egg` (2026-08-26, in progress)

Question this answers before any context-vector design: **do the 200 human demos replay faithfully
under eval-exact physics, and where they don't, what is the physical cause?** The prior thesis
(SYSTEM_PLAYBOOK §1) was that the teleop rig's ranged "magnetic" assisted-grasp makes demo action
streams illegal at contact-commit moments. Everything below is measured on this box (RunPod RTX
A5000, BEHAVIOR-1K v3.9.0, Isaac 5.1.0-rc.19) with `task62/env.py`, which mirrors
`omnigibson/eval/evaluator.py::load_env` line for line. Scripts: `task62/gate0_replay.py`,
`task62/step0_replay_sweep.py`, `task62/step0_closure_probe.py`, `task62/step0/demo_census.py`.

## 0. Facts from the raw data (no sim)

| fact | value | source |
|---|---|---|
| demo env config | robot `robot`, `grasping_mode: assisted`, 30 Hz action / 120 Hz physics, `action_normalize: false` — **identical to eval/r1pro.yaml** | hdf5 `config` attr, 200/200 |
| demo length | mean 6,476 / median 6,406 / min 3,497 / max 24,094 steps → eval budget 1.5× ≈ 9,714 | census |
| skill sequence | 181/200 share one 16-skill sequence; 11 add a `push to`; 4 drop a `move to` | annotations |
| rollbacks | 198/200 demos contain rollback branches, mean 30/demo, max 80 | hdf5 `rollbacks/` |
| serialized state | only ~12 frames/demo carry the full state (1418/1454 dims); later frames = awake bodies only → **sequential restore is mandatory** | `state_size` |
| annotation alignment | chop reward spike lies inside the annotated chop segment in 191/198 demos; annotation end ≤ raw T | census |
| reward stream | ±0.2/±0.4 **spike train** (Δq at the flip frame), goes negative before the slice in 83 demos; Σreward ∈ {−0.4 … 1.2}. **Not a usable q(t); true q must come from `goal_status` in sim** | census |
| arm at scoring events | chop / halves→plate: left arm in 198/200; knife→sink: mixed (L 119 / R 79) | census |
| frames per skill (Σ) | move to 376k · pick up 354k · place on 169k · open door 160k · close door 110k · place in 45k · chop 44k | annotations |

## 1. Gate 0 — replay fidelity (PASSED, 2026-08-25)

Restore demo 620010 sequentially (stride 5) to chop−100, replay its own 23-d actions open-loop.
**Slice fired at frame 3599 = the recorded frame (Δ = 0), q = 0.40** (`real(half_1)`, `real(half_2)`).
The restored state carried the knife in the left hand via native AG (`_ag_obj_in_hand`). Env boot
603 s cold / ~300 s warm; sequential restore 700 frames in 48 s; `env.step` median 598 ms
(pilot: 499–503 ms). GPU dynamics / scene reduction NOT attempted (rejected previously against this gate).

## 2. Replay-fidelity pilot — isolated, open-loop (3 demos × 11 segments)

Protocol: for each scoring-relevant annotated segment, cold-restore the recorded state at
segment−30, replay the demo's actions open-loop to segment+60, score the family predicate.
Post-slice segments: slice in-episode from chop−60, then cold-restore the segment anchor.

| family | replays faithfully | notes |
|---|---|---|
| open fridge | 3/3 | |
| pick egg from fridge (right arm) | **3/3** | closure 0.6–1.4 cm from egg center; native AG engaged 12–13 frames later |
| close fridge | 3/3 | |
| **pick knife from board (left arm)** | **0/3** | gripper closed 9–19 cm from knife *center*, no finger contact, AG never engaged |
| place egg on board | 3/3 | `OnTop` true, released |
| chop | 3/3 | slice within 0–5 frames of the recording |
| place knife in sink | 3/3 | conjunct 4 |
| pick half (left arm) | 2/3 | |
| **pick half (right arm)** | **0/3** | right hand empty at segment end |
| first half → plate | 3/3 | |
| second half → plate | 0/1 real | other two: 620010's human also ended at q=0.8; 620030 satisfied both conjuncts in the first window |

Files: `/root/step0/replay_sweep_isolated_620010_620030_620040.jsonl` (33 rows).
Infra verified in sim: (a) composition guard (`env.reset()` + purge `transition_rule_api.obj_init_info`)
restores the pre-slice object set; (b) **post-slice demo states cold-restore once the halves exist**
→ post-slice anchors cost ~75 replay steps (slice-then-restore), not a full chain.

## 3. Closure-state probe — the failing grasps, from the RECORDED states (demo 620010)

Restore the human's own state at closure−30 … +60 (no replay) and read AG, EE→object distance,
object height. Then a short replay from closure−30.

| grasp | d(EE, obj center) at closure | AG first appears | object lifts | short replay from closure−30 |
|---|---|---|---|---|
| knife, left, closure 2990 | 10.2 cm, **flat 10.1–10.4 cm before and after** (handle grasp, no pull-in) | +10…+20 frames | 0.93 → 1.03 m | contact +5, **AG +20, held through +60** |
| half, left, closure 4732 | 1.0 cm | +10…+20 | held | (probe restore bug — rerun pending) |
| half, right, closure 5134 | 1.0–1.1 cm | +10…+20 | 0.93 → 1.09 m | (same) |

Reading: AG engages 0.3–0.6 s after the close command with the hand at contact distance — the
signature of OmniGibson's **native** assisted-grasp rule (contact ∧ between-finger raycast ∧ 0.3 s
closure), not of an external `_establish_grasp` weld (instantaneous, at range, followed by pull-in).
The pilot replayed the knife closure at the same 9.3 cm radius and still missed → millimetre
misalignment at a thin handle after ~230 frames of open-loop replay.

## 4. Verdict so far

1. **The task-62 demos are eval-legal.** Every grasp examined was a physical contact grasp closed by the
   eval's own AG rule. The "magnetic rig" regime is not what breaks these demos.
2. **The replay failures are open-loop drift**, concentrated in grasps of small/thin objects on flat
   surfaces (knife handle, egg halves) after long approaches. Large-tolerance skills (doors, egg from
   shelf, placements, chop, knife into sink) survive hundreds of frames of drift.
3. Consequences for the plan:
   - The context vector's "assist-regime" channel has no supporting evidence on task-62; keep it out
     unless the overnight sweep finds a counter-example. Channels with evidence: family/stage, progress
     q(t) from sim, pre/post-slice composition, AG-held per arm, and a per-frame **anchor-quality /
     drift label** (how far a segment can be replayed before it diverges).
   - Demo relabeling must use **closed-loop or near-anchor replay** (playbook §2.6) — pending A/B below.
   - The hard families are the precision ones: second-hand grasps of halves, halves→plate, knife→sink.
     Reverse-curriculum starts for them are cheap (slice-then-restore).

## 5. Overnight sweep (2026-08-26 03:18 → 18:07 UTC) — results

**A. Closed-loop replay A/B (same 3 demos).** No help: the arm/trunk tracking error never exceeds the
0.03 rad tolerance, so no substeps are triggered on the failing grasps (knife closure at the identical
9.31 cm) — and where substeps do fire (620030 close-fridge) the changed timing makes things worse
(cascade to 3/11). **Controller lag is not the mechanism.** Files: `replay_sweep_isolated_closed_*.jsonl`.

**X. Drift attribution (620010 knife pick, replay from closure−{30,60,120,230}).** Robot base, yaw,
trunk, arm joints and EE position replay with **0.0 cm / 0.0 rad** error at every anchor. Only the
**knife's position** diverges: 0.06 / 0.06 / 0.37 / **3.58 cm**; AG fails only at the 230-frame anchor.
→ The robot is deterministic; **the object gets displaced** during long windows. (`drift_probe_*.json`)

**B. Open-loop isolated, 7 new demos** (620050/080/100/140/170/270/290; human q_final 1.0/1.0/0.8/0.4/0.4/0.4/0.2):
doors 14/14 · egg pick 7/7 · egg→board 5/7 · chop 7/7 · knife→sink 6/6 · **knife pick 5/7** · half pick
left 6/7, right 5/5 · half→plate 8/12. Every successful knife pick: AG +13 frames after closure at
8–11 cm from the knife *center* — the recordings' signature. One demo (620290) hit the post-slice
composition `KeyError` after reset (robustness item).

**Pooled over 10 demos (open-loop, isolated):** doors 20/20 · egg pick 10/10 · chop 10/10 · knife→sink
9/9 · egg→board 8/10 · **knife pick 5/10** · half pick L 8/10, R 5/8 · half→plate 11/18 (several of the
misses are scoring-window artifacts or match the human's own outcome).

**C. Continuous full-episode replay (620040).** Exact through open-fridge / egg pick / close-fridge; the
knife pick misses at ~2835 and everything downstream cascades (q peaks 0.4 when the hand pushes the egg
into the *free* knife on the board — `chop` is non-discriminative — then drops to 0.2). **Full-episode
replay is not a labeling strategy; per-segment restore is.**

**D. Random-policy floors** (uniform joint noise on the active arm from the segment−30 anchor, 3 demos × 3
trials): chop 0/9 · egg pick 0/9 · knife pick 0/9 · egg→board 0/9 · **half pick 8/18 (44%) · half→plate
5/18 (28%) · knife→sink 3/9 (33%)**. The three high floors are anchors where the hand already sits over
the target; RL bars for those families must use deeper anchors or be floor-adjusted (radio audit lesson).

**E. Closure-probe rerun (half grasps, restore-order fix).** Both replay from closure−30: finger contact
+5, **AG +20**, held through +60. Recorded closure distances 1.0–1.2 cm. Honest grasps.

## 6. Verdict (final for Step 0)

1. **The task-62 demos are eval-legal.** Every examined grasp is a contact grasp closed by OmniGibson's
   native assisted-grasp rule; the actions are the ones eval physics executes. The "magnetic rig" thesis
   does not apply here.
2. **Replay failures are object-displacement drift over long open-loop windows**, not robot drift and not
   controller lag. They concentrate on the knife (thin handle on a flat board) and, less so, the halves.
   Per-segment replay from anchors ≤120 frames before closure reproduces the human outcome.
3. **For the plan:** (a) demos are a reliable training source as recorded — the VLA never replays; (b)
   labels that need sim (q(t), object poses) must come from **per-segment near-anchor replay**, never
   continuous replay; (c) a context vector needs no "assist regime" channel — use family/stage, progress,
   composition, AG-held, and an anchor-quality label; (d) the hard problem is policy *commitment* on the
   precision families (half→plate, knife→sink, second-hand grasp), and their RL bars need floor-adjusted,
   deeper anchors.
4. **Object-trace probe (answered):** after the 2760 restore the knife is stable in a pure settle
   (0.06 cm / 60 physics steps — no restore artifact), but during replay it is pushed continuously at
   0.3–0.6 cm/frame across frames 2857–2866 (4.2 cm total, ~130 frames before closure) — the left
   arm's reach brushes the handle. The anchor at 2760 is **mid-motion** (inside the preceding `move to`);
   restoring positions without the in-flight transient perturbs the early reach path by millimetres.
   Anchors ≤120 frames before closure (after the reach has started from a settled pose) succeed.
   **Relabel rule: anchor at quasi-static frames** (the radio start-bank rule), never mid-motion.
   Robustness item: the post-slice composition `KeyError` on 1/10 demos (620290).

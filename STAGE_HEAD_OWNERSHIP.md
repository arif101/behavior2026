# Temporal / Task-Progress Head — Ownership Spec (v1, 2026-07-15)

Owner: cofounder. Scope: the complete temporal workstream — stage-label extraction at scale,
the learned per-arm stage/progress head, and the literal-ledger (predicate-satisfaction)
estimator. Self-contained until integration; touches no serving fleet or GPU sweep.

## Mission

BEHAVIOR tasks are long-horizon (30–75 min episodes, 10k–140k frames) and scored by
final-state partial credit per BDDL goal literal. A reactive policy compounds errors over
thousands of steps and never knows *what subtask it is in* or *what is already done*.
This workstream gives the system task-progress perception: for each arm, which goal
literal it is driving toward and what phase it is in — plus a live estimate of which
literals are already satisfied. Measured motivation: the organizers' scaled π0.5 baseline
fails precisely for lack of this structure, and the BEHAVIOR-2025 winner's single biggest
lever was a stage head + stage-keyed voting (~2× on long-horizon, pure BC).

## The one-sentence architecture

A task is a set of goal literals; **every subtask drives one literal toward true**; the
phase is progress toward that transition (approach → contact → effect-the-change →
complete). Relocation is the special case where the watched variable is object position;
`open` watches a joint angle, `toggle_on` an attribute, `fill`/`clean` particle counts.
The literal itself names the variable to watch — that is the label-generation trick.

## What already exists (starting assets)

| asset | where | state |
|---|---|---|
| Validated pilot extractor | `behavior2026/stage/extract_stages.py` (+ box `/root/stage/`) | works on tidying_bedroom: 3/3 objects, 8/8 phase boundaries visually verified |
| Per-frame object positions (all swept tasks) | box `/root/sweep_labels/<task>/` + HF `arif101/behavior2026-artifacts` `sweep_100/sweep_labels/` | growing (~25 tasks now → all ~92 by ~Jul 19) |
| Proprio per frame (LeRobot parquet) | `sweep_out/<task>/b1k/<task>/data/` (box + HF) | R1Pro `observation.state` 61-dim: eef_left_pos 17:20, eef_left_quat 20:24, gripper_left_qpos 24:26, eef_right 42:49, gripper_right 49:51. eef_pos is BASE-frame → world via camera matrix (see extractor). |
| RGB-D videos (head + wrists) | `sweep_out/.../videos/` (HF after offload) | training frames for the head |
| Task semantics | `behavior2026/task_targets.json` (+ extractor `probes/extract_task_targets.py`) | targets/references per task, 100/100 parsed |
| BDDL goals | `BEHAVIOR-1K/bddl3/bddl/activity_definitions/<task>/problem0.bddl` | goal literals, forall/or structure |
| Calibrated head-cam intrinsics | `behavior2026/probes/zed_intrinsics_calibrated.json` | fx=238.9 fy=315.8 cx=364.7 cy=356.2 @720p; projection convention in extractor |
| Grounding head (shared backbone) | `behavior2026/grounding/{dataset,model,train,eval}.py` | frozen DINOv2/v3 ViT-B feature pipeline to REUSE — one backbone pass at serve, two heads reading it |

### Known gotchas (hard-won; do not rediscover)
1. **`gripper_qpos` is unreliable under OmniGibson assisted grasping** — thick objects lift
   with qpos ~open. EE-proximity + object-motion-onset is the validated primary grasp
   signal; use relative qpos drop only to sharpen timing on thin objects.
2. Label files have **2 records per rendered frame** (physics sub-steps); take
   `idx = int((k+1)·len(rows)/n_video) − 1`.
3. Use **AABB nearest-point** distance, not object center (elongated objects grasped at an
   edge exceed center-radius thresholds).
4. Videos: depth mp4s must be decoded **gray16le via ffmpeg** (torchcodec collapses
   12-bit); depth calibration `z = 8.928e-05·raw − 0.0119`.
5. Episodes are **bimanual-parallel** (left arm completed a full pick-place nested inside
   the right arm's carry, verified) — hence per-arm tracks.

## Phase 1 — labels at scale (CPU-only, ~1 week)

**Input:** sweep labels + proprio parquet (relocation family); raw HDF5 episode state from
public `behavior-1k/2026-challenge-rawdata` (non-relocation families — joint angles,
toggle attributes, particle counts live in the recorded sim state, not our position labels).
**Output:** per-frame, per-arm stage labels for every groundable task:

```json
{"frame": i,
 "arms": {"left":  {"active_literal": 2, "phase": "transport", "progress": 0.6},
          "right": {"active_literal": 0, "phase": "approach",  "progress": 0.2}},
 "ledger": [true, false, false],            // per goal-literal satisfaction (privileged)
 "events": ["grasp:left:sandal_189"]}       // sparse transition events
```

Build items, in order:
1. **Per-arm tracks + shared ledger** (refactor of the pilot's smallest-window heuristic).
2. **Non-relocation signal families**: parse episode HDF5 state for joint angles /
   toggle attrs / particle counts. Known unknown: OmniGibson's serialized state layout —
   two paths: (a) offline parse of the state arrays (preferred, pure CPU; the playback
   wrapper's (de)serialization code documents the layout), (b) fallback: a light replay
   pass logging extra state via the same callback machinery the sweep uses (needs sim,
   coordinate with us before using a GPU).
3. **Real BDDL predicate evaluation** for the ledger (nextto/ontop/inside with AABBs,
   honoring forall/or) — replaces the pilot's relocation-completion proxy.
4. **Synset-aware target matching** (name-prefix is brittle).

**Validation gate (diagnostics-first, non-negotiable):** overlay grids on ≥4 tasks spanning
families — one multi-object relocation (e.g. collecting_childrens_toys), one articulation
(open/close), one toggle (turning_on_radio), one heavy-bimanual. A "grasp" label must land
on the frame where the hand closes on the labeled object; a ledger flip must land on the
visible completion. We eyeball together before any head training.

## Phase 2 — the stage head (light GPU, ~1–2 weeks)

**Train input:** head-cam RGB-D frames (sweep videos) + proprio history (~2–5 s window) +
task literal embedding. **Supervision:** Phase-1 labels.
**Serve output (the API contract):**

```python
stage.update(dino_features, depth_feats, proprio_hist) -> {
  "left":  {"stage_id": int, "active_literal": int, "progress": float},
  "right": {...},
  "ledger": [p_satisfied_per_literal],      # learned — no privileged state at serve
  "stage_age_ratio": {"left": 1.7, "right": 0.4},  # duration vs. extractor's median
}
```

Constraints: **shares the frozen DINOv3 backbone with the grounding head** (read its
`model.py`; one backbone pass, two heads), <20M trainable params, bf16, must fit in the
24 GB serve budget beside policy+grounding (measured headroom exists). Temporal
aggregation design (GRU / small transformer over pooled features) is owner's choice.
**Metrics:** per-arm stage accuracy + boundary timing error (frames) on held-out episodes
AND held-out scene instances; ledger F1 vs privileged truth; report per task family.

## Integration — how it layers (interface contracts)

```
                    ┌────────────────────────────────────────────┐
    sweep dataset ─▶│ PHASE 1: extractor → per-arm stage labels  │  (offline, CPU)
    rawdata state ─▶│          + privileged ledger               │
                    └───────────────┬────────────────────────────┘
                                    ▼ trains
                    ┌────────────────────────────────────────────┐
   DINOv3 feats ──▶ │ PHASE 2: stage head (per-arm stage +       │
   proprio hist ──▶ │          progress + learned ledger)        │
                    └──┬──────────────┬──────────────┬───────────┘
                       ▼              ▼              ▼
             [1] policy AdaLN   [2] grounding    [3] metacog head
                 stage token        query router     P(fail|stage) +
                 (per arm)          (which object    stage_age features
                                    to localize NOW)
```

1. **→ π0.5 policy:** per-arm stage token joins our 3D point in the AdaLN conditioning
   vector (one pathway, concatenated — the spatial+stage interference question is a
   planned G3-adjacent ablation; coordinate the embedding dims with us).
2. **→ 3D grounding head (the direct coupling):** the active literal names the object →
   the stage head's output *is* the grounding head's query. Stage says "left arm:
   transport toward `inside(toy, bin)`" → grounding localizes the *bin* for that arm.
   Grounding stays stateless ("where is X?"); stage owns *which X, when, per arm*.
3. **→ metacognition head:** stage_id + stage_age_ratio are its highest-value features
   ("grasp at 3× median duration" ≈ failing). Ledger feeds partial-credit-aware dispatch
   ("bank 3/5 literals, move on").
4. **→ task director:** consumes the ledger + per-arm stages; runs programmed arbitration
   (assign literals to arms, no double-claim). Director stays ours; its inputs are yours.

**Not blocking / not blocked:** G3 (Jul 31, point-conditioning gate) runs without the
stage head (gate tasks are short-horizon). The stage head becomes critical for the
long-horizon submission tasks and the Sep 22 metacog gate. Your Phase 1 needs no GPU;
Phase 2 training is minutes–hours on any of our boxes between sweep/eval jobs.

## Calendar & gates

| date | gate |
|---|---|
| ~Jul 25 | Phase-1 labels for all swept tasks + 4-family overlay validation (joint review) |
| ~Aug 8 | Stage head v1 + offline metric table (per-family, held-out instances) |
| ~Aug 15 | Serve-time API integrated behind the G3 point-conditioned policy; stage-keyed voting (Larchenko trick) as eval-time bonus |
| Sep 22 | Metacog gate consumes stage features (joint) |

## Reading list (fastest onboarding order)
1. This doc + `VALIDATION.md` (how we gate everything)
2. `stage/extract_stages.py` + the tidying_bedroom overlay (what "validated" looks like)
3. `probes/extract_task_targets.py` + one BDDL file (task semantics)
4. `grounding/model.py` (the backbone you share) + `grounding/dataset.py` (frame/label pairing)
5. Papers: Larchenko BEHAVIOR-25 report (stage head + voting); TD-calibration for VLA
   (ICML 63834 — the metacog consumer); our banked ICML sweep notes in memory.

# Stage / Subtask Prediction Head — Task Spec v2 (2026-07-19)

Supersedes STAGE_HEAD_OWNERSHIP.md (v1/v1.1) as the buildable spec; v1 retained for history and
gotchas. Owner: cofounder. Everything below is self-contained until the Aug 15 integration.

Live references (interactive, built from real official annotations):
- Single-goal target behavior: claude.ai/code/artifact/c5cd1c39-2731-47ce-8443-7e6c3cf5edab (turning_on_radio ep10)
- Multi-goal target behavior: claude.ai/code/artifact/cb535ca0-8691-4b3e-af91-b46f791bd6a9 (picking_up_trash ep10: literal hopping, sequential P_sat flips, q staircase)
- Full-system context: claude.ai/code/artifact/4b1c54bd-8601-422c-8e09-bd1bfc4fbae3
These three ARE the acceptance picture: the trained head, run on held-out episodes, should
reproduce these curves from pixels+proprio.

## 1. Mission (one sentence)
Per control tick, per arm, from the last 2–5 s of observation only: report which skill the arm
is executing, how far along it is, which goal literal the behavior serves, and the instantaneous
probability that each goal literal is currently satisfied.

## 2. The four outputs (serve API)
```python
stage.update(dino_feats, proprio_window, literal_embeds, task_mask) -> {
  "left"/"right": {
     "stage_dist":  softmax over OFFICIAL skill taxonomy (task-masked),   # soft, calibrated
     "progress":    float 0-1 within current skill,
     "active_literal_dist": softmax over the task's L literals,
  },
  "p_sat":   [L] sigmoid — instantaneous, NOT latched (director latches),
  "entropy": per-arm stage-dist entropy (metacog feature),
}
```
Consumers: (1) director reflex tier — latches ledger from p_sat w/ hysteresis, computes
stage_age, selects control mode (incl. HOLD when a gripper is latched+stable); (2) policy —
z_stage = Σ p_i·E_i soft mixture + sincos(progress), via AdaLN (encoders are policy-side; you
ship NUMBERS, never embeddings); (3) metacog — entropy, stage_age, progress-stall features;
(4) Opus deliberative tier — your outputs are its situation briefing at consult time.

## 3. Labels (Phase 1) — annotations PRIMARY, enrichment yours
Source of truth: `annotations/task-XXXX/episode_*.json` in behavior-1k/2026-challenge-demos.
Parse RAW JSON (HF viewer/parquet chokes on object_id nesting). Fields per segment: skill_id
(shared ~70-verb taxonomy — this IS the class vocabulary), skill_description, object_id
(target+reference args), manipulating_object_id, frame_duration, skill_type
(navigation/uncoordinated/coordinated), memory_prefix, spatial_prefix. Two granularities
(skill_annotation fine, primitive_annotation coarse w/ skill_idxes mapping) — train on skills;
primitives optional aux.

Your enrichment layers (build in this order):
1. **Per-arm attribution.** Segments are episode-global; arms act in parallel (real ep:
   left holds ashcan f642–6817 while right does everything). Attribute via skill_type +
   EE-proximity + gripper/attachment state (v1 gotchas apply: qpos unreliable under assisted
   grasping; AABB nearest-point, not centers). Untagged gaps between segments = "transition"
   class, low-confidence labels.
2. **Soft boundary labels.** Frames within ~0.3 s of a segment edge get cosine-blended
   targets between adjacent skills. Mid-segment ≈ one-hot. (This trains calibration — the
   softness in the live demos is the intended output, not sloppiness.)
3. **Progress labels.** (t − seg_start)/(seg_len) per segment. Free. Sincos-encode downstream.
4. **Privileged ledger (P_sat targets).** Per-frame predicate evaluation over recorded sim
   state via the v3.9 KnowledgeBase API — same code path as leaderboard scoring. First task:
   confirm the API surface; fallback = v1's AABB predicate evaluators.
5. **Active-literal labels.** Segment object_id → literal (target+reference match). Segments
   serving no literal (strategy moves like carry-the-container, wrap-up) → null class.
6. **Cross-check** annotation boundaries vs the validated extractor (stage/extract_stages.py)
   on ≥4 task families; disagreements go to the Jul 25 joint eyeball review.

## 4. Reference architecture (≤20M trainable; temporal core is your call)
```
per frame: pooled DINOv3 patch feats (shared frozen backbone — read grounding/model_mt.py)
           ⊕ proprio 61-D (+gripper) → fuse MLP → frame token d≈256
window:    ~30–75 frames (2–5 s) → small causal transformer (~4L) or GRU
readout:   two learned ARM QUERIES attend over window (cross-arm visible — coordination)
heads:     stage logits (taxonomy, task-masked) · progress · active-literal attention
           (dot vs literal embeds) · p_sat bilinear vs literal embeds · entropy (derived)
```
Literal embeddings: predicate type + object category encodings (your design; keep them
task-agnostic so one head serves 100 tasks). NO ledger input (circularity), NO episode-long
recurrence ("what's done" lives in the director's ledger — window perception only).

## 5. Losses & tricks (Larchenko-validated where noted)
- Stage: CE vs soft labels. Progress: MSE. P_sat: BCE. Active literal: CE.
- Head-only weight decay ~1e-3 (aux heads "massively overfit" — measured, LeHome).
- P_sat tail boost: upweight frames near satisfaction flips (near-miss states dominate
  otherwise — LeHome used 20× on last 20 frames).
- Label smoothing on p_sat toward per-task base rates.
- IS-debias if sampling non-uniformly.
- Sim-overfit probe (diagnostic, cheap): train a tiny probe to classify renderer/version of
  frames; if it succeeds across a pipeline change, the head sees the difference too.

## 6. Evaluation — Aug 8 gate (pre-registered)
On held-out EPISODES and held-out TASKS (both, reported separately, per task family):
1. Per-arm stage accuracy + boundary timing error (frames) — overlay plots in the exact style
   of the live demos; joint eyeball review.
2. Ledger F1: latched p_sat (director rules applied) vs privileged truth.
3. Progress MAE + monotonicity-violation rate on nominal segments (violations off-nominal are
   a feature, not counted).
4. **Kill ablation:** identical head trained on Larchenko-style temporal-position bins vs your
   semantic labels, held-out tasks. Semantic must win or we learn the labels bought nothing.
5. Calibration: entropy high at boundaries, low mid-segment (reliability diagram).

## 7. Calendar
- Jul 25: Phase-1 labels, 4-family overlay + annotation-vs-extractor cross-check (joint review)
- Aug 8: head v1 + metric table (gate above)
- Aug 15: serve API integrated behind the point-conditioned policy (z_stage live)
- Sep: progress output doubles as potential-shaping Φ in the corrective RFT (drifts less than
  success signals — LeHome-validated role); entropy/stage_age feed the Sep 22 metacog gate.

## 8. Non-goals (so scope stays sane)
No decisions (director's job) · no memory beyond the window (belief/ledger) · no grounding
(where-is-X is the grounding head) · no reward/value estimation (metacog/RFT machinery) ·
no per-pixel outputs. Ship calibrated numbers on the API above; everything else is consumers.

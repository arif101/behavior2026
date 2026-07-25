# Data Assembly Spec — August Phase A

**Date:** 2026-07-23 · **Basis:** `AUGUST_EXECUTION_SPEC.md` §3 · **Blocks:** BC retrain, commit
distillation, RL rewards · **Needs:** no GPU to prep (CPU + state-decoder path); no stage head.

**Why this is first:** data starvation is the root cause (30 vs 100–500 eps/task). This is the
long pole — everything downstream waits on it — and it can start now, with boxes down.

---

## 1. Inputs we already have

- **Full demos:** HF `behavior-1k/2026-challenge-demos` — 200 eps × 100 tasks (20k episodes),
  RGB + depth + state + `skill_annotation` / `primitive_annotation` (~70-verb taxonomy;
  `skill_type` nav/uncoordinated/coordinated; `manipulating_object_id`).
- **Labels:** `sweep_100/` (81 tasks labeled), `g3_deep_labels/` (30 eps × 4 gate tasks).
- **Assembler:** `g3_pipeline/g3_assemble.py` (partial-mirror fetch → action-fp label map →
  `task_targets` from annotations → `add_target_points` → LeRobot dataset). Proven on 4 tasks.
- **Action-fp mapper** (`map_labels.py`): provably-correct label→episode mapping (avoids the
  contiguous-index / multi-demo corruption bugs — see `reference_g3_conversion_bugs`).
- **State decoder** (`probes/state_decoder.py`): parses object world poses from state arrays
  **without rendering** (mm-accurate rigid; open for articulated). = 40× cheaper conditioning
  labels where it applies.

---

## 2. Task-set selection (diversity-first, not depth-first)

**Principle:** generalization scales with task/object/scene *diversity* (r=0.96–0.99), not
depth. So the selection optimizes **coverage of the skill taxonomy**, not "the 4 tasks deepest."

**Method (CPU, do now):**
1. Parse `skill_annotation` + `primitive_annotation` across all 100 tasks → build a
   (task × primitive-verb × object-category × skill_type) coverage matrix.
2. Cluster tasks by their primitive/object signature.
3. Sample tasks to **maximize taxonomy coverage** under the phase budget (§6), with these
   priorities:
   - **Tier 0 (always in):** the 4 gate tasks (radio, trash, tripod, thawing) — we have deep
     labels + eval baselines, so they anchor the A/Bs.
   - **Tier 1:** tasks sharing primitives with the gate tasks (grasp-place, press, open/close)
     — direct transfer to the skills we must fix.
   - **Tier 2:** taxonomy-filling tasks (each adds an uncovered primitive/object/scene) — this
     is where diversity generalization comes from.
4. Output: `phaseA_taskset.json` — the prioritized list + the coverage each task adds.

**Deliverable of this step:** a script `analyze_task_coverage.py` (reads annotations, emits the
matrix + the sampled set). No GPU. This is the first concrete thing to write.

---

## 3. Assembler extension (4→N tasks, 30→200 eps)

Extend `g3_assemble.py`:
- **Loop over the selected task set** (not hardcoded 4).
- **Full episode count:** map + convert all available eps/task (target 200), via the
  action-fp mapper at scale. Keep the largest-demo guard and contiguous-index handling.
- **Partial-mirror fetch stays** — pull only the files the selected episodes live in.
- **Merge** into one multi-task LeRobot dataset (`b1k_phaseA`) with `target_points` /
  `target_points_mask` columns + `task_index`.
- **Norm stats** over the merged set.

**Disk/compute:** RGB-only (drop depth for the policy; grounding uses depth separately). Budget
per §6 phasing — do NOT attempt all 100×200 at once.

---

## 4. Contact-window oversampling (the cheapest high-leverage fix)

**Problem:** contact/grasp frames are a sliver of each demo; the loss is dominated by transit.
With 30 demos the total contact signal was too weak. Re-weight toward contact.

**Identify contact/grasp frames — three signals, use the union:**
1. **Primitive annotation:** frames inside `grasp` / `press` / `place` / `open`/`close`
   segments (the annotations label these directly — strongest signal).
2. **Gripper transition:** frames where the gripper action crosses open↔closed (± a window).
3. **Proximity:** EE within a threshold of the `manipulating_object_id` (from state decoder /
   labels).

**Re-weighting:** add a per-frame `sample_weight` column; contact-window frames get weight
`w_contact` (start 3–5×, A/B), transit frames weight 1. Feed to a weighted sampler in the
training loader (or duplicate contact frames in the index). Keep it a **tunable knob** for the
§9 A/B (oversampling on/off, weight sweep).

**Deliverable:** `tag_contact_windows.py` — reads annotations + state, writes `sample_weight`.

---

## 5. BDDL reward-shaping map (the RL reward spec)

**Purpose:** dense staged rewards for Phase B online RL — independent of the stage head (which
adds a smoother signal later). Built from the task BDDL, no GPU.

**Per task, extract:**
- **Goal predicates** (terminal success condition).
- **Intermediate predicates** implied by the demo skill sequence (near / holding / on / inside
  / toggled) — each becomes a reward step.
- **A monotone progress schedule:** ordered subgoals → cumulative reward, e.g. radio:
  `at(table) +0.2 → near(button,<10cm) +0.3 → contact +0.3 → toggled +0.2`.

**Requirements (carry into RL):** dense, **monotone**, per-subtask. The stage head must match
this shape when it comes online (cofounder handoff, §11 of the August spec).

**Deliverable:** `bddl_reward_map.py` → `reward_shaping.json` (per-task subgoal→predicate→weight).

---

## 6. Phasing + budget (be realistic)

Do **not** assemble 100×200 at once. Phase by coverage value:

- **Phase A0 (start now, CPU):** the analysis + tooling — `analyze_task_coverage.py`,
  `tag_contact_windows.py`, `bddl_reward_map.py`, assembler extension. No GPU, no box.
- **Phase A1 (first data box):** assemble the **Tier 0 + Tier 1** set (~15–25 tasks) at full
  200 eps → `b1k_phaseA1`. This is enough to retrain BC and get the first honest nonzero.
- **Phase A2 (scale):** add Tier 2 taxonomy-fillers as compute allows → `b1k_phaseA2`.

Disk: estimate ~4.4GB per 120 eps observed for gate tasks → budget ~1–2GB/task for 200 eps
RGB-only; a 20-task A1 set ≈ 20–40GB. Fits a single data box.

---

## 7. Validation gates (before any training)

Reuse the discipline that made the gate data trustworthy:
- **Action-fp mapping:** every label→episode match verified (0 collision), largest-demo guard.
- **target_points sanity:** coverage %, |d| distribution in 0.1–3m (per `add_target_points
  --validate`).
- **Contact-tag audit:** spot-check that tagged windows actually contain gripper transitions /
  proximity (a few per task).
- **Reward-map audit:** confirm each task's subgoal schedule is monotone and terminates at the
  goal predicate.
- **No silent truncation:** log any task/episode dropped (missing labels, mapping failure).

---

## 8. Concrete first actions (all no-GPU, startable now)

1. `analyze_task_coverage.py` — the coverage matrix + `phaseA_taskset.json`.
2. `bddl_reward_map.py` — `reward_shaping.json` (also feeds the cofounder's stage-head target).
3. `tag_contact_windows.py` — the oversampling weights.
4. Extend `g3_assemble.py` to loop the task set at 200 eps.

Then Phase A1 assembly runs on the first data box; BC retrain (+SAM A/B) follows on a trainer.

---

## 9. What this unblocks

- **BC retrain** on diverse + oversampled data → the floor-raiser.
- **Commit-primitive distillation** target set (contact windows identify where to teach).
- **RL staged rewards** (`reward_shaping.json`) → Phase B, without waiting on the stage head.
- **Cofounder alignment:** `reward_shaping.json` is the shape the stage head should output.

---

## 10. AS-BUILT (2026-07-25) — where reality diverged from the plan

Recorded because three of these change how the Phase-A A/B must be *read*, not just how it was built.

**Dataset.** 1,837 episodes / 18,837,213 frames / 24 tasks, on the 2×A100 trainer box at
`/root/phaseA/b1k_phaseA` (94 GB). Norm stats sampled at 40k frames; both arms share them
(`assets_phaseA/pi05_phaseA_point` on HF).

**Depth is bimodal — this is NOT "200 eps/task".** Only 4 tasks reached 200 episodes
(`picking_up_trash`, `thawing_frozen_food`, `turning_on_radio`, `attach_a_camera_to_a_tripod`
= 800 eps). The other 20 tasks have ~50 each (1,037 eps); median is 51.5, and **20 of 24 tasks
sit below the 100 eps/task level that is the lowest documented nonzero reference**
(π0.5 ≈56% tabletop at 100/task). Consequences:
- Do not describe Phase A as "30→200/task". It is 30→200 on four tasks and 30→50 on twenty.
- **Eval must be stratified deep-vs-shallow** (`/root/phaseA/eval_strata.json`). This is a free
  within-run dose-response on episodes/task at otherwise identical settings.
- If success is nonzero it should concentrate on the four deep tasks. If it does *not*, depth
  is not the binding constraint — which argues for RL over collecting more demos.

**204 episodes were dropped.** The fetch produced 2,041 episodes, but 204 had data parquets
with no video chunks (bonus episodes riding along in shared parquets). LeRobot rejects the
whole load, surfacing as a misleading HF 401. Kept the 1,837 video-complete ones. Diagnosis
ladder is in the `setup-training-box` skill.

**Contact numbers, corrected on the final dataset.** Commit frames are **2.98%** (the 3.12%
figure was measured on the 2,041-episode set), weighted 10.36× → **24.1% of draws**
(empirically 24.3%). That is ~8× more contact exposure than uniform. Concretely, over the
60k×32 = 1.92M draws of a run: the weighted arm draws ~463k commit samples against 561k
commit frames (~0.83× coverage), the uniform arm ~57k (~0.10× coverage).

**Two sampler bugs nearly voided the A/B** (both fixed, see `_patch_contact_sampler.py`):
weights were patched into `create_torch_data_loader()` while `train_b1k.py` calls
`create_b1k_data_loader()`, so the "weighted" arm silently trained uniform with no log line
either way; and `WeightedRandomSampler`→`torch.multinomial` caps at 2²⁴ = 16.8M categories
< 18.8M frames, so it would have crashed on the correct path anyway. Unusable weights now
**raise**. The A/B is only valid if `CONTACT OVERSAMPLING ON` appears in the arm-1 log.

**Gradient budget caveat.** 60k steps × 32 = 1.92M samples = **0.10 epochs** over 18.8M frames,
and per-task budget is ~3× lower than G3 (60k/24 tasks vs 30k/4). Justified by the diversity
power law, but it means a null result should not be read as "BC scale doesn't help" without
noting the steps/task drop.

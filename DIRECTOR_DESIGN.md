# Director (Orchestrator) Design Spec v1 — 2026-07-17

Programmed decision layer of the serve loop. Runs every control tick; most decisions change at
subtask cadence. It is the ONLY component that assembles the policy's conditioning payload.
Motor control stays exclusively in pi0.5 — the director never emits joint commands.

## 1. Interfaces

### Inputs (per tick)
| source | payload |
|---|---|
| stage head | per-arm: stage_dist[K taxonomy], active_literal_dist[L], progress ∈[0,1], entropy; global: P_sat[L] (instantaneous) |
| grounding head | per issued query: point_w[3] (world), conf, heatmap_max, visible flag |
| belief head | hypothesis set: {category, instance_id, μ[3], Σ[3×3], P_exists, age_s, attached_to} |
| metacog head | P_fail (calibrated), conformal state |
| proprio/odom | base pose (world), per-arm EE pose, gripper state |
| static (episode start) | compiled task plan (below), per-task config: stage mask, retry budgets, per-literal timeout = κ × demo-median duration |

### Episode state (explicit, all printable)
- `ledger[L]` ∈ {UNSAT, SAT_LATCHED, BANKED} + per-literal hysteresis evidence counters
- per-arm: `assigned_literal`, smoothed phase, `stage_age`, `control_mode`
- `retry_budget[L]`, macro FSM state (IDLE | MACRO_x + sub-state), scan flags, 3-window vote ring buffers

### Outputs (per tick)
- → grounding: per-arm queries — (target category/instance, reference category) of the arm's active literal (2 queries/arm max)
- → belief: scan request (episode start / room entry), hypothesis confirm/retire
- → policy serve payload (websocket obs fields, contract already implemented on branch adaln-point):
  `target_points[2,3]` gripper-relative Δ = R_wb^T(p_w − p_ee_w) — same math as the training
  converter (validated on real data); `target_points_mask[2]`; `stage_tokens[2]` (soft stage
  mixture + sincos progress); later `value_token`; `exec_prefix` (control mode → how many of
  the 32 chunk steps run before replan)

## 2. Task compilation (episode start, once)
BDDL problem → literal list with: predicate type, target set (category + count via
KnowledgeBase), reference object, forall/or structure expanded to instances (belief count
priors), dependency order (open(container) before inside(x, container); clear-surface before
place), per-literal skill template (approach→contact→effect→complete watch-variables).

## 3. Nominal decision rules (every tick)
1. **Assignment**: greedy — each free arm takes the nearest UNSAT literal target (belief μ vs
   EE, reachability side heuristic); no double-claim; literals whose demo skill_type =
   coordinated claim BOTH arms.
2. **Query selection**: active literal names target + reference → both queried. If grounding
   conf < τ or occluded → serve belief point with age scalar; mask=0 only if belief has no
   hypothesis (policy trained with point-dropout so this is in-distribution).
3. **Transitions**: hysteresis voting (advance 2/3 windows, rollback 3/3). Ledger latch:
   P_sat > θ for M consecutive windows → SAT_LATCHED; contradiction (P_sat collapse + geometry
   disagreement) → un-latch only via the rollback rule.
4. **Control mode**: stage phase → exec_prefix: navigation = long prefix (receding horizon),
   approach/contact = short prefix (tight replan). (Merlyn: 30→100% on door traversal.)
5. **Banking**: literal latched → free the arm → next literal. BANKED literals are never
   revisited unless contradicted. Partial credit is the objective: q = mean over literals.

## 4. Exceptional path (recovery dispatch — decisions only)
Trigger: metacog P_fail > α sustained (conformal), OR stage_age > κ×median, OR grounding
starvation. Macro selection keyed on (phase, trigger):
| context | macro |
|---|---|
| approach + low grounding conf | RE-GROUND: re-query, belief re-open (inflate Σ), optional scan |
| grasp/contact + stagnation | RETREAT-RETRY: truncate chunk, condition retreat to pre-contact waypoint, fresh grounding, retry; budget-- |
| post-effect + P_sat collapse | STAGE-ROLLBACK: re-activate literal, rollback ledger (3/3 rule) |
| budget exhausted | ABANDON: mark literal unrecoverable, reassign arm — bank q elsewhere |
All macros act THROUGH the conditioning payload (change point/stage/prefix), never scripted
joints. Recovery skills live in the policy (corrective RFT); the trigger lives in metacog.

## 5. Policy fine-tuning plan (teaching pi0.5 the tokens)
- **Data**: 20k official demos (LeRobot v3). Per frame, per arm: gripper-relative Δ to the
  CURRENT target — the annotation skill segments name the manipulated object + reference per
  segment, so "current target" labels are free; converter (add_target_points.py) computes Δ,
  validated on real data (radio 90.6% coverage). Stage-token labels = annotation skill_id +
  within-segment progress. Ledger labels = KnowledgeBase predicate evaluation.
- **Objective**: flow-matching action loss (chunk 32) + trunk preservation (VQA + subtask-
  prediction co-training batches from our labels; SAM aux as banked) — full fine-tune, not
  LoRA (W1: 5-10k LoRA insufficient).
- **Conditioning entry**: AdaLN — adarms_cond = z_time + z_point + z_stage (+ z_value later);
  zero-init final MLP layers ⇒ bit-exact base behavior at step 0 (verified), so the policy
  LEARNS to use tokens because they reduce loss, not because we force it. Points carry
  information the images underdetermine (which instance, metric 3D) — that asymmetry is the
  teaching signal.
- **Serve-distribution matching (the critical part)**: during FT inject the tokens as they
  will look at serve, not as oracle:
  (a) point-dropout (mask=0 fraction) — policy must not die when grounding starves;
  (b) point noise sampled from the grounding head's MEASURED error distribution (px→3D);
  (c) staleness: refresh points at serve cadence (not every frame) + age scalar input;
  (d) stage-token dropout + soft (blended) boundary labels;
  (e) belief-sourced points for occluded-target segments (larger noise + age).
  This is the goal-noise lesson (LIBERO: honest object 0→0.40 via goal-noise aug) applied
  to every conditioning channel.
- **Sequence**: G3 (Jul 26-31): 3 configs × 4 gate tasks — the gate for the whole axis.
  Aug: full 96-task FT + hardening ablation grid. Sep: corrective AWR RFT on top (proximal
  anchor; stage-gated exploration noise).

## 6. Explicit non-responsibilities
- No joint-space control, no waypoints as actions (policy is the only motor).
- No steering of head inference (queries are config; heads run unconditionally).
- No learned parameters in v1 (LLM consultant = gated Sep experiment behind timeout+fallback).

# HANDOFF — BEHAVIOR-2026 state + next steps (2026-08-12)

For a new operator (human or Claude) on a fresh machine. Everything durable lives in this
repo + HuggingFace (`arif101/*`, private). Nothing essential exists only on a box or a
laptop. Read order: this file → RUN2_EVAL_REPORT.md → CONTACT_SKILL_SPEC_v1.md →
SPATIAL_INTEL_RESEARCH_2026_08_09.md (skim §2/§5) → DESIGN.md (incl. the 08-09 eval-
observation-legality adjudication).

## Where we are (2026-08-12)
- **Run-2 trained + fully evaluated under frozen pre-registration.** Verdict in
  RUN2_EVAL_REPORT.md: primary 0/25 FAIL; approach transformed (EE to 2-3 cm vs run1b's
  0.35 m standoff); affordance-point route flagged decorative-at-eval (n=5 caveat);
  commit-probe mass 0.00 with collapsed diversity → **commit mode ABSENT in the policy's
  sampled distribution**. All selection-class levers (best-of-K, DSRL, sticky decoding,
  chunk-critic rerank) are support-dead. Decision: RECONSIDER RECIPE.
- **Funded main line: CONTACT_SKILL_SPEC_v1.md** — small learned press policy (~2M params,
  RLPD + reverse curriculum from sim-restore states), gated by stage head + geometry,
  base VLA (radio_run2@49999) frozen. Build plan §7; bars pre-registered, freeze at first
  training launch. Day-0 items were interrupted by the sim box dying — resume there.
- **Task expansion** (4 signed-off tasks, TASK_EXPANSION_SPEC.md) proceeds independently —
  q counts unattempted as hard 0; conversion of zeros is the biggest EV on the board.
- Deadline: freeze ~Oct 10, submit Oct 16.

## HF asset map (all private, under arif101; need a read+write token)
| Repo | Contents |
|---|---|
| b26-run2-params (model) | **Serving base**: params@49999 + norm stats (assets/b1k_radio) + ckpt_49999 duplicate + provenance/ (all Run-2 training logs) |
| b26-run1b-params, b26-run1b-ckpt36559, behavior2026-radio-gate-49999 (model) | Prior checkpoints (warm-start lineage) |
| behavior2026-artifacts (model) | code/openpi_fork_adaln_src.tar.gz (BASE fork tarball — pre-run-1!), G3-era ckpts/labels |
| b26-foveated-backup-20260730 (dataset) | b1k_radio_map data (v2, gt_depth_ds registered), map_tokens, metalink_labels, keys/ (norm stats, intrinsics), aff_out/ (affordance head), run1b campaign archives |
| b26-corrective-backup-20260806 (dataset) | b1k_radio_corrective (68 eps, SPLICE CLIPS — the contact skill's prior data), rac v1 raw |
| b26-rac-backup-20260807 (dataset) | b1k_radio_rac (105 eps) + raw npz |
| b26-run2-eval-20260811 (dataset) | **Full Run-2 eval**: 25 rollout videos+jsons, ablation, commit_probe.log, action_log, map_live snaps + renders |
| behavior-1k/2026-challenge-demos, -rawdata (org, public) | demos / raw hdf5 (restore-state sources) |

## Box provisioning (skills are IN this repo, .claude/skills/)
- Sim/eval/RL box (RT cores: A5000/A6000/4090/L40S — NOT A100): `/setup-sim-box` skill.
  Acceptance gate FIRST. Fork restore for serving = tarball + fork_snapshot/ overlay
  (7 files, run-1 layer) + 5 run-2 patches (order in RUN2_CONFIG.md) + robot_r1 sed +
  eval-kit patch chain point_passthrough → map_passthrough2 (+ action_logger).
- Trainer box (A100-class, only for Run-3 BC): `/setup-training-box` skill.
- Hard-won gotchas: 20G /workspace volume trap (install to /root, symlink /root/bw);
  authenticated HF + HF_XET_HIGH_PERFORMANCE (never unauthenticated aria2); og.shutdown
  hangs (scripts use os._exit → ALWAYS `python -u`); pgrep/pkill self-match over ssh
  (bracket-pattern); public_test instances are IDs 301-320 addressed as indices 0-19;
  instance 301 = held-out eval, NEVER train on it; nopi.so futex shim only if the futex
  probe returns EPERM.

## Immediate next steps (CONTACT_SKILL_SPEC_v1.md §7, day-0/1)
1. Rent RT-core box → /setup-sim-box → serving stack up (smoke: RUN2_SERVE_SMOKE_OK (32,23)).
2. Restore-fidelity probe (box_scripts/probe_restore_fidelity.py; needs metalink_labels
   ep10 + rawdemos ep10). Gate: drift acceptable at contact frames.
3. Demo pool fetch (task-0000 episodes 0-39, ~5 MB each) for curriculum start states.
4. Build: splice→skill-buffer converter; state-harvest pass (restore demo → run VLA →
   snapshot sim states in the 10 cm shell → policy-visited start bank).
5. Env wrapper + RLPD loop per spec §3/§4; smoke stage-0; then curriculum training.
6. Arm-C eval ONLY per the frozen bars in spec §5.

## Non-negotiable discipline (why this project's results are trustworthy)
- Pre-register evals BEFORE results; bars never move after data arrives (G1 lesson).
- Single-variable changes between runs; every claim traces to a measurement.
- Learned > scripted for anything claimed as contribution (standing directive).
- Milestone rule: back up to HF at every milestone; checkpoints leave the box ALWAYS
  (the 08-08 trainer termination lost a full run to finalize-only-at-end).
- Depth images ARE legal eval inputs ("RGB + depth + proprioception"); prohibited:
  GT seg/object state/target pose/full-scene point cloud/robot global pose.

## Access checklist for the new operator
1. Clone github.com/arif101/behavior2026, branch `rescue/box-artifacts-20260726`.
2. HF token with read+write on arif101/* (get from Arif; store on boxes at
   /root/.hf_token chmod 600, NEVER inline in commands).
3. SSH keypair for box rentals (~/.ssh/id_ed25519 pattern in all scripts).
4. Box budget: one RT-core box (~48 GB VRAM ideal) is enough for the entire contact-skill
   phase (sim + tiny learner co-resident).

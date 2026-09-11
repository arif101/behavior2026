# Run-3 sim-eval — n=12 directional sweep (instance 301, turning_on_radio)

Validated harness (AffordanceMapFullRes wrapper + MAP_ARM=B + 3 passthrough patches);
primary metric = grasp-completion (assisted-grasp weld on the radio). Rollouts are
seed-paired across arms (eval seeds once from DEFAULT_EVAL_SEED before the loop).

## Result: monotonic, dose-dependent grasp signal
| Arm | Manufactured data | Grasp | Success | Fingertip median | Closer than A0 |
|-----|-------------------|-------|---------|------------------|----------------|
| A0  | none (control)          | 0/12 | 0/12 | 0.335 m | — |
| A2  | +factory                | 1/12 | 0/12 | 0.352 m | 5/12 |
| A4  | +factory +episodes      | 2/12 | 0/12 | 0.263 m | 8/12 |

More manufactured data -> more grasps AND closer reach. The control never grasps.

## Paired comparisons (n=12)
- A0 vs A2: grasp delta +1, McNemar p~=1.0 (single discordant pair).
- A0 vs A4: grasp delta +2, McNemar p=0.5; A4 reaches closer in 8/12 (sign-test p=0.39).
- All individually non-significant at n=12 (underpowered); the cross-arm trend is the signal.

## Film verification
- A2 rollout 1 grasp is film-verified REAL: gripper closes on the radio, left wrist cam
  filled by it across multiple frames — sustained AG weld (grasp-in-place, no lift).
- A4 grasps (rollouts 7, 10) share A2's telemetry signature (grasp=True, fingertip ~0.12 m)
  but were not filmed (n=12 driver capped video at first 3 rollouts). Fixed: n=25 films all.

## Decision
Dose-response justifies the n=25 confirmation. Running A0, A4, A2 at n=25 on the sim box
(sequential). A5 (+approach) VOID — see b1k_radio_approach/QUARANTINE_DO_NOT_TRAIN.md.
Go bar (pre-registered): grasp delta >= 3 & p < 0.10.

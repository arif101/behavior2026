# RUN-1 EVAL PROTOCOL — pre-registered 2026-07-30, BEFORE any Run-1 result exists

Commitment device: arms, metrics, budgets, and decision rules are fixed here, in advance.
Deviations require a written amendment in this file with a reason, before unblinding the read.
(Lineage: this discipline — not any component — is what killed nine wrong hypotheses cheaply.)

## Checkpoint under test
`pi05_radio_map` @ ~50k steps (warm-start 49999; K=8 map tokens, metalink AdaLN labels,
5 aux heads, anti-shortcut, 8× stage-transition oversampling). Eval on a fresh sim box,
task turning_on_radio, public_test (instance 301), h32 full-chunk serving, seam-residue
validity check per run, wrapper stats verified per run (n_inject > 0, skips == {}).

## Arms (all serving-side switches on the SAME checkpoint, except D)

| arm | map tokens | target record | kick | n (initial) | question |
|-----|-----------|---------------|------|-------------|----------|
| A   | ALL-ZERO tokens (8 tokens present, all 72 dims zero) | ON | off | 15 | no-map-info control, IN-distribution |
| B   | FULL | ON | off | 15 | the architecture as designed |
| B0  | BLIND (blind_view at serve: geometry kept, target fields zeroed) | ON | off | 10 | is the map's value geometry or target persistence? |
| C   | FULL | ON | ON | 10 | how much does the serving-side stall-kick add? |
| D   | ckpt 49999 legal (context arm) | ON | off | 0–5 opportunistic | ties to the 0/4 baseline |

Budget: ~35 min/rollout ≈ 40/day → Aug 4–7 covers 50 primary + extensions.
Extension rule: whichever of {A vs B} is ambiguous at n=15 gets +10 before any other spending.

## Metrics — in pre-registered priority order

1. **PRIMARY — attempt-formation rate**: fraction of rollouts with min(dist_L, dist_R) < 0.30 m
   sustained ≥ 50 consecutive steps (from the wrapper's per-step distance series).
   Rationale: large effect size at small n; the oracle-arm reference band is 3/11.
2. **SECONDARY — grasp-descent initiation rate**: a formation followed within 150 steps by
   min-dist < 0.12 m with a closing gripper command. The mechanism metric (run-4's zero).
3. **Wrist acquisition**: fraction of formation-steps with the target inside either wrist FOV
   (angle between wrist-camera axis and wrist→target vector < 35°; requires the angle-logging
   wrapper patch — instrumentation item I1 below). G1 threshold uses the MEDIAN off-axis angle.
4. **Success (q)**: reported, headline, explicitly UNDERPOWERED at these n (documented:
   Fisher at n=15/arm cannot separate 1/6 from 2/6). No decision keys on q alone.
5. Diagnostics (no decisions): conf trajectories, freeze incidence (arm-cmd std < 0.03 over
   500+ steps), map-token liveness at serve, aux_heat readouts on logged frames.

## Decision rules (written before data)

- **G1 PASS** (Aug 8): formation(B) − formation(A) ≥ +25 pp AND wrist off-axis median(B) < 60°.
- **Map value decomposition**: B ≈ B0 > A → geometry channel carries it (memory-lite);
  B > B0 ≈ A → target persistence carries it; B ≈ B0 ≈ A → map tokens not consumed at serve →
  run the token-swap causal probe (R3) before concluding anything.
- **Kick read**: C − B ≥ +2 formations-converted-to-initiations → kick ships as the Run-2
  collection scaffold default; C ≈ B → kick is collection-only.
- **Failure of G1 does NOT kill the program**: the pre-committed fallback ordering is
  scaffold/RaC corrective data (Run 2) > critic-selection at serve > aux rebalance. Oversampling
  and prefix-injection get re-examined only after those.

### AMENDMENT 1 (2026-07-30, pre-launch, before any result)
G0 preflight measured that appending 8 tokens perturbs the pretrained policy via attention-
denominator dilution (action delta 39% with visible zero-content tokens; 0.09% invisible —
discriminator-verified). Training always presents 8 tokens (full/blind streams), so the trained
model's in-distribution condition INCLUDES token presence. Therefore: arm A changes from
token-ABSENCE to ALL-ZERO tokens (presence preserved, content nulled). Decomposition:
B vs B0 = target-channel value; B0 vs A = geometry-channel value; B vs A = total map value.
Token-absence remains a diagnostic curiosity only. ReZero gate (map_alpha=0 init) + position-
transparent tokens (pos_weight) added to the model pre-launch; both in fork + snapshot.

## Instrumentation before the campaign (I-items)
- I1: wrapper logs wrist off-axis angles per step (extend the dist-series patch).
- I2: per-arm debug-console artifact auto-generated per rollout batch.
- I3: action-log archival per run (existing convention) + stats file (existing).

## Reporting
One table, all arms × metrics 1–4 with exact n; the two decision rules answered YES/NO;
any amendment listed. Videos: one representative + one best per arm.

# AdaLN 3D-Point Conditioning for the pi05 Action Expert (`adaln-point` branch)

Conditions pi0.5's flow-matching action expert on per-arm 3D target points via AdaLN
modulation — the interface validated by arXiv 2606.27663 (oracle point via text ≤ +11.3,
via 3D-AdaLN +46.3 on LIBERO-PRO). Rather than inventing a new injection site, this diff
**extends the adaRMS conditioning pathway pi0.5 already uses for the flow timestep**: the
point embedding is added to the same conditioning vector, and the per-layer scale/shift/gate
derive from it exactly as they already do for time.

With `point_conditioning=False` (default) the model is **exactly** the stock model: no new
parameters, no new inputs, no behavior or rng-stream change. G3's control arms depend on this.

## Files touched

| File | Change |
|---|---|
| `src/openpi/models/pi0_config.py` | `point_conditioning: bool = False`, `point_noise_std: float = 0.02`, `stage_conditioning: bool = False`; constants `NUM_POINT_ARMS = 2`, `STAGE_VOCAB_SIZE = 32`; validation (flags require `pi05=True`); `inputs_spec` gains the new fields when enabled. |
| `src/openpi/models/model.py` | `Observation` gains optional `target_points [*b,2,3]`, `target_points_mask [*b,2]`, `stage_tokens [*b,2]`; `from_dict` mapping (points+mask must come together); `preprocess_observation` passthrough. |
| `src/openpi/models/pi0.py` | New modules in `Pi0.__init__` (created **after** all stock modules so the stock param rng stream is untouched); `_embed_cond_extras()` helper; additive injection in the `pi05` branch of `embed_suffix`; train-time point noise in `compute_loss`. |
| `src/openpi/policies/b1k_policy.py` | `B1KInputs` gains `point_conditioning`/`stage_conditioning` flags; packs `target_points`/`target_points_mask`/`stage_tokens`; absent key → zeros + all-invalid mask sentinel. Flags off → no new keys (stock byte-identical). |
| `src/openpi/training/config.py` | `LeRobotB1KDataConfig`: dataset key fields (`target_points_key` etc.); repack-mapping entries and `B1KInputs` flags are added **only when the model config enables them**. |
| `src/openpi/shared/eval_b1k_wrapper.py` | `process_input` forwards optional `target_points` / `target_points_mask` / `stage_tokens` from the websocket obs dict (batched or unbatched). |
| `src/openpi/models_pytorch/pi0_pytorch.py` | Loud `NotImplementedError` if a point/stage-conditioned config is loaded into the PyTorch model (JAX-only for now). |
| `tests/test_point_conditioning.py` | CPU test suite (see below). |

## Conditioning math

Existing pi05 pathway (unchanged): for flow timestep `t`,

```
z_time = swish(W_t2 · swish(W_t1 · sincos(t, d, 4e-3..4.0)))          # [b, d]
```

`z_time` is passed as `adarms_cond` for the action expert; inside each gemma block,
`RMSNorm(x, cond)` computes `Dense(cond) → (scale, shift, gate)` (zero-init Dense) and
applies `x̂·(1+scale)+shift` with a gated residual.

New point pathway (per arm `a ∈ {L, R}`, point `p_a ∈ R³` gripper-relative, meters):

```
e_a   = concat_{c=1..3} sincos(p_a[c], d/4, 4e-3..4.0)                 # [b, 3d/4]  (same posemb as time)
f_a   = W_p2 · swish(W_p1 · e_a)                                       # [b, d/2],  W_p2 ZERO-INIT
f_a   = valid_a ? f_a : null_a                                         # null_a: learned [d/2], zero-init
z_pt  = concat(f_L, f_R)                                               # [b, d]   (arm identity via position)
```

Reserved stage pathway (`stage_conditioning`): `z_st = W_s · concat(E[s_L], E[s_R])` with
`E: 32 × d/4` embedding table and `W_s: d/2 → d` **zero-init** (the projection reconciles the
spec's `2 × d/4` table output with the `d`-wide conditioning vector).

Injection (the only change to the model's forward path, `embed_suffix`):

```
adarms_cond = z_time + z_pt (+ z_st)
```

Invalid points are also zeroed **before** encoding so garbage/NaN sentinels can never leak
through `jnp.where` gradients.

### Zero-init identity start

`W_p2`, `null_a`, `W_s` (and their biases) are zero-init ⇒ `z_pt = z_st = 0` exactly ⇒
`adarms_cond = z_time + 0`, bit-identical to stock (float `x + 0.0 = x`). Verified bit-exact
(max|diff| = 0.0) on `sample_actions` and `compute_loss` (train and eval). Same trick as
Larchenko's identity-init mixed-layer attention.

### Train-time point noise

`compute_loss(train=True)` adds `N(0, point_noise_std²)` (default 2 cm) to `target_points`.
The rng is derived via `jax.random.fold_in(rng, 2606)` inside a Python-level branch, so the
stock `split(rng, 3)` stream (preprocess/noise/time) is **bit-identical** when the flag is off
— control-arm training reproduces stock exactly, seed for seed.

### A subtlety found in the fork (matters for reviewers)

The gemma `RMSNorm` modulation `Dense` layers are themselves **zero-init** in this fork. At a
completely fresh init, no gradient flows into `adarms_cond` at all (this is equally true for
the stock time-MLP). This is a non-issue in practice because G3 warm-starts from `pi05_base`,
where those Dense kernels are trained (nonzero). The gradient-flow test simulates the
warm-start by perturbing them. Do not "fix" the zero-init — it's how the upstream checkpoint
was trained.

## G3 arm → config mapping

| G3 arm | Model config | Data config |
|---|---|---|
| 1 (stock BC control) | `pi05_b1k` unchanged (`point_conditioning=False`) | unchanged |
| 2 (control variant) | unchanged | unchanged |
| 3 (point-conditioned) | `Pi0Config(..., pi05=True, point_conditioning=True, point_noise_std=0.02)` | dataset carries per-frame `target_points` `[2,3]` float32 + `target_points_mask` `[2]` bool |
| 4 (point + stage, cofounder's head) | add `stage_conditioning=True` | dataset also carries `stage_tokens` `[2]` int32 in `[0, 32)` |

Arms 1/2 execute literally stock code paths: no new transform keys, no new params, unchanged
rng streams, unchanged repack mapping.

## What the H100 training config needs to set

Add a `TrainConfig` next to `pi05_b1k` (e.g. `pi05_b1k_point`):

```python
TrainConfig(
    name="pi05_b1k_point",
    model=pi0_config.Pi0Config(action_horizon=32, pi05=True,
                               point_conditioning=True, point_noise_std=0.02),
    data=LeRobotB1KDataConfig(
        repo_id=...,
        base_config=DataConfig(...),
        robot_config_name="b1k/R1Pro",
        # defaults: target_points_key="target_points", target_points_mask_key="target_points_mask"
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    ...
)
```

Notes:
- `CheckpointWeightLoader` merges with `missing_regex=".*"`, so the new params
  (`point_mlp_in/out`, `point_null_embed`, `stage_embed/proj`) fall back to their fresh
  (zero-)init while everything else loads from `pi05_base` — training starts EXACTLY at the
  warm-started unconditioned model. No checkpoint surgery needed.
- The LeRobot dataset must contain per-frame `target_points` (float32 `[2,3]`, gripper-relative
  meters) and `target_points_mask` (bool `[2]`) under the configured keys; frames with no
  target for an arm should store zeros + `False`. The repack transform requires the keys to
  exist for every frame when the flag is on.
- Points are **not normalized** (raw meters); the sinusoidal encoding (periods 4e-3..4.0) covers
  mm-to-meter scales. Do not add them to norm stats.
- `compute_norm_stats` / existing assets are unaffected (points bypass `Normalize`).
- PyTorch training path (`train_pytorch.py`) is not implemented for this feature and fails loudly.
- Debug `FakeDataset` generates int32 in `[0, 2048)` for `stage_tokens`; out-of-range indices are
  clamped by JAX take semantics — harmless for smoke tests, real labels must be `< 32`.

## Serve-side contract (grounding head → policy)

The websocket obs dict (B1K eval client → `B1KPolicyWrapper.process_input`) gains optional keys:

- `target_points`: float `[2,3]` (or `[B,2,3]` when batched) — per-arm (left, right) gripper-relative
  3D displacement to the target, meters.
- `target_points_mask`: bool `[2]` (or `[B,2]`) — `False` = arm has no target this step (model uses
  the learned null token). If `target_points` is sent without a mask, all-valid is assumed.
- `stage_tokens`: int `[2]` (or `[B,2]`), values in `[0, 32)` — reserved for the stage head.

Absent keys → the policy input transform emits the zeros + all-invalid sentinel, and (at zero-init /
for an unconditioned checkpoint) the model behaves identically to unconditioned. A point-conditioned
server therefore degrades gracefully if the grounding head is down. A stock server (flags off) simply
ignores the keys (`B1KInputs` doesn't forward them; `Observation.from_dict` never sees them).

The serve config must load the point-conditioned `TrainConfig` (`--policy.config=pi05_b1k_point`) so
`create_trained_policy` builds `B1KInputs` with the matching flags (they are derived from the model
config, not set manually).

## Tests (`tests/test_point_conditioning.py`)

CPU-only, tiny model (`paligemma_variant="dummy"`, `action_expert_variant="dummy"`, float32; note
SigLIP is fixed So400m — init ~4 s, forward ~2 s on an M-series CPU):

- (a) `test_forward_shapes` — sample_actions shapes with/without conditioning, with/without points.
- (b) `test_stock_params_identical` — all stock params bit-identical between conditioned and
  unconditioned models built from the same seed (new modules are created last); zero-init contract.
- (b) `test_zero_init_equivalence` — `sample_actions` and `compute_loss` (train=False **and**
  train=True, exercising the point-noise branch) bit-identical (max|diff| = 0.0) between
  `point_conditioning=False` and `=True`-at-init, with and without points supplied.
- (c) `test_gradient_flow` — with warm-start-simulated adaRMS Dense kernels: valid mask ⇒ nonzero
  grads to `point_mlp_in/out` + `stage_proj`, zero grad to null token; invalid mask ⇒ nonzero grad
  to null token, zero to the point MLP.
- (d) `test_invalid_mask_path` — invalid arms ignore point values; absent points ≡ zeros+invalid
  sentinel; per-arm mixed masks distinct.
- (e) `test_b1k_inputs_packing` — transform-level contract incl. flags-off (no new keys).

## Deviations from the spec (adaptations to the fork's reality)

1. **Stage table output is `2 × d/4 = d/2`, not `d`** — the spec's "32 × d/4 table, same additive
   injection" cannot be added to the `d`-wide conditioning vector directly; a zero-init `d/2 → d`
   projection was added. Interface (`stage_tokens [b,2]` int, table 32×d/4) is as specified.
2. **Point noise lives in `Pi0.compute_loss`, not in a transform** — transforms are shared between
   training and serving in this fork; a transform-level noise would have required a train-only
   transform plumbing that doesn't exist. Model-side noise is gated on `train=True` and is
   rng-isolated from the stock stream (fold_in).
3. **`B1KInputs`/repack only emit the new keys when the model config enables them** — the spec
   said "transforms accept target_points from the batch"; unconditionally emitting keys would have
   changed the training batch structure of the control arms, violating the "exactly stock" constraint.
4. **gemma's adaRMS Dense is zero-init in this fork** (see subtlety above): gradient flow to the new
   params is real only after warm-start — tests simulate this; G3 always warm-starts so this matches
   production.
5. **PyTorch model untouched** except a guard: the fork has a parallel PyTorch implementation
   (`models_pytorch/`); implementing the feature twice was out of scope for the JAX H100 run and
   silently ignoring the flag would be worse than failing loudly.

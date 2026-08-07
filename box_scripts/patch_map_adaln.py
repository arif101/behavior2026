"""Patch the openpi fork with MAP-GEOMETRY -> AdaLN conditioning (Run-2 mandate).

Anchored, assertion-guarded edits (the patch_*.py convention — never blind-overwrite fork
files, they carry box-only changes). Every file is compile()d BEFORE writing and
py_compile'd after, so a bad edit never lands on disk.

WHY (measured, G1 campaign 2026-08-04): the prefix-token map route is consumed but pays
nothing at 90% visibility (pre-registered primary FAILED; K=8 route FROZEN), while B0
(geometry-only map) held the best approach median (0.337) and best wrist runs (28-29 deg)
— and the AdaLN displacement route is the one that produced every conversion. Run-2
therefore pools the map's GEOMETRY channels into the SAME adaRMS conditioning vector that
already carries the timestep + target points.

GEOMETRY CHANNEL SELECTION (mapper/foveated_map.py token contract, (8,72) frozen for Run 1):
  rows [1,3,4,5,6,7] = T1 room occupancy+staleness, T3 EE shells, T4/T6 L/R L2 occupancy,
  T5/T7 L/R appearance+geometry — with T5/T7[28:32] (target-in-cube flag + local pos)
  ZEROED, exactly the fields FoveatedMap.blind_view() identifies as target-derived.
  T0 (target token) and T2 (base->target corridor) are excluded entirely.
  => the geo input is IDENTICAL for the full and blind token streams by construction, so
  the map_blind_prob draw in B1KInputs cannot leak target information through this route.
  Values are bounded with tanh(x/4) (the map_recon convention) before the MLP: occupancy/
  RGB/frac live in [0,1], staleness is clipped at 10, offsets are meters.

  1. models/pi0_config.py   map_geo_conditioning flag (default-off / inert) + validity
                            check (requires pi05 and the frozen K=8 token contract)
  2. models/pi0.py          zero-init MLP in __init__ (created AFTER all existing modules
                            so the rng stream of every existing param is unchanged; no
                            params when map_geo_conditioning=False -> run1b restore
                            bit-identical) + additive term in _embed_cond_extras.

  models/model.py, policies/b1k_policy.py, training/config.py, training/data_loader.py,
  training/weight_loaders.py need NO changes: the route consumes the EXISTING
  Observation.map_tokens field, which already flows train-side (map_tokens_full/blind
  repack + blind draw, gated on map_tokens_k>0) and serve-side (wrapper-supplied
  "map_tokens"). Warm-starting from run1b requires missing_regex including
  ".*map_geo.*" in the CheckpointWeightLoader.

Usage:  FORK_ROOT=/path/to/src/openpi python3 patch_map_adaln.py
        (default FORK_ROOT = /root/openpi_fork/src/openpi)
"""

import os
import py_compile

FORK = os.environ.get("FORK_ROOT", "/root/openpi_fork/src/openpi")

MARKER = "patch_map_adaln"


def patch(path, subs):
    s = open(path).read()
    if MARKER in s:
        print(f"SKIP {path}: already patched")
        return
    for pat, rep in subs:
        assert pat in s, f"anchor not found in {path}: {pat[:70]!r}"
        assert s.count(pat) == 1, f"anchor not unique in {path}: {pat[:70]!r}"
        s = s.replace(pat, rep, 1)
    compile(s, path, "exec")  # syntax-check BEFORE writing — a bad edit never lands on disk
    open(path, "w").write(s)
    py_compile.compile(path, doraise=True)
    print(f"patched {path}")


# ---- 1. models/pi0_config.py ----------------------------------------------------------
A_FIELD = '    pytorch_compile_mode: str | None = "max-autotune"\n'
A_CHECK = (
    "        if (self.point_conditioning or self.stage_conditioning) and not self.pi05:\n"
    "            raise ValueError(\n"
    '                "point_conditioning/stage_conditioning require pi05=True (they extend the adaRMS pathway)."\n'
    "            )\n"
)
patch(f"{FORK}/models/pi0_config.py", [
    (A_FIELD,
     "    # MAP-GEOMETRY AdaLN conditioning (patch_map_adaln.py): geometry channels of the\n"
     "    # (8,72) map tokens (rows T1/T3/T4/T6 + T5/T7 with target fields zeroed; T0/T2\n"
     "    # excluded) -> tanh(x/4) -> MLP (zero-init output) -> additive term on the SAME\n"
     "    # adaRMS conditioning vector as the timestep + target points. False => no params\n"
     "    # created, init and checkpoint restore bit-identical to baseline (run1b safe).\n"
     "    map_geo_conditioning: bool = False\n"
     + A_FIELD),
    (A_CHECK,
     A_CHECK
     + "        if self.map_geo_conditioning and (not self.pi05 or self.map_tokens_k != 8):\n"
     "            raise ValueError(\n"
     '                "map_geo_conditioning requires pi05=True and map_tokens_k == 8 (patch_map_adaln.py: "\n'
     '                "the geometry row/field selection is written against the frozen (8,72) token contract)."\n'
     "            )\n"),
])

# ---- 2. models/pi0.py — params + additive cond term -----------------------------------
INIT_ANCHOR = (
    "        # This attribute gets automatically set by model.train() and model.eval().\n"
    "        self.deterministic = True\n"
)
INIT_BLOCK = (
    "        # MAP-GEOMETRY AdaLN conditioning (patch_map_adaln.py): zero-init MLP over the\n"
    "        # geometry channels of the map tokens, additive into adarms_cond. Created AFTER\n"
    "        # every stock/optional module above so the rng stream (and thus the init) of all\n"
    "        # existing params is unchanged; map_geo_conditioning=False creates no params, so\n"
    "        # restoring run1b checkpoints is bit-identical. Warm-starting WITH the route\n"
    "        # needs missing_regex '.*map_geo.*' in the CheckpointWeightLoader.\n"
    '        self.map_geo_conditioning = getattr(config, "map_geo_conditioning", False)\n'
    "        if self.map_geo_conditioning:\n"
    "            _mg_width = action_expert_config.width\n"
    "            self.map_geo_mlp_in = nnx.Linear(6 * config.map_token_dim, _mg_width // 2, rngs=rngs)\n"
    "            self.map_geo_mlp_out = nnx.Linear(\n"
    "                _mg_width // 2, _mg_width, kernel_init=nnx.initializers.zeros_init(), rngs=rngs\n"
    "            )\n"
    "\n"
)
COND_ANCHOR = (
    "        if not parts:\n"
    "            return None\n"
)
COND_BLOCK = (
    "        # MAP-GEOMETRY AdaLN term (patch_map_adaln.py). Geometry rows [T1,T3,T4,T5,T6,T7]\n"
    "        # with the target-derived fields of T5/T7 ([28:32], blind_view convention) zeroed;\n"
    "        # identical for full and blind token streams by construction. tanh(x/4) bounds the\n"
    "        # mixed feature scales (map_recon convention). Zero-init output => exactly 0 at\n"
    "        # start of training; all-zero map tokens (the null/no-map convention) produce a\n"
    "        # learned constant 'blank map' embedding, consistent with the prefix null token.\n"
    '        if getattr(self, "map_geo_conditioning", False) and obs.map_tokens is not None:\n'
    "            import numpy as _mg_np\n"
    "            _mg = obs.map_tokens.astype(jnp.float32)[:, (1, 3, 4, 5, 6, 7), :]\n"
    "            _mg_mask = _mg_np.ones((6, obs.map_tokens.shape[-1]), _mg_np.float32)\n"
    "            _mg_mask[3, 28:32] = 0.0\n"
    "            _mg_mask[5, 28:32] = 0.0\n"
    "            _mg = _mg * jnp.asarray(_mg_mask)[None]\n"
    "            _mg = jnp.tanh(_mg / 4.0).reshape(batch_size, -1)\n"
    "            _mg = self.map_geo_mlp_in(_mg)\n"
    "            _mg = nnx.swish(_mg)\n"
    "            parts.append(self.map_geo_mlp_out(_mg))  # zero-init: exactly 0 at start of training\n"
)
patch(f"{FORK}/models/pi0.py", [
    (INIT_ANCHOR, INIT_BLOCK + INIT_ANCHOR),
    (COND_ANCHOR, COND_BLOCK + COND_ANCHOR),
])

print("no changes needed: models/model.py, policies/b1k_policy.py, training/config.py,")
print("  training/data_loader.py, training/weight_loaders.py (route reads the existing")
print("  Observation.map_tokens field; train repack + serve passthrough already exist)")
print("REMINDER: a map_geo_conditioning=True TrainConfig warm-started from run1b must set")
print("  CheckpointWeightLoader(..., missing_regex including '.*map_geo.*')")
print("ALL MAP-ADALN PATCHES APPLIED")

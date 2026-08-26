"""Patch the openpi fork with CONTEXT-STATE conditioning (task62/CONTEXT_STATES_SPEC_62.md).

Adds a float context vector c(t) (default 16-d: skill family one-hot, progress q(t), post-slice
flag, AG-held per arm, held-object class per arm) as an ADDITIVE AdaLN term on the action expert,
following the fork's own point/stage conditioning route (`_embed_cond_extras`) and the
patch_map_adaln.py conventions:

  * new Pi0Config flags default OFF -> no params created, init/restore of every existing
    checkpoint stays bit-identical;
  * params created AFTER every existing module in pi0.__init__ so the rng stream of all existing
    params is unchanged; output layer zero-init -> the term is exactly 0 at start of training;
  * new Observation field `context` through the trap-triple (field + from_dict + preprocess);
  * B1KInputs packs `context` (zeros sentinel when absent) so serve and train share one contract;
  * training/config.py repacks the parquet `context` column ONLY when the model consumes it
    (RepackTransform KeyErrors on absent columns -> every dataset in a mix must carry it);
  * modality dropout (patch_modality_dropout.py) also drops the context stream (fold_in 8207,
    independent of the 8206 streams) when modality_dropout_p > 0.

Anchored, assertion-guarded, idempotent. Warm-starting needs missing_regex '.*context_.*'.
Run with any python: FORK_ROOT env overrides the default /root/openpi_fork/src/openpi.
"""
import os, py_compile

FORK = os.environ.get("FORK_ROOT", "/root/openpi_fork/src/openpi")


def patch(path, edits, marker):
    s = open(path).read()
    if marker in s:
        print(f"already patched {path}"); return
    for pat, rep in edits:
        assert pat in s, f"anchor not found in {path}: {pat[:80]!r}"
        assert s.count(pat) == 1, f"anchor not unique in {path}: {pat[:80]!r}"
        s = s.replace(pat, rep, 1)
    open(path, "w").write(s)
    py_compile.compile(path, doraise=True)
    print(f"patched {path}")


# ---- 1. models/pi0_config.py ------------------------------------------------------------
A = "    stage_conditioning: bool = False\n"
patch(f"{FORK}/models/pi0_config.py", [(
    A,
    A + "    # CONTEXT-STATE conditioning (patch_context_cond.py): float context vector c(t) ->\n"
        "    # zero-init MLP -> additive adaRMS term (same route as point/stage conditioning).\n"
        "    # False => no params, bit-identical init/restore for every existing checkpoint.\n"
        "    context_conditioning: bool = False\n"
        "    context_dim: int = 16\n",
)], marker="context_conditioning: bool")

# ---- 2. models/model.py — trap-triple -----------------------------------------------------
F1 = '    stage_tokens: at.Int[ArrayT, "*b n"] | None = None\n'
F2 = '            stage_tokens=data.get("stage_tokens"),\n'
F3 = "        stage_tokens=observation.stage_tokens,\n"
patch(f"{FORK}/models/model.py", [
    (F1, F1 + "    # CONTEXT-STATE vector c(t) (patch_context_cond.py), shape [*b, context_dim]. Zeros = 'no context'.\n"
              '    context: at.Float[ArrayT, "*b c"] | None = None\n'),
    (F2, F2 + '            context=data.get("context"),\n'),
    (F3, F3 + "        context=observation.context,\n"),
], marker='context: at.Float[ArrayT, "*b c"]')

# ---- 3. models/pi0.py — params + AdaLN term + dropout ------------------------------------
INIT_ANCHOR = (
    "        # This attribute gets automatically set by model.train() and model.eval().\n"
    "        self.deterministic = True\n"
)
INIT_BLOCK = (
    "        # CONTEXT-STATE conditioning (patch_context_cond.py). Created AFTER every stock/\n"
    "        # optional module above so the rng stream of all existing params is unchanged;\n"
    "        # context_conditioning=False creates no params (bit-identical restore). Output\n"
    "        # zero-init => the AdaLN term is exactly 0 at start of training. Warm-start needs\n"
    "        # missing_regex '.*context_.*'.\n"
    '        self.context_conditioning = getattr(config, "context_conditioning", False)\n'
    "        if self.context_conditioning:\n"
    '            self.context_dim = getattr(config, "context_dim", 16)\n'
    "            self.context_mlp_in = nnx.Linear(self.context_dim, action_expert_config.width // 2, rngs=rngs)\n"
    "            self.context_mlp_out = nnx.Linear(\n"
    "                action_expert_config.width // 2, action_expert_config.width,\n"
    "                kernel_init=nnx.initializers.zeros_init(), rngs=rngs,\n"
    "            )\n"
    "\n"
)
COND_ANCHOR = "        if not parts:\n            return None\n        return sum(parts[1:], start=parts[0])\n"
COND_BLOCK = (
    "        # CONTEXT-STATE AdaLN term (patch_context_cond.py). Missing context => zeros =>\n"
    "        # a learned constant 'no context' embedding (consistent with the point null token).\n"
    '        if getattr(self, "context_conditioning", False):\n'
    "            _ctx = obs.context\n"
    "            if _ctx is None:\n"
    "                _ctx = jnp.zeros((batch_size, self.context_dim), dtype=jnp.float32)\n"
    "            _ctx = jnp.tanh(_ctx.astype(jnp.float32) / 4.0)\n"
    "            _ctx = self.context_mlp_in(_ctx)\n"
    "            _ctx = nnx.swish(_ctx)\n"
    "            parts.append(self.context_mlp_out(_ctx))  # zero-init: exactly 0 at start of training\n"
    + COND_ANCHOR
)
DROP_ANCHOR = "            _mdmasks = dict(observation.image_masks)\n"
DROP_BLOCK = (
    "            # CONTEXT-STATE dropout (patch_context_cond.py): independent stream, fold_in 8207\n"
    "            # so the 8206 streams (points/map/wrists) stay bit-identical.\n"
    '            if getattr(self, "context_conditioning", False) and observation.context is not None:\n'
    "                _mdkeepc = jax.random.bernoulli(jax.random.fold_in(rng, 8207), 1.0 - _mdp, (_mdb,))\n"
    "                observation = _mdc.replace(\n"
    "                    observation,\n"
    "                    context=observation.context * _mdkeepc[:, None].astype(observation.context.dtype),\n"
    "                )\n"
    + DROP_ANCHOR
)
patch(f"{FORK}/models/pi0.py", [
    (INIT_ANCHOR, INIT_BLOCK + INIT_ANCHOR),
    (COND_ANCHOR, COND_BLOCK),
    (DROP_ANCHOR, DROP_BLOCK),
], marker="self.context_conditioning = getattr")

# ---- 4. policies/b1k_policy.py — B1KInputs packing --------------------------------------
B1 = "    stage_conditioning: bool = False\n"
B2 = '            inputs["stage_tokens"] = stage_tokens\n'
patch(f"{FORK}/policies/b1k_policy.py", [
    (B1, B1 + "    # CONTEXT-STATE conditioning (patch_context_cond.py): pack the per-frame float context\n"
              "    # vector; zeros sentinel when absent (model embeds it as 'no context').\n"
              "    context_conditioning: bool = False\n"
              "    context_dim: int = 16\n"),
    (B2, B2 + "        if self.context_conditioning:\n"
              '            if data.get("context") is not None:\n'
              '                _ctx = np.asarray(data["context"], dtype=np.float32).reshape(-1)[: self.context_dim]\n'
              "                if _ctx.shape[0] < self.context_dim:\n"
              "                    _ctx = np.concatenate([_ctx, np.zeros(self.context_dim - _ctx.shape[0], np.float32)])\n"
              "            else:\n"
              "                _ctx = np.zeros(self.context_dim, dtype=np.float32)\n"
              '            inputs["context"] = _ctx\n'),
], marker="context_conditioning: bool = False\n    context_dim")

# ---- 5. training/config.py — repack + B1KInputs flags ------------------------------------
C1 = '        if stage_conditioning:\n            repack_mapping["stage_tokens"] = self.stage_tokens_key\n'
C2 = "                stage_conditioning=stage_conditioning,\n                map_tokens_k=map_tokens_k,\n"
patch(f"{FORK}/training/config.py", [
    (C1, C1 + '        # CONTEXT-STATE conditioning (patch_context_cond.py): repack the parquet "context"\n'
              "        # column only when the model consumes it (RepackTransform KeyErrors otherwise).\n"
              '        context_conditioning = getattr(model_config, "context_conditioning", False)\n'
              "        if context_conditioning:\n"
              '            repack_mapping["context"] = "context"\n'),
    (C2, C2 + "                context_conditioning=context_conditioning,\n"
              '                context_dim=getattr(model_config, "context_dim", 16),\n'),
], marker='repack_mapping["context"] = "context"')

for f in ("models/pi0.py", "models/pi0_config.py", "models/model.py", "policies/b1k_policy.py", "training/config.py"):
    py_compile.compile(f"{FORK}/{f}", doraise=True)
print("ALL CONTEXT-COND PATCHES APPLIED")
print("REMINDER: every dataset in a mix must carry a float32 'context' column of context_dim;")
print("          warm-start CheckpointWeightLoader needs missing_regex including '.*context_.*'")

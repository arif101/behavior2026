"""Expose the STAGE/PROGRESS head at inference (CONTEXT_STATES_SPEC_62 §2: live c(t) source).

patch_stage_head.py trains the head as an aux loss only; sample_actions returns a bare Actions
array. This adds:
  * Pi0.predict_stage(observation, actions): one prefix+suffix pass at flow time t=0 with
    x_t = the sampled actions (the same features the loss-side head sees, mean-pooled over the
    action tokens) -> float32 [b, stage_classes + 1] = [stage logits..., progress logit].
  * Policy: when the model has stage_head, jit predict_stage and attach
    outputs["stage_head"] (numpy [stage_classes + 1]) to every infer() result.
No new params; no change to training or to sample_actions. Idempotent, anchored, assertion-guarded.
"""
import os, py_compile
FORK = os.environ.get("FORK_ROOT", "/root/openpi_fork/src/openpi")


def patch(path, edits, marker):
    s = open(path).read()
    if marker in s:
        print(f"already patched {path}"); return
    for pat, rep in edits:
        assert s.count(pat) == 1, f"anchor not unique/found in {path}: {pat[:80]!r}"
        s = s.replace(pat, rep, 1)
    open(path, "w").write(s); py_compile.compile(path, doraise=True); print(f"patched {path}")


# ---- 1. models/pi0.py: predict_stage ------------------------------------------------------
A = "    def sample_actions(\n        self,\n        rng: at.KeyArrayLike,\n        observation: _model.Observation,\n"
NEW = '''    def predict_stage(self, observation: _model.Observation, actions: _model.Actions) -> at.Float[at.Array, "b c"]:
        """STAGE/PROGRESS head at inference (patch_stage_infer.py): [b, stage_classes+1] =
        [stage logits..., progress logit], from one prefix+suffix pass at flow time 0 with the
        sampled actions as x_t. Mirrors the loss-side feature (mean over the action tokens)."""
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, actions, jnp.zeros(batch_size, dtype=jnp.float32)
        )
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        feat = suffix_out[:, -self.action_horizon :, :].mean(axis=1)
        return self.stage_head_out(jax.nn.gelu(self.stage_head_in(feat))).astype(jnp.float32)

'''
patch(f"{FORK}/models/pi0.py", [(A, NEW + A)], marker="def predict_stage(")

# ---- 2. policies/policy.py: attach stage_head to infer() outputs --------------------------
P1 = "            self._sample_actions = nnx_utils.module_jit(model.sample_actions)\n            self._rng = rng or jax.random.key(0)\n"
P2 = '        outputs = {\n            "state": inputs["state"],\n            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),\n        }\n'
patch(f"{FORK}/policies/policy.py", [
    (P1, P1 + "            # STAGE/PROGRESS head at inference (patch_stage_infer.py)\n"
              '            self._predict_stage = nnx_utils.module_jit(model.predict_stage) if getattr(model, "stage_head", False) else None\n'),
    (P2, P2 + '        if not self._is_pytorch_model and getattr(self, "_predict_stage", None) is not None:\n'
              '            outputs["stage_head"] = self._predict_stage(observation, outputs["actions"])\n'),
], marker='outputs["stage_head"]')
print("ALL STAGE-INFER PATCHES APPLIED")

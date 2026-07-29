"""Add the Run-1 aux losses to the fork's pi0.py (FOVEATED_MEMORY_SPEC_v1, training-only).

Two decode heads, both supervised by target_points already present in every batch:

  aux_map : mean of the map-token PREFIX OUTPUTS -> target_points (6,)
            Anti-sink: the injected pathway must CARRY the target through the LLM.
            Skipped for blind-stream samples (their T0 token is zeroed by construction).
  aux_img : mean of the HEAD-camera image-token outputs (tokens [0:256], camera order
            base_0_rgb first) -> target_points (6,)
            Grounding: the VLM must localize the target VISUALLY — the gaze-KL effect
            without attention-capture surgery into the bridged gemma (Run-2 candidate).

Weight 0.1 each; per-arm masking via target_points_mask. Heads exist only when
map_tokens_k > 0 (K=0 bit-parity intact). Scope note: stage aux CUT from Run 1 (labels not
built); pixel-space decode deferred until wrist intrinsics are calibrated.
"""

import py_compile

P = "/root/openpi_fork/src/openpi/models/pi0.py"
s = open(P).read()
if "aux_map_head_in" in s:
    raise SystemExit("already patched")

init_anchor = (
    "            self.map_registers = nnx.Param(\n"
    "                nnx.initializers.normal(0.02)(rngs.params(), (self.map_k, paligemma_config.width))\n"
    "            )\n"
)
assert init_anchor in s, "init anchor not found (map slot must be patched first)"
s = s.replace(init_anchor, init_anchor +
    "            # Run-1 aux decode heads (training-only; see patch_aux_losses.py)\n"
    "            self.aux_map_head_in = nnx.Linear(paligemma_config.width, 64, rngs=rngs)\n"
    "            self.aux_map_head_out = nnx.Linear(64, 6, rngs=rngs)\n"
    "            self.aux_img_head_in = nnx.Linear(paligemma_config.width, 64, rngs=rngs)\n"
    "            self.aux_img_head_out = nnx.Linear(64, 6, rngs=rngs)\n", 1)

loss_anchor = (
    "        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])\n"
    "\n"
    "        return jnp.mean(jnp.square(v_t - u_t), axis=-1)\n"
)
assert loss_anchor in s, "compute_loss anchor not found"
s = s.replace(loss_anchor,
    "        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])\n"
    "        loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)\n"
    "\n"
    "        # FOVEATED MEMORY aux decodes (patch_aux_losses.py): anti-sink + visual grounding.\n"
    "        tp_raw = getattr(observation, \"target_points\", None)\n"
    "        if self.map_k > 0 and observation.map_tokens is not None and tp_raw is not None:\n"
    "            b = tp_raw.shape[0]\n"
    "            tp = tp_raw.reshape(b, -1)[:, :6].astype(jnp.float32)\n"
    "            tpm = getattr(observation, \"target_points_mask\", None)\n"
    "            w = (jnp.repeat(tpm.reshape(b, -1)[:, :2].astype(jnp.float32), 3, axis=-1)\n"
    "                 if tpm is not None else jnp.ones((b, 6), jnp.float32))\n"
    "            map_out = prefix_out[:, -self.map_k :, :].mean(axis=1)\n"
    "            img_out = prefix_out[:, :256, :].mean(axis=1)  # camera 0 = base_0_rgb (head)\n"
    "            pred_map = self.aux_map_head_out(nnx.swish(self.aux_map_head_in(map_out)))\n"
    "            pred_img = self.aux_img_head_out(nnx.swish(self.aux_img_head_in(img_out)))\n"
    "            not_blind = (jnp.abs(observation.map_tokens[:, 0, :]).sum(-1) > 0).astype(jnp.float32)\n"
    "            e_map = (jnp.abs(pred_map.astype(jnp.float32) - tp) * w).mean(-1) * not_blind\n"
    "            e_img = (jnp.abs(pred_img.astype(jnp.float32) - tp) * w).mean(-1)\n"
    "            loss = loss + (0.1 * (e_map + e_img)).astype(loss.dtype)[:, None]\n"
    "        return loss\n", 1)

compile(s, P, "exec")
open(P, "w").write(s)
py_compile.compile(P, doraise=True)
print("aux losses patched into fork pi0.py")

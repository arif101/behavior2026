"""Aux round 2 (answers 'why wait'): stage CE + per-camera patch-heatmap CE decode heads.

Adds on top of patch_aux_losses.py (must be applied first):

  aux_stage : mean prefix outputs -> 4-way stage CE (w=0.05). Geometry-derived labels;
              Larchenko lineage (stage structure ~2x in BEHAVIOR-25).
  aux_heat  : per camera (head/base_0, left wrist, right wrist), a Linear(width->1) over that
              camera's 256 image tokens -> 16x16 patch logits -> CE toward the label patch,
              masked by per-camera visibility (w=0.1, mean over visible cams).
              THE on-mechanism loss: wrist tokens must answer "where is the button in MY view
              + is it visible" — measured demo visibility: head 89.9%, wristL 18.7%, wristR 48%.

New Observation fields stage (int) + aux_pixels (9,) — full trap-triple in model.py, plus
b1k_policy passthrough and config repack. Labels: add_stage_pixel_labels.py columns.
"""

import py_compile

FORK = "/root/openpi_fork/src/openpi"


def patch(path, subs):
    s = open(path).read()
    for pat, rep in subs:
        if rep in s:
            print(f"  (already applied in {path})")
            continue
        assert pat in s, f"anchor not found in {path}: {pat[:70]!r}"
        s = s.replace(pat, pat + rep, 1)
    compile(s, path, "exec")
    open(path, "w").write(s)
    py_compile.compile(path, doraise=True)
    print(f"patched {path}")


# ---- model.py: two new fields, trap-triple ---------------------------------------------
patch(f"{FORK}/models/model.py", [
    ('    map_tokens: at.Float[ArrayT, "*b k d"] | None = None\n',
     '    # Aux-v2 labels: 4-way stage id + [head_u,v,vis, wl_u,v,vis, wr_u,v,vis] (normalized).\n'
     '    stage: at.Int[ArrayT, "*b"] | None = None\n'
     '    aux_pixels: at.Float[ArrayT, "*b 9"] | None = None\n'),
    ('            map_tokens=data.get("map_tokens"),\n',
     '            stage=data.get("stage"),\n            aux_pixels=data.get("aux_pixels"),\n'),
    ('        map_tokens=observation.map_tokens,\n',
     '        stage=observation.stage,\n        aux_pixels=observation.aux_pixels,\n'),
])

# ---- b1k_policy.py: passthrough --------------------------------------------------------
patch(f"{FORK}/policies/b1k_policy.py", [(
    '                inputs["map_tokens"] = np.asarray(data[_key], np.float32).reshape(\n'
    '                    self.map_tokens_k, -1)\n',
    '            if data.get("stage") is not None:\n'
    '                inputs["stage"] = np.int32(np.asarray(data["stage"]).reshape(-1)[0])\n'
    '            if data.get("aux_pixels") is not None:\n'
    '                inputs["aux_pixels"] = np.asarray(data["aux_pixels"], np.float32).reshape(9)\n'
)])

# ---- config.py: repack -----------------------------------------------------------------
patch(f"{FORK}/training/config.py", [(
    '            repack_mapping["map_tokens_full"] = "map_tokens_full"\n'
    '            repack_mapping["map_tokens_blind"] = "map_tokens_blind"\n',
    '            repack_mapping["stage"] = "stage"\n'
    '            repack_mapping["aux_pixels"] = "aux_pixels"\n'
)])

# ---- pi0.py: heads + losses ------------------------------------------------------------
patch(f"{FORK}/models/pi0.py", [
    ('            self.aux_img_head_in = nnx.Linear(paligemma_config.width, 64, rngs=rngs)\n'
     '            self.aux_img_head_out = nnx.Linear(64, 6, rngs=rngs)\n',
     '            self.aux_stage_in = nnx.Linear(paligemma_config.width, 64, rngs=rngs)\n'
     '            self.aux_stage_out = nnx.Linear(64, 4, rngs=rngs)\n'
     '            self.aux_heat = nnx.Linear(paligemma_config.width, 1, rngs=rngs)\n'),
    ('            loss = loss + (0.1 * (e_map + e_img)).astype(loss.dtype)[:, None]\n',
     '            # aux-v2 (patch_aux_v2.py): stage CE + per-camera patch-heatmap CE\n'
     '            if observation.stage is not None:\n'
     '                sl = self.aux_stage_out(nnx.swish(self.aux_stage_in(\n'
     '                    prefix_out[:, : 3 * 256, :].mean(axis=1)))).astype(jnp.float32)\n'
     '                lse = jax.nn.log_softmax(sl, axis=-1)\n'
     '                st = jnp.clip(observation.stage.reshape(-1), 0, 3)\n'
     '                e_stage = -jnp.take_along_axis(lse, st[:, None], axis=-1)[:, 0]\n'
     '                loss = loss + (0.05 * e_stage).astype(loss.dtype)[:, None]\n'
     '            if observation.aux_pixels is not None:\n'
     '                ap = observation.aux_pixels.astype(jnp.float32)\n'
     '                e_heat = 0.0\n'
     '                n_vis = 0.0\n'
     '                for ci in range(3):  # camera order: base_0(head), left wrist, right wrist\n'
     '                    logits = self.aux_heat(prefix_out[:, ci * 256 : (ci + 1) * 256, :])\n'
     '                    logits = logits[..., 0].astype(jnp.float32)\n'
     '                    u, v, vis = ap[:, 3 * ci], ap[:, 3 * ci + 1], ap[:, 3 * ci + 2]\n'
     '                    pu = jnp.clip((u * 16).astype(jnp.int32), 0, 15)\n'
     '                    pv = jnp.clip((v * 16).astype(jnp.int32), 0, 15)\n'
     '                    tgt = pv * 16 + pu\n'
     '                    lsm = jax.nn.log_softmax(logits, axis=-1)\n'
     '                    ce = -jnp.take_along_axis(lsm, tgt[:, None], axis=-1)[:, 0]\n'
     '                    e_heat = e_heat + ce * vis\n'
     '                    n_vis = n_vis + vis\n'
     '                e_heat = e_heat / jnp.maximum(n_vis, 1.0)\n'
     '                loss = loss + (0.1 * e_heat).astype(loss.dtype)[:, None]\n'),
])
print("AUX V2 PATCHES APPLIED")

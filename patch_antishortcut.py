"""Anti-shortcut input-channel dropout (FOVEATED_MEMORY_SPEC_v1), train-only by construction.

Lives in compute_loss (never in the serve path):
  - AdaLN point dropout: P(drop)=0.4 per sample -> target_points zeroed + mask False (the
    trained sentinel). Forces the policy to keep its own perception alive instead of leaning
    on the injected point — the fix for "conditioning as a crutch".
  - Camera dropout: each wrist camera masked with P=0.2 per sample (head kept: it is the map
    and affordance source at serve). Forces cross-camera redundancy.
Config-gated (Pi0Config.anti_shortcut, ON only in pi05_radio_map) for a clean A/B.
Note: the rng split changes 3->6, which reshuffles noise/time draws — training-neutral.
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


patch(f"{FORK}/models/pi0_config.py", [(
    "    map_token_dim: int = 72\n",
    "    # Anti-shortcut input-channel dropout in compute_loss (train-only): P(drop point)=0.4,\n"
    "    # P(drop each wrist cam)=0.2. See patch_antishortcut.py.\n"
    "    anti_shortcut: bool = False\n",
)])

patch(f"{FORK}/models/pi0.py", [
    ("        self.map_k = getattr(config, \"map_tokens_k\", 0)\n",
     "        self.anti_shortcut = getattr(config, \"anti_shortcut\", False)\n"),
])

# compute_loss: widen the rng split + insert the dropout block after preprocessing
P = f"{FORK}/models/pi0.py"
s = open(P).read()
if "ANTI-SHORTCUT" in s:
    print("compute_loss already patched")
else:
    a = ("        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)\n"
         "        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)\n")
    assert a in s, "compute_loss rng anchor not found"
    b = ("        preprocess_rng, noise_rng, time_rng, _as_drop, _as_wl, _as_wr = jax.random.split(rng, 6)\n"
         "        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)\n"
         "\n"
         "        # ANTI-SHORTCUT (patch_antishortcut.py): train-only input-channel dropout.\n"
         "        if self.anti_shortcut and train:\n"
         "            import dataclasses as _dc\n"
         "            _b = observation.state.shape[0]\n"
         "            if getattr(observation, \"target_points\", None) is not None:\n"
         "                _keep = jax.random.bernoulli(_as_drop, 0.6, (_b,))\n"
         "                _tp = observation.target_points * _keep[:, None, None].astype(observation.target_points.dtype)\n"
         "                _tpm = observation.target_points_mask\n"
         "                if _tpm is not None:\n"
         "                    _tpm = jnp.logical_and(_tpm, _keep[:, None])\n"
         "                observation = _dc.replace(observation, target_points=_tp, target_points_mask=_tpm)\n"
         "            _masks = dict(observation.image_masks)\n"
         "            for _r, _nm in ((_as_wl, \"left_wrist_0_rgb\"), (_as_wr, \"right_wrist_0_rgb\")):\n"
         "                if _nm in _masks:\n"
         "                    _kc = jax.random.bernoulli(_r, 0.8, _masks[_nm].shape)\n"
         "                    _masks[_nm] = jnp.logical_and(_masks[_nm], _kc)\n"
         "            observation = _dc.replace(observation, image_masks=_masks)\n")
    s = s.replace(a, b, 1)
    compile(s, P, "exec")
    open(P, "w").write(s)
    py_compile.compile(P, doraise=True)
    print("compute_loss anti-shortcut block inserted")

# config.py: enable in pi05_radio_map only
C = f"{FORK}/training/config.py"
s = open(C).read()
if "anti_shortcut=True" in s:
    print("config already enables anti_shortcut")
else:
    i = s.index('name="pi05_radio_map"')
    j = s.index("map_tokens_k=8,\n", i)
    assert j > 0
    s = s[:j] + "map_tokens_k=8,\n            anti_shortcut=True,\n" + s[j + len("map_tokens_k=8,\n"):]
    compile(s, C, "exec")
    open(C, "w").write(s)
    py_compile.compile(C, doraise=True)
    print("pi05_radio_map: anti_shortcut=True")
print("ANTI-SHORTCUT PATCHES APPLIED")

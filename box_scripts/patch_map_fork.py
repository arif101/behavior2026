"""Patch the openpi fork with the FOVEATED MEMORY train stack (Run-1, FOVEATED_MEMORY_SPEC_v1).

Anchored, assertion-guarded edits (the _patch_*.py convention — never blind-overwrite fork
files, they carry box-only changes). All files py_compile'd afterwards.

  1. models/pi0_config.py  map_tokens_k / map_token_dim fields
  2. models/model.py       Observation.map_tokens + from_dict + preprocess passthrough
                           (three places per new Obs field — the passthrough-drop trap)
  3. models/pi0.py         zero-init projection + registers in __init__, injection in
                           embed_prefix after text (ar_mask=False, RoPE-safe tail)
  4. policies/b1k_policy.py  B1KInputs: map_tokens_k/map_blind_prob fields + train-time
                           full-vs-blind draw / serve-time passthrough
  5. training/config.py    map-column repack + B1KInputs kwarg + new config pi05_radio_map
                           (clone of pi05_radio_gate: K=8, augmented dataset, warm-start
                           49999 with fresh .*map_.* params)
"""

import py_compile
import re
import sys

FORK = "/root/openpi_fork/src/openpi"


def patch(path, subs):
    s = open(path).read()
    if "map_tokens" in s:
        print(f"SKIP {path}: already patched")
        return
    for pat, rep, literal in subs:
        if literal:
            assert pat in s, f"anchor not found in {path}: {pat[:70]!r}"
            s = s.replace(pat, rep, 1)
        else:
            s2, n = re.subn(pat, rep, s, count=1)
            assert n == 1, f"regex anchor not found in {path}: {pat[:70]!r}"
            s = s2
    compile(s, path, "exec")   # syntax-check BEFORE writing — a bad edit never lands on disk
    open(path, "w").write(s)
    py_compile.compile(path, doraise=True)
    print(f"patched {path}")


# ---- 1. pi0_config.py -----------------------------------------------------------------
patch(f"{FORK}/models/pi0_config.py", [(
    r"(    discrete_state_input: bool = None[^\n]*\n)",
    r"\1\n"
    r"    # FOVEATED MEMORY: K map soft tokens appended to the prefix after text (0 = off;\n"
    r"    # the K=0 path creates no params and is bit-identical to baseline).\n"
    r"    map_tokens_k: int = 0\n"
    r"    map_token_dim: int = 72\n",
    False,
)])

# ---- 2. model.py ----------------------------------------------------------------------
patch(f"{FORK}/models/model.py", [
    (r'(    token_loss_mask:[^\n]*= None\n)',
     '\\1\n    # FOVEATED MEMORY: map soft-token features (K, D) from FoveatedMap.query().\n'
     '    map_tokens: at.Float[ArrayT, "*b k d"] | None = None\n', False),
    (r'(token_loss_mask=data\.get\("token_loss_mask"\),\n)',
     '\\1            map_tokens=data.get("map_tokens"),\n', False),
    (r'(token_loss_mask=observation\.token_loss_mask,\n)',
     '\\1        map_tokens=observation.map_tokens,\n', False),
])

# ---- 3. pi0.py ------------------------------------------------------------------------
patch(f"{FORK}/models/pi0.py", [
    (r"(\n        self\.action_out_proj = nnx\.Linear\()",
     "\n        # FOVEATED MEMORY: zero-init out-proj => warm-start tokens EQUAL the 0.02\n"
     "        # registers exactly; map_tokens_k == 0 creates no params (bit-parity).\n"
     "        self.map_k = getattr(config, \"map_tokens_k\", 0)\n"
     "        if self.map_k > 0:\n"
     "            self.map_proj_in = nnx.Linear(config.map_token_dim, 256, rngs=rngs)\n"
     "            self.map_proj_out = nnx.Linear(\n"
     "                256, paligemma_config.width, kernel_init=nnx.initializers.zeros_init(), rngs=rngs\n"
     "            )\n"
     "            self.map_registers = nnx.Param(\n"
     "                nnx.initializers.normal(0.02)(rngs.params(), (self.map_k, paligemma_config.width))\n"
     "            )\n"
     r"\1", False),
    (r"(ar_mask \+= \[False\] \* tokenized_inputs\.shape\[1\]\n)",
     "\\1\n"
     "        # FOVEATED MEMORY: K map tokens after text; ar_mask=False (full prefix\n"
     "        # attention — perception CAN see the map), tail position = RoPE-safe.\n"
     "        if getattr(self, \"map_k\", 0) > 0 and obs.map_tokens is not None:\n"
     "            _mt = nnx.swish(self.map_proj_in(obs.map_tokens))\n"
     "            _mt = self.map_proj_out(_mt) + self.map_registers\n"
     "            tokens.append(_mt)\n"
     "            input_mask.append(jnp.ones(_mt.shape[:2], dtype=jnp.bool_))\n"
     "            ar_mask += [False] * _mt.shape[1]\n", False),
])

# ---- 4. b1k_policy.py -----------------------------------------------------------------
patch(f"{FORK}/policies/b1k_policy.py", [
    (r"(    stage_conditioning: bool = False\n)",
     r"\1\n    # FOVEATED MEMORY: emit (K, D) map tokens. Train: draw full-vs-blind columns\n"
     r"    # (blind = anti-shortcut stream). Serve: pass through a supplied \"map_tokens\".\n"
     r"    map_tokens_k: int = 0\n    map_blind_prob: float = 0.3\n", False),
    (r"(            inputs\[\"target_points\"\] = points\n            inputs\[\"target_points_mask\"\] = mask\n)",
     "\\1\n"
     "        if self.map_tokens_k > 0:\n"
     "            if data.get(\"map_tokens\") is not None:          # serve path (online map)\n"
     "                inputs[\"map_tokens\"] = np.asarray(data[\"map_tokens\"], np.float32).reshape(\n"
     "                    self.map_tokens_k, -1)\n"
     "            elif data.get(\"map_tokens_full\") is not None:    # train path (dataset columns)\n"
     "                _key = \"map_tokens_blind\" if np.random.random() < self.map_blind_prob else \"map_tokens_full\"\n"
     "                inputs[\"map_tokens\"] = np.asarray(data[_key], np.float32).reshape(\n"
     "                    self.map_tokens_k, -1)\n", False),
])

# ---- 5. training/config.py ------------------------------------------------------------
cfgp = f"{FORK}/training/config.py"
s = open(cfgp).read()
if "pi05_radio_map" in s:
    print(f"SKIP {cfgp}: already patched")
    sys.exit(0)

anchor = 'if stage_conditioning:\n            repack_mapping["stage_tokens"] = self.stage_tokens_key\n'
assert anchor in s, "repack anchor not found"
s = s.replace(anchor, anchor +
              '        map_tokens_k = getattr(model_config, "map_tokens_k", 0)\n'
              '        if map_tokens_k > 0:\n'
              '            repack_mapping["map_tokens_full"] = "map_tokens_full"\n'
              '            repack_mapping["map_tokens_blind"] = "map_tokens_blind"\n', 1)

b1k_anchor = ("            b1k_policy.B1KInputs(\n"
              "                robot_config=robot_config,\n"
              "                model_type=model_config.model_type,\n"
              "                point_conditioning=point_conditioning,\n"
              "                stage_conditioning=stage_conditioning,\n")
assert b1k_anchor in s, "B1KInputs anchor not found"
s = s.replace(b1k_anchor, b1k_anchor + "                map_tokens_k=map_tokens_k,\n", 1)

# clone the pi05_radio_gate TrainConfig entry (paren-balanced)
i = s.index('name="pi05_radio_gate"')
start = s.rindex("TrainConfig(", 0, i)
depth = 0
end = None
for j in range(start, len(s)):
    if s[j] == "(":
        depth += 1
    elif s[j] == ")":
        depth -= 1
        if depth == 0:
            end = j + 1
            break
assert end, "unbalanced parens in radio_gate block"
block = s[start:end]
new = block.replace('name="pi05_radio_gate"', 'name="pi05_radio_map"')
new = new.replace('exp_name="radio_gate"', 'exp_name="radio_map"')
assert "point_noise_std=0.02,\n" in new
new = new.replace("point_noise_std=0.02,\n",
                  "point_noise_std=0.02,\n            map_tokens_k=8,\n", 1)
assert 'dataset_root="/root/b1k_radio2"' in new
new = new.replace('dataset_root="/root/b1k_radio2"', 'dataset_root="/root/b1k_radio_map"')
# warm-start Run 1 from ckpt 49999 (params already contain trained point_ AdaLN); only the
# map params are new. Path is a symlink the launch script points at the restored checkpoint.
new = re.sub(r'weight_loader=weight_loaders\.CheckpointWeightLoader\([^)]*\)',
             'weight_loader=weight_loaders.CheckpointWeightLoader(\n'
             '            "/root/warmstart_49999/params",\n'
             '            missing_regex=".*lora.*|.*map_.*",  # fresh zero-init map params\n'
             '        )', new, count=1)
suffix = s[end:end + 2]
assert suffix.startswith(","), f"expected comma after block, got {suffix!r}"
s = s[:end + 1] + "\n    " + new + "," + s[end + 1:]
open(cfgp, "w").write(s)
py_compile.compile(cfgp, doraise=True)
print(f"patched {cfgp} (+pi05_radio_map)")
print("ALL FORK PATCHES APPLIED")

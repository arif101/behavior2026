"""Add the `pi05_t62_ctx` TrainConfig — task-62 context-conditioned fine-tune (CONTEXT_STATES_SPEC_62).

Cloned from pi05_radio_run2 (paren-balanced), single-variable-change discipline vs run-2 except
where task-62 has no labels yet:
  * context_conditioning=True, context_dim=16   (patch_context_cond.py)
  * stage_head=True, stage_classes=8            (family one-hot labels in the `stage` column)
  * point_conditioning=False, map_tokens_k=0, map_geo_conditioning=False, depth_aux=False
    (task-62 dataset v0 carries no target_points / map_tokens / gt_depth_ds columns;
    RepackTransform would KeyError on them)
  * modality_dropout_p=0.2 (drops wrists + context), anti_shortcut=False
  * dataset_root=/root/b1k_t62, repo_id=b1k_t62 (norm stats under assets/b1k_t62)
  * warm start from run-2 params (/root/warmstart_t62/params); missing_regex adds .*context_.*
Idempotent. FORK_ROOT env overrides the fork path.
"""
import os, py_compile
FORK = os.environ.get("FORK_ROOT", "/root/openpi_fork/src/openpi")
P = f"{FORK}/training/config.py"
s = open(P).read()
if 'name="pi05_t62_ctx"' in s:
    print("already patched"); raise SystemExit
i = s.index('name="pi05_radio_run2"')
start = s.rindex("    TrainConfig(", 0, i)
depth = 0; end = None
for j in range(start, len(s)):
    if s[j] == "(": depth += 1
    elif s[j] == ")":
        depth -= 1
        if depth == 0: end = j + 1; break
assert end, "unbalanced parens in run2 block"
blk = s[start:end]
if s[end:end + 1] == ",": end += 1
new = blk.replace('name="pi05_radio_run2"', 'name="pi05_t62_ctx"', 1)
subs = [
    ("point_conditioning=True,", "point_conditioning=False,"),
    ("map_tokens_k=8,", "map_tokens_k=0,"),
    ("map_geo_conditioning=True,", "map_geo_conditioning=False,"),
    ("depth_aux=True,", "depth_aux=False,"),
    ("stage_head=True,", "stage_head=True,\n            stage_classes=8,\n            context_conditioning=True,\n            context_dim=16,"),
    ('repo_id="b1k_radio",', 'repo_id="b1k_t62",'),
    ('dataset_root="/root/b1k_radio_run2mix",', 'dataset_root="/root/b1k_t62",'),
    ('"/root/warmstart_run2/params",', '"/root/warmstart_t62/params",'),
    ('missing_regex=".*lora.*|.*stage_head.*|.*map_geo.*|.*depth_aux.*",', 'missing_regex=".*lora.*|.*stage_head.*|.*map_geo.*|.*depth_aux.*|.*context_.*",'),
    ('exp_name="radio_run2",', 'exp_name="t62_ctx",'),
    ("num_train_steps=50_000,", "num_train_steps=30_000,"),
    ("save_interval=3_650,", "save_interval=2_500,"),
]
for a, b in subs:
    assert a in new, f"anchor missing in run2 block: {a!r}"
    new = new.replace(a, b, 1)
s = s[:end] + "\n" + new + ("," if not new.rstrip().endswith(",") else "") + s[end:]
open(P, "w").write(s); py_compile.compile(P, doraise=True)
print("pi05_t62_ctx TrainConfig added")

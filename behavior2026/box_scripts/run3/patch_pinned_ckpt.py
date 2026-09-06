"""Patch openpi/training/checkpoints.py::save_state to transfer arrays through PINNED host
memory (orbax PyTreeSave(enable_pinned_host_transfer=True)).

Why (measured 2026-09-06, single A100-80GB trainer): with the default pageable path the
train_state item's device->host transfer crawled at ~10 MB/s (one thread, kernel time;
a 40 GB train_state = ~70 min per save, the 20-step smoke never finished its final save)
while the params item finished in 24 s. save_test.py: pinned = 33 s total for the same
state; raw np.asarray on this box = 0.63 GB/s pageable vs 2.5 GB/s pinned. Run-2's 2-GPU
box never hit this (3 min saves). Restore path untouched.
Usage: FORK_ROOT=/root/openpi_fork/src/openpi python3 patch_pinned_ckpt.py
"""
import os
import py_compile

FORK = os.environ.get("FORK_ROOT", "/root/openpi_fork/src/openpi")
MARKER = "patch_pinned_ckpt"
path = f"{FORK}/training/checkpoints.py"
s = open(path).read()
if MARKER in s:
    print(f"SKIP {path}: already patched")
    raise SystemExit(0)
OLD = (
    '    items = {\n'
    '        "assets": save_assets,\n'
    '        "train_state": train_state,\n'
    '        "params": {"params": params},\n'
    '    }\n'
    '    checkpoint_manager.save(step, items)\n'
)
NEW = (
    f'    # {MARKER}: pinned-host D2H transfer (default pageable path crawled at ~10 MB/s on a\n'
    '    # single-GPU box: 70 min per save; pinned = 33 s). Same on-disk format; restore unchanged.\n'
    '    checkpoint_manager.save(\n'
    '        step,\n'
    '        args=ocp.args.Composite(\n'
    '            assets=CallbackSave(save_assets),\n'
    '            train_state=ocp.args.PyTreeSave(train_state, enable_pinned_host_transfer=True),\n'
    '            params=ocp.args.PyTreeSave({"params": params}, enable_pinned_host_transfer=True),\n'
    '        ),\n'
    '    )\n'
)
assert s.count(OLD) == 1, "anchor not found/unique in checkpoints.py"
s = s.replace(OLD, NEW, 1)
compile(s, path, "exec")
open(path, "w").write(s)
py_compile.compile(path, doraise=True)
print(f"patched {path}: pinned-host checkpoint transfer")

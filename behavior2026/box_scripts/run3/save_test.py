"""Standalone orbax save timing for a Run-3 TrainState (no training loop) — diagnoses the
single-A100 checkpoint stall (train_state D2H at ~10 MB/s under the default path).
  cd /root/openpi_fork && .venv/bin/python save_test.py <default|pinned|noparallel>
"""
import dataclasses
import os
import shutil
import sys
import threading
import time

sys.path.insert(0, "/root/openpi_fork/scripts")
import jax
import orbax.checkpoint as ocp
from orbax.checkpoint._src.serialization import type_handlers as th

mode = sys.argv[1]
if mode == "noparallel":
    th.register_type_handler(jax.Array, th.ArrayHandler(use_replica_parallel=False), override=True)
    print("registered ArrayHandler(use_replica_parallel=False)", flush=True)

import train as T  # scripts/train.py
import openpi.shared.array_typing as at
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.sharding as sharding


def busiest():
    best = (0, 0, 0)
    for t in os.listdir("/proc/self/task"):
        try:
            f = open(f"/proc/self/task/{t}/stat").read().split(")")[-1].split()
        except Exception:
            continue
        u, s = int(f[11]), int(f[12])
        if u + s > best[0] + best[1]:
            best = (u, s, int(t))
    return best


stop = False


def sampler():
    while not stop:
        rss = int(open("/proc/self/status").read().split("VmRSS:")[1].split()[0]) // 1024
        u, s, tid = busiest()
        print(f"  [sampler {time.strftime('%H:%M:%S')}] rss {rss} MB busiest tid {tid} utime {u / 100:.0f}s stime {s / 100:.0f}s", flush=True)
        time.sleep(15)


threading.Thread(target=sampler, daemon=True).start()
cfg = _config.get_config("pi05_radio_run3_a2")
cfg = dataclasses.replace(cfg, exp_name=f"savetest_{mode}", overwrite=True)
mesh = sharding.make_mesh(cfg.fsdp_devices)
t0 = time.time()
train_state, _ = T.init_train_state(cfg, jax.random.key(0), mesh, resume=False)
print(f"init_train_state {time.time() - t0:.0f}s", flush=True)
mngr, _ = _checkpoints.initialize_checkpoint_dir(cfg.checkpoint_dir, keep_period=cfg.keep_period, overwrite=True, resume=False)
with at.disable_typechecking():
    ts, params = _checkpoints._split_params(train_state)
t0 = time.time()
if mode == "pinned":
    mngr.save(1, args=ocp.args.Composite(
        assets=_checkpoints.CallbackSave(lambda d: None),
        train_state=ocp.args.PyTreeSave(ts, enable_pinned_host_transfer=True),
        params=ocp.args.PyTreeSave({"params": params}, enable_pinned_host_transfer=True),
    ))
else:
    mngr.save(1, {"assets": lambda d: None, "train_state": ts, "params": {"params": params}})
t1 = time.time()
print(f"SAVE blocking part {t1 - t0:.0f}s", flush=True)
mngr.wait_until_finished()
t2 = time.time()
print(f"SAVE async part {t2 - t1:.0f}s  TOTAL {t2 - t0:.0f}s", flush=True)
sz = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(str(cfg.checkpoint_dir)) for f in fs) / 1e9
print(f"SAVETEST_OK mode={mode} total_s={t2 - t0:.0f} size_GB={sz:.1f}", flush=True)
stop = True
shutil.rmtree(str(cfg.checkpoint_dir), ignore_errors=True)

"""Peak-memory + step-time probe for a Run-3 arm config on THIS box: model-spec batches
(fake_obs/fake_act), random init (NoOp loader), the same train_step / jit / sharding as
scripts/train.py — no data loader, no checkpoint. The generic fake-data path
(--data.repo_id fake) does NOT survive the B1K repack (KeyError 'action'), hence this.
Run under the launch flags you intend to use (XLA_PYTHON_CLIENT_MEM_FRACTION=0.92).
Needs a config `pi05_radio_run3_memprobe` = _run3_cfg(...) with weight_loader=NoOpWeightLoader.
  cd /root/openpi_fork && .venv/bin/python memprobe_direct.py <batch> <steps>
"""
import functools
import sys
import time

sys.path.insert(0, "/root/openpi_fork/scripts")
import jax
import train as T  # scripts/train.py
import openpi.training.config as _config
import openpi.training.sharding as sharding

cfg = _config.get_config("pi05_radio_run3_memprobe")
BS = int(sys.argv[1]) if len(sys.argv) > 1 else 32
N = int(sys.argv[2]) if len(sys.argv) > 2 else 40
rng = jax.random.key(cfg.seed)
train_rng, init_rng = jax.random.split(rng)
mesh = sharding.make_mesh(cfg.fsdp_devices)
data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
train_state, train_state_sharding = T.init_train_state(cfg, init_rng, mesh, resume=False)
print("params (B):", sum(x.size for x in jax.tree.leaves(train_state.params)) / 1e9, flush=True)
obs, act = cfg.model.fake_obs(BS), cfg.model.fake_act(BS)
batch = jax.tree.map(lambda x: jax.device_put(x, data_sharding), (obs, act))
ptrain_step = jax.jit(
    functools.partial(T.train_step, cfg),
    in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
    out_shardings=(train_state_sharding, replicated_sharding),
    donate_argnums=(1,),
)
t0 = time.time()
train_state, info = ptrain_step(train_rng, train_state, batch)
jax.block_until_ready(info)
loss0 = float(info["loss"])
print(f"JIT+step0 {time.time() - t0:.1f}s loss {loss0:.4f}", flush=True)
t1 = time.time()
for _ in range(N):
    train_state, info = ptrain_step(train_rng, train_state, batch)
jax.block_until_ready(info)
dt = (time.time() - t1) / N
print(f"MEMPROBE_OK batch={BS} steady {dt:.3f} s/step (pure compute, no data loading) loss {float(info['loss']):.4f}", flush=True)

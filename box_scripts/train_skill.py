"""Contact-skill training driver (spec §4/§7): one demo per process (Isaac env-reuse leaks;
chunk like the collector), cycled by an outer loop. Stage-0 smoke: `--demo-id 10 --stage 0
--max-env-steps 40000` — smoke bar (§7): rolling success ≥50% in ≤2 h.

Per env step: 1 RLPD update round (UTD critic updates inside). Online episodes go to the
online buffer; prior buffer = converter output MINUS held-out demos (bank json). Rolling
success over the last 20 episodes prints every episode; checkpoints + stats every 25
episodes to /root/skill_ckpts/ (push to HF at milestones — checkpoints never live only on
a rented box).

Run INSIDE the behavior env:
  PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 OMNIGIBSON_HEADLESS=1 \
    python -u train_skill.py --demo-id 10 --stage 0
"""

import argparse
import json
import os
import time

import numpy as np

os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-id", type=int, required=True)
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--max-env-steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--utd", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-shaping", action="store_true",
                    help="sparse-only arm (§4: report primary at sparse too)")
    ap.add_argument("--stop-at", type=float, default=0.5,
                    help="early-stop when rolling20 >= this (0 disables; stage-0 smoke "
                         "used 0.5; curriculum chunks want 0 or ~0.95)")
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    ap.add_argument("--out", default="/root/skill_ckpts")
    ap.add_argument("--buffer-path", default="/root/online_buffer.npz",
                    help="persist the online buffer across chunks (2026-08-13: per-chunk "
                         "fresh buffers made every new scene start from zero — only "
                         "weights carried; empty string disables)")
    a = ap.parse_args()

    import omnigibson as og  # noqa: F401
    from omnigibson.macros import gm
    gm.HEADLESS = True
    gm.ENABLE_TRANSITION_RULES = False
    from omnigibson.envs.hdf5_data_wrapper import HDF5PlaybackWrapper

    from reverse_curriculum_collect import EVAL_PROPRIO_KEYS  # noqa: F401
    from rlpd_sac import RLPD, ReplayBuffer, load_meta
    from skill_env_wrapper import SkillCommitEnv, STACK, PER_FRAME  # noqa: F401

    bank_all = json.load(open("/root/skill_start_bank.json"))
    # Reverse-curriculum start mixing (2026-08-13, after the press-10 cliff): the frontier
    # rung alone both over-specializes and forgets mastered rungs (exclusive -25 training
    # degraded the actor below its mastered -5 rung; exclusive -10 then failed 0/3 from a
    # clean ckpt). Sample ~50% frontier (stage == S) / 50% mastered (stage < S) via
    # duplication (the wrapper picks uniformly from the list).
    frontier = [e for e in bank_all["entries"]
                if e["demo"] == a.demo_id and e["stage"] == a.stage]
    mastered = [e for e in bank_all["entries"]
                if e["demo"] == a.demo_id and e["stage"] < a.stage]
    entries = frontier * max(1, len(mastered)) + mastered
    assert frontier, f"no bank entries for demo {a.demo_id} stage {a.stage}"
    held_out = bank_all["held_out_demos"]
    assert a.demo_id not in held_out, \
        f"demo {a.demo_id} is HELD OUT (sim-gate eval only) -- refusing to train on it"

    h5 = f"/root/rawdemos/task-0000/episode_{a.demo_id:08d}.hdf5"
    wrapper = HDF5PlaybackWrapper.create_from_hdf5(
        input_path=h5, output_path=f"/root/skill_tmp_{a.demo_id}.hdf5",
        robot_obs_modalities=("rgb", "proprio"),
        robot_proprio_keys=EVAL_PROPRIO_KEYS,
    )
    env = SkillCommitEnv(wrapper, entries, shaping=not a.no_shaping, seed=a.seed)

    meta = load_meta()
    obs_dim = meta["obs_layout"]["l2_geometry"][1]
    act_dim = 12
    scale = np.asarray(json.load(open("/root/skill_wrapper_scale.json"))["scale"], np.float32)
    agent = RLPD(obs_dim, act_dim, utd=a.utd, seed=a.seed)
    if a.resume:
        agent.load(a.resume)
        print(f"resumed from {a.resume}")
    online = ReplayBuffer(obs_dim, act_dim, capacity=200_000)
    if a.buffer_path and os.path.exists(a.buffer_path):
        z = np.load(a.buffer_path)
        n = int(z["n"])
        online.obs[:n], online.act[:n] = z["obs"][:n], z["act"][:n]
        online.rew[:n], online.nobs[:n] = z["rew"][:n], z["nobs"][:n]
        online.done[:n] = z["done"][:n]
        online.idx, online.full = n % online.capacity, n == online.capacity
        print(f"online buffer restored: {n} transitions", flush=True)
    prior = ReplayBuffer.from_prior_npz("/root/skill_buffer_prior/prior_buffer.npz",
                                        scale, hold_out_demos=held_out)
    print(f"prior buffer: {len(prior)} tuples (held out {held_out})")

    os.makedirs(a.out, exist_ok=True)
    results, env_steps, ep = [], 0, 0
    t0 = time.time()
    while env_steps < a.max_env_steps:
        try:
            obs = env.reset()
        except RuntimeError as e:
            print(f"RESET_EXHAUSTED: {e}")
            break
        done, ep_r = False, 0.0
        while not done:
            act = agent.act(obs)
            nobs, r, done, info = env.step(act)
            online.add(obs, act, r, nobs, float(info["success"]))
            obs = nobs
            ep_r += r
            env_steps += 1
            if len(online) > 1000:
                stats = agent.update(online, prior, a.batch)
        results.append(bool(info["success"]))
        ep += 1
        roll = float(np.mean(results[-20:]))
        print(f"EP {ep} steps={info['steps']} success={info['success']} "
              f"dist={info['dist']:.3f} ep_r={ep_r:.2f} rolling20={roll:.2f} "
              f"env_steps={env_steps} elapsed={time.time() - t0:.0f}s", flush=True)
        if ep % 25 == 0 or env_steps >= a.max_env_steps:
            agent.save(f"{a.out}/skill_d{a.demo_id}_s{a.stage}_ep{ep}.pt")
            json.dump({"demo": a.demo_id, "stage": a.stage, "episodes": ep,
                       "env_steps": env_steps, "results": results,
                       "rolling20": roll, "elapsed_s": time.time() - t0},
                      open(f"{a.out}/stats_d{a.demo_id}_s{a.stage}.json", "w"))
        if a.stop_at > 0 and roll >= a.stop_at and len(results) >= 20:
            print(f"SMOKE_BAR_MET rolling20={roll:.2f} at ep {ep} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            agent.save(f"{a.out}/skill_d{a.demo_id}_s{a.stage}_smoke.pt")
            break
    if a.buffer_path:
        n = len(online)
        np.savez_compressed(a.buffer_path, n=np.int64(n), obs=online.obs[:n],
                            act=online.act[:n], rew=online.rew[:n],
                            nobs=online.nobs[:n], done=online.done[:n])
        print(f"online buffer saved: {n} transitions", flush=True)
    final_roll = float(np.mean(results[-20:])) if results else 0.0
    if results and final_roll < 0.3:
        # quarantine: a failure-dominated chunk must not become the next chunk's resume
        # source (2026-08-13: 20 failure-only episodes degraded the actor enough to lose
        # the already-mastered press-5 rung)
        import glob as _g
        import shutil
        os.makedirs(f"{a.out}/lowperf", exist_ok=True)
        for p in _g.glob(f"{a.out}/skill_d{a.demo_id}_s{a.stage}_*.pt"):
            shutil.move(p, f"{a.out}/lowperf/{os.path.basename(p)}")
        print(f"CHUNK_QUARANTINED rolling20={final_roll:.2f} -> {a.out}/lowperf/", flush=True)
    print(f"TRAIN_CHUNK_DONE ep={ep} env_steps={env_steps} "
          f"rolling20={final_roll:.2f}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()

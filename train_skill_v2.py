"""V2 skill trainer (SKILL_TRAINER_V2_SPEC.md D1/D3/D6/D7/D9).

ONE process, ONE policy, ALL scenes: snapshots make every instance loadable into the same
booted sim, so multi-scene round-robin costs ~1 s per switch. Per-entry frontier sampling
(D6): entries sampled proportional to (1 - success_rate) + floor. Episode-denominated
budgets per scene per round (D3) with the demo-replay SEEDING trigger for stuck scenes.
Flight recorder (D7): one JSON line per episode. Stats dumped on EVERY exit path.

Smoke (spec d1-d2): --demos 20,80,110 --episodes-per-scene 25 --rounds 2
  -> bar: warm scenes reach rolling >= 0.5; seeding fires on d110.

Run inside behavior env:
  PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 OMNIGIBSON_HEADLESS=1 \
    python -u train_skill_v2.py --demos 20,80,110 --l2 on --out /root/v2_smoke
"""

import argparse
import collections
import json
import os
import time

import numpy as np

os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")

HELD_OUT = [10, 60, 120, 180, 230, 280, 330, 380]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="", help="comma list; empty = all non-held-out")
    ap.add_argument("--stages", default="0,1,2,3")
    ap.add_argument("--l2", choices=["on", "off"], default="on")
    ap.add_argument("--episodes-per-scene", type=int, default=25)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--seed-trigger", type=int, default=12,
                    help="0 successes after this many episodes in a scene -> demo seeding")
    ap.add_argument("--utd", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--out", default="/root/v2_run")
    ap.add_argument("--bank-dir", default="/root/snapshot_bank")
    ap.add_argument("--prior", choices=["on", "off"], default="on",
                    help="off: retire the v1 splice prior (stale-physics press data)")
    ap.add_argument("--harvest", action="store_true",
                    help="V2.1-b: include harvest boundary states in the rotation")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import omnigibson as og  # noqa: F401
    from omnigibson.macros import gm
    gm.HEADLESS = True
    gm.ENABLE_TRANSITION_RULES = False
    from omnigibson.envs.hdf5_data_wrapper import HDF5PlaybackWrapper
    from reverse_curriculum_collect import EVAL_PROPRIO_KEYS
    from rlpd_sac import RLPD, ReplayBuffer
    from skill_env_wrapper_v2 import (SkillCommitEnvV2, load_snapshot_bank,
                                      load_harvest_bank, FAMILIES, L2_GRID)

    stages = {int(s) for s in a.stages.split(",") if s.strip().lstrip("-").isdigit()}
    entries = [e for e in load_snapshot_bank(a.bank_dir, exclude_demos=HELD_OUT)
               if e["stage"] in stages or e["family"] != "press"]
    if a.harvest:
        entries += load_harvest_bank(exclude_demos=HELD_OUT)
    if a.demos:
        want = {int(d) for d in a.demos.split(",")}
        entries = [e for e in entries if e["demo"] in want]
    demos = sorted({e["demo"] for e in entries})
    assert entries and not any(e["demo"] in HELD_OUT for e in entries)
    print(f"V2 entries={len(entries)} scenes={demos}", flush=True)

    # boot the shared sim off ANY demo's hdf5 (scaffolding only; restores are snapshots)
    boot = f"/root/rawdemos/task-0000/episode_{demos[0]:08d}.hdf5"
    t0 = time.time()
    wrapper = HDF5PlaybackWrapper.create_from_hdf5(
        input_path=boot, output_path=f"{a.out}/v2_tmp.hdf5",
        robot_obs_modalities=("proprio",), robot_proprio_keys=EVAL_PROPRIO_KEYS)
    env = SkillCommitEnvV2(wrapper, entries, l2_on=(a.l2 == "on"), seed=a.seed)
    print(f"V2 booted {time.time() - t0:.0f}s obs_dim={env.obs_dim}", flush=True)

    agent = RLPD(env.obs_dim, 12, utd=a.utd, seed=a.seed)
    if a.resume:
        agent.load(a.resume, reset_alpha=True)
    online = ReplayBuffer(env.obs_dim, 12, capacity=400_000)
    # prior buffer: v1 splice tuples padded to the V2 layout (L2 zeros + press one-hot)
    prior = None
    if a.prior == "off":
        print("prior buffer RETIRED (--prior off): v1 splices carry stale-weld physics",
              flush=True)
    z = np.load("/root/skill_buffer_prior/prior_buffer.npz") if a.prior == "on" else None
    seedbuf = ReplayBuffer(env.obs_dim, 12, capacity=20_000)
    sc = json.load(open("/root/skill_wrapper_scale.json"))
    scale = np.asarray(sc["scale"], np.float32)
    if z is not None:
        keep = ~np.isin(z["demo"], np.asarray(HELD_OUT))
        n = int(keep.sum())
        v1_dim = z["obs"].shape[1]
        pad = env.obs_dim - v1_dim
        press = np.zeros((n, len(FAMILIES)), np.float32)
        press[:, FAMILIES.index("press")] = 1.0
        assert pad >= len(FAMILIES), f"V2 obs_dim {env.obs_dim} must exceed v1 {v1_dim}"

        def pad_obs(o):
            l2pad = np.zeros((o.shape[0], pad - len(FAMILIES)), np.float32)
            return np.concatenate([o, l2pad, press[:o.shape[0]]], axis=1)
        prior = ReplayBuffer(env.obs_dim, 12, capacity=n)
        prior.obs[:n] = pad_obs(z["obs"][keep].astype(np.float32))
        prior.act[:n] = np.clip(z["action"][keep] / scale[None, :], -1, 1)
        prior.rew[:n] = z["reward"][keep]
        prior.nobs[:n] = pad_obs(z["next_obs"][keep].astype(np.float32))
        prior.done[:n] = z["done"][keep].astype(np.float32)
        prior.idx, prior.full = 0, True
        print(f"prior buffer padded: {n} tuples v1_dim={v1_dim} -> {env.obs_dim}",
              flush=True)

    # per-entry frontier stats (D6) + per-scene round budgets (D3)
    ent_stats = {id(e): collections.deque(maxlen=20) for e in entries}
    by_scene = collections.defaultdict(list)
    for e in entries:
        by_scene[e["demo"]].append(e)
    recorder = open(f"{a.out}/flight_recorder.jsonl", "a")
    sel_rng = np.random.default_rng(a.seed)
    # v21g annealed AG assist: per-scene engage radius, 0.30 m (teleop-rig parity)
    # -> shrink 5 cm each time the scene's grasp rolling rate holds >= 0.5 -> 0 = eval physics
    assist_range = collections.defaultdict(lambda: 0.30)
    results_all = []
    env_steps = 0
    t0 = time.time()

    def dump_stats(tag=""):
        succ = sum(r["success"] for r in results_all)
        json.dump({"episodes": len(results_all), "successes": succ,
                   "env_steps": env_steps, "elapsed_s": round(time.time() - t0, 1),
                   "l2": a.l2, "scenes": demos,
                   "per_scene": {str(d): [sum(ent_stats[id(e)]) / max(len(ent_stats[id(e)]), 1)
                                          for e in by_scene[d]] for d in demos}},
                  open(f"{a.out}/stats.json", "w"), indent=1)
        agent.save(f"{a.out}/ckpt{tag}.pt")

    def run_episode(entry, seeded=False):
        nonlocal env_steps
        env.ag_assist_range = (assist_range[entry["demo"]]
                               if entry["family"] == "pick_up_from" else 0.0)
        obs = env.reset(entry=entry)
        done, ep_r, t_ep = False, 0.0, time.time()
        while not done:
            act = agent.act(obs)
            nobs, r, done, info = env.step(act)
            online.add(obs, act, r, nobs, float(info["success"]))
            obs = nobs
            ep_r += r
            env_steps += 1
            if len(online) > 1000:
                agent.update(online, prior, a.batch, seed_buf=seedbuf)
        rec = {"demo": entry["demo"], "stage": entry["stage"], "success": info["success"],
               "assist_r": round(env.ag_assist_range, 2),
               "af": info.get("assist_fired"), "ag": info.get("ag"),
               "lift": info.get("lift_prog"),
               "steps": info["steps"], "dist": round(info["dist"], 3),
               "ep_r": round(ep_r, 2), "seeded": seeded, "env_steps": env_steps,
               "wall_s": round(time.time() - t_ep, 1)}
        recorder.write(json.dumps(rec) + "\n")
        recorder.flush()
        results_all.append(rec)
        ent_stats[id(entry)].append(1 if info["success"] else 0)
        return rec

    bridge_count = collections.defaultdict(int)
    chain_seeded = set()

    def run_chain_episode(gentry, ptmpl):
        """Full chain: grasp -> goal-switch -> press, ONE episode. The grasp-success
        step is stored with done=0 so press value bootstraps BACK through the boundary
        (the chain is one task, not two stapled skills)."""
        nonlocal env_steps
        env.ag_assist_range = assist_range[gentry["demo"]]
        obs = env.reset(entry=gentry)
        done, ep_r, t_ep, phase = False, 0.0, time.time(), 1
        info = {}
        while True:
            act = agent.act(obs)
            nobs, r, done, info = env.step(act)
            online.add(obs, act, r, nobs, float(info["success"] and phase == 2))
            obs = nobs
            ep_r += r
            env_steps += 1
            if len(online) > 1000:
                agent.update(online, prior, a.batch, seed_buf=seedbuf)
            if done:
                if phase == 1 and info["success"]:
                    phase = 2
                    env.entry = dict(ptmpl)
                    env.active_arm = ptmpl["active_arm"]
                    env.steps = 0
                    env.dwell = 0
                    env.dwell_bonus = 0.0
                    env.contact_made = False
                    env._lift_dwell = 0
                    env.target_base = env._live_target_base()
                    from skill_env_wrapper import _np as __np
                    env.hold_anchor = __np(
                        env.target.get_position_orientation()[0]).copy()
                    env._drift_pot = 0.0
                    # refresh the stationary command: hold_act was computed at RESET
                    # (pre-grasp pose) — stale targets steer the holding arm back down,
                    # dragging the radio (the filmed 'drift')
                    env.hold_act = env._stationary_act(env._proprio61())
                    env.steps = -150   # phase-2 budget = 450 (transport + press)
                    done = False
                    obs = env._obs()
                    continue
                break
        rec = {"demo": gentry["demo"], "stage": "C",
               "success": bool(info["success"] and phase == 2), "phase": phase,
               "assist_r": round(env.ag_assist_range, 2),
               "steps": info["steps"], "dist": round(info["dist"], 3),
               "ep_r": round(ep_r, 2), "seeded": False, "env_steps": env_steps,
               "wall_s": round(time.time() - t_ep, 1)}
        recorder.write(json.dumps(rec) + "\n")
        recorder.flush()
        results_all.append(rec)
        print(f"CHAIN d{gentry['demo']} phase={phase} succ={rec['success']} "
              f"steps={rec['steps']}", flush=True)
        return rec

    try:
        for rnd in range(a.rounds):
            for d in demos:
                es = by_scene[d]
                scene_succ = 0
                fam_eps, fam_succ, fam_seeded = {}, {}, set()
                for i in range(a.episodes_per_scene):
                    # V2.1-e rung-ORDERED frontier sampling: press rungs unlock in offset
                    # order (k+1 only after rung k rolling >= 0.5 in this scene); non-press
                    # families (grasp 'G') and boundary 'B' states are their own tracks.
                    OFF = {0: 5, 1: 10, 2: 25, 4: 50, "B": 60, 3: 100}

                    def rate(e):
                        h = ent_stats[id(e)]
                        return sum(h) / max(len(h), 1)
                    press = sorted([e for e in es if e["family"] == "press"],
                                   key=lambda e: OFF.get(e["stage"], 999))
                    allowed = []
                    for e in press:
                        allowed.append(e)
                        if rate(e) < 0.5 or len(ent_stats[id(e)]) < 6:
                            break                      # gate: stop unlocking past frontier
                    # bridge states ('H') bypass the press rung-gate: they are the chain's
                    # own curriculum track, not a demo-offset rung
                    allowed += [e for e in es
                                if e["family"] != "press" or e.get("stage") == "H"]
                    wts = np.array([(1 - rate(e)) + 0.15 for e in allowed])
                    entry = allowed[int(sel_rng.choice(len(allowed), p=wts / wts.sum()))]
                    # guaranteed grasp cadence (v21o): the frontier sampler starves solved
                    # entries (~2%), but grasp FEEDS the chain (bridges, gate, anneal) —
                    # every 5th episode is grasp when the scene has one
                    _gent = next((x for x in es if x["family"] == "pick_up_from"), None)
                    if _gent is not None and i % 5 == 1:
                        entry = _gent
                    # guaranteed press cadence (v21v): press rungs starve the same way
                    # grasp did once bridges/chains dominate the weights
                    _pent = next((x for x in press if str(x.get("stage")) == "0"), None)
                    if _pent is not None and i % 5 == 3:
                        entry = _pent

                    # CHAIN scheduling: once this scene's grasp is consolidating, every
                    # 4th episode runs the full grasp->press chain; the first chain gets
                    # a demo chain-seed (grasp-through-press replayed as ONE trajectory)
                    gent = next((x for x in es if x["family"] == "pick_up_from"), None)
                    ptmpl = next((x for x in press if str(x.get("stage")) == "0"),
                                 press[0] if press else None)
                    chain_due = (gent is not None and ptmpl is not None
                                 and len(ent_stats[id(gent)]) >= 2
                                 and rate(gent) >= 0.5 and i % 4 == 3)
                    if chain_due:
                        if (rnd, d) not in chain_seeded:
                            chain_seeded.add((rnd, d))
                            trans, ok = env.seed_from_demo(
                                gent, chain_press_entry=ptmpl)
                            if ok:
                                for o_, a_, r_, no_, dn_ in trans:
                                    online.add(o_, a_, r_, no_, dn_)
                                    seedbuf.add(o_, a_, r_, no_, dn_)
                                print(f"CHAIN_SEEDED d{d}: {len(trans)} transitions",
                                      flush=True)
                                for _bp in getattr(env, "last_chain_rungs", []):
                                    _be = {"demo": d, "stage": "H", "frame": -1,
                                           "holding_arm": gent["active_arm"],
                                           "active_arm": ptmpl["active_arm"],
                                           "snapshot": _bp, "family": "press",
                                           "lift_z": gent.get("lift_z"),
                                           "grasp_closure_frame": None,
                                           "target_name_sub": "radio"}
                                    es.append(_be)
                                    ent_stats[id(_be)] = collections.deque(maxlen=20)
                                print(f"TRANSPORT_RUNGS d{d}: "
                                      f"{len(getattr(env, 'last_chain_rungs', []))}",
                                      flush=True)
                            else:
                                print(f"CHAIN_SEED_FAILED d{d}", flush=True)
                        rec = run_chain_episode(gent, ptmpl)
                        entry = gent
                    else:
                        rec = run_episode(entry)
                    scene_succ += rec["success"]
                    if (entry["family"] == "pick_up_from" and rec["success"]
                            and rec.get("stage") != "C"):
                        h = ent_stats[id(entry)]
                        if len(h) >= 6 and sum(h) / len(h) >= 0.5:
                            assist_range[entry["demo"]] = max(
                                0.0, assist_range[entry["demo"]] - 0.05)
                            print(f"ASSIST_ANNEAL d{entry['demo']} -> "
                                  f"{assist_range[entry['demo']]:.2f}", flush=True)
                    # BRIDGE harvest: every grasp success donates its end state (radio
                    # held, lifted) as a press start — the handoff distribution
                    if (entry["family"] == "pick_up_from" and rec["success"]
                            and rec.get("stage") != "C" and bridge_count[d] < 3):
                        import omnigibson as _og
                        st = _og.sim.dump_state(serialized=True)
                        st = st.cpu().numpy() if hasattr(st, "cpu") else np.asarray(st)
                        os.makedirs(f"{a.out}/bridge", exist_ok=True)
                        bp = f"{a.out}/bridge/d{d}_{bridge_count[d]}.npz"
                        np.savez_compressed(bp, state=st)
                        be = {"demo": d, "stage": "H", "frame": -1,
                              "holding_arm": entry["active_arm"], "active_arm": "left",
                              "snapshot": bp, "family": "press",
                              "lift_z": entry.get("lift_z"),
                              "grasp_closure_frame": None, "target_name_sub": "radio"}
                        es.append(be)
                        ent_stats[id(be)] = collections.deque(maxlen=20)
                        bridge_count[d] += 1
                        print(f"BRIDGE d{d} harvested #{bridge_count[d]}", flush=True)
                    fam = entry["family"]
                    fam_eps[fam] = fam_eps.get(fam, 0) + 1
                    fam_succ[fam] = fam_succ.get(fam, 0) + rec["success"]
                    roll = np.mean([r["success"] for r in results_all[-20:]])
                    print(f"R{rnd} d{d} EP{i + 1} succ={rec['success']} "
                          f"steps={rec['steps']} dist={rec['dist']} roll20={roll:.2f} "
                          f"env_steps={env_steps}", flush=True)
                    # per-FAMILY starvation trigger (scene-level masked grasp forever:
                    # press successes suppressed seeding on every warm scene)
                    if (fam_succ.get(fam, 0) == 0 and fam not in fam_seeded
                            and fam_eps.get(fam, 0) >= a.seed_trigger):
                        fam_seeded.add(fam)
                        print(f"SEED_TRIGGER d{d} fam={fam}", flush=True)
                        for e in [x for x in es if x["family"] == fam and str(x.get("stage")) != "H"][:2]:
                            trans, ok = env.seed_from_demo(e)
                            if ok:
                                for o_, a_, r_, no_, dn_ in trans:
                                    online.add(o_, a_, r_, no_, dn_)
                                    seedbuf.add(o_, a_, r_, no_, dn_)
                                print(f"SEEDED d{d} s{e['stage']}: "
                                      f"{len(trans)} transitions SUCCESS", flush=True)
                            else:
                                print(f"SEED_FAILED d{d} s{e['stage']}", flush=True)
                    # graduation: all entries in scene at rolling >= 0.95
                    rates = [sum(ent_stats[id(e)]) / max(len(ent_stats[id(e)]), 1)
                             for e in es]
                    if min(len(ent_stats[id(e)]) for e in es) >= 10 and min(rates) >= 0.95:
                        print(f"GRADUATED d{d} round {rnd}", flush=True)
                        break
                dump_stats()
            dump_stats(tag=f"_r{rnd}")
    finally:
        dump_stats(tag="_final")
        succ = sum(r["success"] for r in results_all)
        print(f"V2_DONE eps={len(results_all)} succ={succ} env_steps={env_steps} "
              f"elapsed={time.time() - t0:.0f}s", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()

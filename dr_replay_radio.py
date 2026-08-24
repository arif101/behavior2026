"""DR-replay sweep for radio press segments (task-62 pivot, radio edition).

For each press anchor (press-k frames, demos 20/30): playback-restore, apply graded
perturbation (radio pose jitter sigma), closed-loop replay of the demo's press actions,
score ToggledOn. Faithful perturbed trajectories are EXPORTED as strict-physics clips
(actions + snapshot-free: we save the perturbed start as a fresh snapshot + act stream)
— off-manifold press data for the flywheel. Collapse points map the press RL frontier.

Run (pauses training):  OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNIGIBSON_HEADLESS=1 \
    python -u dr_replay_radio.py --sigmas 0,0.005,0.01,0.02,0.03 --per 4
"""
import argparse
import json
import os

import numpy as np

os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmas", default="0,0.005,0.01,0.02,0.03")
    ap.add_argument("--per", type=int, default=4, help="replays per (anchor, sigma)")
    ap.add_argument("--out", default="/root/dr_press_clips")
    a = ap.parse_args()
    sigmas = [float(s) for s in a.sigmas.split(",")]

    import omnigibson as og
    from omnigibson.macros import gm
    gm.HEADLESS = True; gm.ENABLE_TRANSITION_RULES = False
    import h5py
    import torch as th
    from omnigibson.envs.hdf5_data_wrapper import HDF5PlaybackWrapper
    from omnigibson.object_states import ToggledOn
    from reverse_curriculum_collect import EVAL_PROPRIO_KEYS, restore_to_frame
    from skill_env_wrapper import _np, P, A_TORSO

    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(11)
    results = []
    sb = json.load(open("/root/skill_start_bank_v21.json"))["entries"]
    pf = {(e["demo"], str(e["stage"])): e.get("press_frame") for e in sb}
    for demo in (20, 30):
        man = json.load(open(f"/root/snapshot_bank_v21/manifest_d{demo}.json"))
        anchors = [dict(r, press_frame=pf.get((demo, str(r["stage"]))))
                   for r in man["entries"]
                   if r.get("family", "press") == "press"
                   and str(r["stage"]) in ("0", "1", "2")
                   and pf.get((demo, str(r["stage"])))]
        wrapper = HDF5PlaybackWrapper.create_from_hdf5(
            input_path=f"/root/rawdemos/task-0000/episode_{demo:08d}.hdf5",
            output_path=f"/root/drtmp_{demo}.hdf5",
            robot_obs_modalities=("proprio",), robot_proprio_keys=EVAL_PROPRIO_KEYS)
        radio = [o for o in wrapper.scene.objects if "radio" in o.name.lower()][0]
        rob = wrapper.scene.robots[0]

        def q61():
            obs = wrapper.env.get_obs()[0]
            def find(node, sub):
                if isinstance(node, dict):
                    for k, v in node.items():
                        r = find(v, sub)
                        if r is not None:
                            return r
                        if sub in str(k):
                            return v
                return None
            return _np(find(obs, "proprio")).reshape(-1)

        with h5py.File(f"/root/rawdemos/task-0000/episode_{demo:08d}.hdf5", "r") as f:
            key = sorted(k for k in f["data"].keys() if k.startswith("demo_"))[0]
            all_acts = f[f"data/{key}/action"][:]

        for anc in anchors:
            f0 = anc["frame"]
            budget = (anc["press_frame"] - f0) + 60
            acts = all_acts[f0:f0 + budget]
            for sig in sigmas:
                for rep in range(a.per if sig > 0 else 1):
                    restore_to_frame(wrapper, 0, f0)
                    if sig > 0:
                        rp, rq = radio.get_position_orientation()
                        jit = rng.normal(0, sig, 3); jit[2] = abs(jit[2]) * 0.3
                        radio.set_position_orientation(
                            position=th.as_tensor(_np(rp) + jit, dtype=th.float32),
                            orientation=rq)
                    try:
                        if radio.states[ToggledOn].get_value():
                            radio.states[ToggledOn].set_value(False)
                    except KeyError:
                        pass
                    ep_acts, fired = [], False
                    for cmd in acts:
                        cmd = np.asarray(cmd, np.float32)
                        for _ in range(8):
                            wrapper.env.step(cmd)
                            q = q61()
                            if (np.abs(q[P["left"]["arm_qpos"]] - cmd[7:14]).max() < 0.03
                                    and np.abs(q[P["right"]["arm_qpos"]]
                                               - cmd[15:22]).max() < 0.03
                                    and np.abs(q[P["trunk_qpos"]]
                                               - cmd[A_TORSO]).max() < 0.03):
                                break
                        ep_acts.append(cmd)
                        if radio.states[ToggledOn].get_value():
                            fired = True
                            break
                    results.append((demo, str(anc["stage"]), sig, int(fired)))
                    print(f"DR d{demo} s{anc['stage']} sig={sig:.3f} rep{rep}: "
                          f"{'FIRED' if fired else 'miss'} ({len(ep_acts)} acts)",
                          flush=True)
                    if fired and sig > 0:
                        st = og.sim.dump_state(serialized=True)
                        st = st.cpu().numpy() if hasattr(st, "cpu") else np.asarray(st)
                        n = len([x for x in os.listdir(a.out) if x.endswith(".npz")])
                        np.savez_compressed(
                            f"{a.out}/drclip_{n:04d}.npz",
                            actions=np.stack(ep_acts), end_state=st,
                            meta=json.dumps({"demo": demo, "stage": str(anc["stage"]),
                                             "sigma": sig, "frame": f0,
                                             "family": "press"}))
        og.clear()
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0])
    for demo, st, sig, ok in results:
        agg[sig][1] += 1; agg[sig][0] += ok
    for sig in sorted(agg):
        s, n = agg[sig]
        print(f"DR_SUMMARY sigma={sig:.3f}: {s}/{n} = {s/max(n,1):.2f}", flush=True)
    os._exit(0)


main()

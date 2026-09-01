"""Relay episode factory: manufacture COMPLETE task episodes = certified honest
grasp+transport (factory clip replay) + trained-policy finish (in-hand press),
capturing converter-ready obs for the WHOLE episode.

Per demo: replay the factory clip cmd stream (weld re-established at meta
weld_k) with obs captured at every distinct-command boundary (v2 convention:
3 cameras rgb+depth 1080x1080, 61-proprio, base_pose, radio pose), then hand
control to the policy (relay protocol) and keep capturing per executed action.
If the episode toggles with the weld intact, save rac_<demo>_r<try>.npz in the
converter's rac_ family format (success=True). Stochastic policy => --tries
attempts; save every success.

Run (one process per demo, sim free; worker spawned internally):
  OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNIGIBSON_HEADLESS=1 \
      python -u /root/relay_episode_factory.py --demo 20 --tries 2
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np

os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")

REQ = "/dev/shm/relay_req.tmp.npz"
REQF = "/dev/shm/relay_req.npz"
ACT = "/dev/shm/relay_act.npy"
OUT = "/root/factory_obs2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", type=int, required=True)
    ap.add_argument("--tries", type=int, default=2)
    ap.add_argument("--execute", type=int, default=16)
    ap.add_argument("--budget", type=int, default=400)
    a = ap.parse_args()
    for p in (REQF, ACT):
        if os.path.exists(p):
            os.remove(p)

    wlog = open(f"/root/relay_worker_ef{a.demo}.log", "w")
    worker = subprocess.Popen(
        ["/root/miniconda3/envs/openpi/bin/python", "-u", "/root/relay_worker.py"],
        stdout=wlog, stderr=subprocess.STDOUT,
        env={**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"})

    from PIL import Image
    import torch as th
    import omnigibson as og  # noqa: F401
    from omnigibson.macros import gm
    gm.HEADLESS = True; gm.ENABLE_TRANSITION_RULES = False
    from omnigibson.envs.hdf5_data_wrapper import HDF5PlaybackWrapper
    from reverse_curriculum_collect import EVAL_PROPRIO_KEYS, restore_to_frame
    from skill_env_wrapper import _np

    z = np.load(f"/root/factory_clips/d{a.demo:03d}_grasp_transport.npz",
                allow_pickle=True)
    cmds = z["cmds"]
    meta = json.loads(str(z["meta"]))

    wrapper = HDF5PlaybackWrapper.create_from_hdf5(
        input_path=f"/root/rawdemos/task-0000/episode_{a.demo:08d}.hdf5",
        output_path=f"/root/ref_tmp_{a.demo}.hdf5",
        robot_obs_modalities=("proprio", "rgb", "depth_linear"),
        robot_proprio_keys=EVAL_PROPRIO_KEYS)
    radio = [o for o in wrapper.scene.objects if "radio" in o.name.lower()][0]
    rob = wrapper.scene.robots[0]
    from omnigibson.object_states import ToggledOn
    toggled = lambda: bool(radio.states[ToggledOn].get_value())  # noqa: E731

    def get_obs():
        obs = wrapper.env.get_obs()[0]
        o = {"pro": None, "zed_rgb": None, "zed_dep": None, "l_rgb": None,
             "l_dep": None, "r_rgb": None, "r_dep": None}
        def walk(node, path=""):
            if isinstance(node, dict):
                for k, v in node.items():
                    kl = str(k).lower()
                    pl = path + "/" + kl
                    if "proprio" in kl:
                        o["pro"] = _np(v).reshape(-1)
                    elif isinstance(v, dict):
                        walk(v, path + "/" + str(k))
                    else:
                        c = ("zed" if "zed" in pl else
                             "l" if "left" in pl else
                             "r" if "right" in pl else None)
                        if c:
                            if kl.endswith("rgb") or kl == "rgb":
                                o[f"{c}_rgb"] = _np(v)
                            elif "depth" in kl:
                                o[f"{c}_dep"] = _np(v)
        walk(obs)
        return o

    def im224(arr):
        img = Image.fromarray(np.asarray(arr)[..., :3].astype(np.uint8))
        return np.asarray(img.resize((224, 224), Image.LANCZOS), np.uint8)

    def base_pose():
        bp, bq = rob.get_position_orientation()
        return np.concatenate([_np(bp), _np(bq)]).astype(np.float32)

    def radio_pose():
        rp, rq = radio.get_position_orientation()
        return np.concatenate([_np(rp), _np(rq)]).astype(np.float32)

    def wait_worker():
        deadline = time.time() + 600
        logp = f"/root/relay_worker_ef{a.demo}.log"
        while time.time() < deadline:
            if worker.poll() is not None:
                print("EF_ABORT worker died", flush=True)
                os._exit(1)
            if "RELAY_WORKER_READY" in open(logp).read():
                return
            time.sleep(2.0)
        print("EF_ABORT worker never ready", flush=True)
        worker.kill(); os._exit(1)

    n_saved = 0
    for trial in range(a.tries):
        R = {k: [] for k in ("p", "act", "hz", "hd", "lz", "ld", "rz", "rd",
                             "rp", "bp")}

        def capture(cmd):
            o = get_obs()
            if o["pro"] is None or o["zed_rgb"] is None:
                return False
            R["p"].append(o["pro"].astype(np.float32))
            R["act"].append(np.asarray(cmd, np.float32).copy())
            R["hz"].append(o["zed_rgb"][..., :3].astype(np.uint8))
            R["hd"].append(np.zeros((2, 2), np.float16) if o["zed_dep"] is None
                           else o["zed_dep"].astype(np.float16))
            R["lz"].append(np.zeros((2, 2, 3), np.uint8) if o["l_rgb"] is None
                           else o["l_rgb"][..., :3].astype(np.uint8))
            R["ld"].append(np.zeros((2, 2), np.float16) if o["l_dep"] is None
                           else o["l_dep"].astype(np.float16))
            R["rz"].append(np.zeros((2, 2, 3), np.uint8) if o["r_rgb"] is None
                           else o["r_rgb"][..., :3].astype(np.uint8))
            R["rd"].append(np.zeros((2, 2), np.float16) if o["r_dep"] is None
                           else o["r_dep"].astype(np.float16))
            R["rp"].append(radio_pose())
            R["bp"].append(base_pose())
            return True

        # phase 1: certified clip replay with obs at distinct-command boundaries
        restore_to_frame(wrapper, 0, meta["t0"])
        try:
            rob._refresh_rigid_contact_view()
        except Exception:  # noqa: BLE001
            pass
        welded = False
        prev_cmd = None
        for i, cmd in enumerate(cmds):
            cmd = np.asarray(cmd, np.float32)
            if prev_cmd is None or not np.array_equal(cmd, prev_cmd):
                capture(cmd)
                prev_cmd = cmd.copy()
            wrapper.env.step(cmd)
            if (not welded and i >= meta["weld_k"]
                    and rob._ag_obj_constraint_params.get("right") is None):
                ft = _np(th.stack([l.get_position_orientation()[0]
                                   for l in rob.finger_links["right"]]).mean(dim=0))
                rob._establish_grasp(radio, radio.root_link_name, "right",
                                     th.as_tensor(ft, dtype=th.float32),
                                     "FixedJoint")
                welded = True
            elif (not welded
                  and rob._ag_obj_constraint_params.get("right") is not None):
                welded = True
        n_clip = len(R["p"])
        ag0 = rob._ag_obj_constraint_params.get("right") is not None
        print(f"EF t{trial}: clip replayed, {n_clip} obs steps, ag={ag0} "
              f"tg={toggled()}", flush=True)
        if not ag0 or toggled():
            print(f"EF t{trial}: bad handoff, skip trial", flush=True)
            continue
        if trial == 0:
            wait_worker()

        # phase 2: policy finish, capture per executed action
        steps = 0
        while steps < a.budget and not toggled():
            o = get_obs()
            if o["pro"] is None or o["zed_rgb"] is None:
                break
            np.savez(REQ, im0=im224(o["zed_rgb"]),
                     im1=(im224(o["l_rgb"]) if o["l_rgb"] is not None
                          else np.zeros((224, 224, 3), np.uint8)),
                     im2=(im224(o["r_rgb"]) if o["r_rgb"] is not None
                          else np.zeros((224, 224, 3), np.uint8)),
                     state=o["pro"].astype(np.float32))
            os.replace(REQ, REQF)
            t1 = time.time()
            while not os.path.exists(ACT):
                if time.time() - t1 > 120:
                    print("EF_ABORT infer timeout", flush=True)
                    worker.kill(); os._exit(1)
                time.sleep(0.01)
            chunk = np.load(ACT); os.remove(ACT)
            for k in range(min(a.execute, len(chunk))):
                capture(chunk[k])
                wrapper.env.step(np.asarray(chunk[k], np.float32))
                steps += 1
                if toggled():
                    break
        ag1 = rob._ag_obj_constraint_params.get("right") is not None
        ok = toggled() and ag1
        print(f"EF t{trial}: policy steps={steps} toggled={toggled()} "
              f"ag={ag1} ok={ok} total_obs={len(R['p'])}", flush=True)
        if ok:
            np.savez_compressed(
                f"{OUT}/rac_{a.demo}_r{trial}.npz",
                proprio=np.stack(R["p"]), actions=np.stack(R["act"]),
                head_rgb=np.stack(R["hz"]), head_depth=np.stack(R["hd"]),
                left_rgb=np.stack(R["lz"]), left_depth=np.stack(R["ld"]),
                right_rgb=np.stack(R["rz"]), right_depth=np.stack(R["rd"]),
                objpose_radio_89=np.stack(R["rp"]), base_pose=np.stack(R["bp"]),
                radio_rest_z=np.float64(0.534), success=np.bool_(True),
                meta=json.dumps({**meta, "episode": "relay_full",
                                 "policy_steps": steps, "n_clip_obs": n_clip,
                                 "trial": trial}))
            n_saved += 1
            print(f"EF_SAVED rac_{a.demo}_r{trial}.npz "
                  f"({len(R['p'])} steps)", flush=True)
    print(f"EF_DONE d{a.demo}: saved {n_saved}/{a.tries}", flush=True)
    worker.kill()
    os._exit(0)


main()

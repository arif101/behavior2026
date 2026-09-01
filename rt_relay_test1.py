"""POLICY RELAY test 1: run the certified chain to the post-transport state, then
hand control to the trained pi0.5 checkpoint (radio_run2) and watch whether it can
finish the task. Isolates the data-poison thesis: the demos are causally sound
AFTER the grasp moment, so a policy spawned past the poison should behave; if it
also stalls here, the corpus needs longer clips.

Mechanics: replay the factory clip's cmd stream verbatim (weld re-established at
meta weld_k, exactly like render_factory_clips.py), then loop: build eval-style
obs (3 cameras resized to 224 + raw 61-proprio), file-RPC to relay_worker.py in
the openpi env (spawned by this script), execute the first --execute actions of
each (32, 23) chunk, re-infer. Stop on toggle or --budget control steps.

Run (sim free):
  OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNIGIBSON_HEADLESS=1 \
      python -u /root/rt_relay_test1.py --demo 30
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np

os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")

REQ = "/dev/shm/relay_req.npz"
REQ_TMP = "/dev/shm/relay_req.tmp.npz"
ACT = "/dev/shm/relay_act.npy"
OUT = "/root/relay_film"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", type=int, default=30)
    ap.add_argument("--execute", type=int, default=16)
    ap.add_argument("--budget", type=int, default=400)
    a = ap.parse_args()
    out = f"{OUT}/d{a.demo:03d}"
    os.makedirs(out, exist_ok=True)
    for p in (REQ, ACT):
        if os.path.exists(p):
            os.remove(p)

    # spawn the policy worker now: it loads params (~minutes) while the sim boots
    wlog = open(f"/root/relay_worker_d{a.demo}.log", "w")
    worker = subprocess.Popen(
        ["/root/miniconda3/envs/openpi/bin/python", "-u", "/root/relay_worker.py"],
        stdout=wlog, stderr=subprocess.STDOUT,
        env={**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"})

    from PIL import Image
    import imageio
    import torch as th
    import omnigibson as og
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
        output_path=f"/root/relay_tmp_{a.demo}.hdf5",
        robot_obs_modalities=("proprio", "rgb"),
        robot_proprio_keys=EVAL_PROPRIO_KEYS)
    radio = [o for o in wrapper.scene.objects if "radio" in o.name.lower()][0]
    rob = wrapper.scene.robots[0]
    cam = og.sim.viewer_camera
    from omnigibson.object_states import ToggledOn
    toggled = lambda: bool(radio.states[ToggledOn].get_value())  # noqa: E731

    def get_obs():
        obs = wrapper.env.get_obs()[0]
        out_ = {"pro": None, "zed": None, "l": None, "r": None}
        def walk(node, path=""):
            if isinstance(node, dict):
                for k, v in node.items():
                    kl = str(k).lower()
                    pl = path + "/" + kl
                    if "proprio" in kl:
                        out_["pro"] = _np(v).reshape(-1)
                    elif isinstance(v, dict):
                        walk(v, path + "/" + str(k))
                    elif kl.endswith("rgb") or kl == "rgb":
                        c = ("zed" if "zed" in pl else
                             "l" if "left" in pl else
                             "r" if "right" in pl else None)
                        if c:
                            out_[c] = _np(v)
        walk(obs)
        return out_

    def im224(arr):
        img = Image.fromarray(np.asarray(arr)[..., :3].astype(np.uint8))
        return np.asarray(img.resize((224, 224), Image.LANCZOS), np.uint8)

    def grab():
        rp = _np(radio.get_position_orientation()[0])
        ft = _np(th.stack([l.get_position_orientation()[0]
                           for l in rob.finger_links["right"]]).mean(dim=0))
        target = 0.5 * (rp + ft)
        pos = target + np.array([0.55, -0.5, 0.35])
        f = target - pos; f /= np.linalg.norm(f)
        up = np.array([0.0, 0.0, 1.0])
        r = np.cross(f, up); r /= np.linalg.norm(r)
        u = np.cross(r, f)
        m = np.stack([r, u, -f], axis=1)
        w = np.sqrt(max(1e-9, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
        q = np.array([(m[2, 1] - m[1, 2]) / (4 * w),
                      (m[0, 2] - m[2, 0]) / (4 * w),
                      (m[1, 0] - m[0, 1]) / (4 * w), w])
        cam.set_position_orientation(th.as_tensor(pos, dtype=th.float32),
                                     th.as_tensor(q, dtype=th.float32))
        for _ in range(2):
            og.sim.render()
        o, _ = cam.get_obs()
        return np.asarray(o["rgb"])[..., :3].astype(np.uint8)

    # ---- phase 1: replay the certified clip to post-transport ------------------
    restore_to_frame(wrapper, 0, meta["t0"])
    try:
        rob._refresh_rigid_contact_view()
    except Exception:  # noqa: BLE001
        pass
    welded = False
    for i, cmd in enumerate(cmds):
        wrapper.env.step(np.asarray(cmd, np.float32))
        if (not welded and i >= meta["weld_k"]
                and rob._ag_obj_constraint_params.get("right") is None):
            ft = _np(th.stack([l.get_position_orientation()[0]
                               for l in rob.finger_links["right"]]).mean(dim=0))
            rob._establish_grasp(radio, radio.root_link_name, "right",
                                 th.as_tensor(ft, dtype=th.float32), "FixedJoint")
            welded = True
        elif not welded and rob._ag_obj_constraint_params.get("right") is not None:
            welded = True
    rp0 = _np(radio.get_position_orientation()[0])
    ag0 = rob._ag_obj_constraint_params.get("right") is not None
    print(f"RELAY handoff: radio z={rp0[2]:.3f} ag={ag0} tg={toggled()}",
          flush=True)

    # ---- phase 2: policy takes over -------------------------------------------
    deadline = time.time() + 600
    while time.time() < deadline:
        if worker.poll() is not None:
            print("RELAY_ABORT worker died before ready", flush=True)
            os._exit(1)
        if "RELAY_WORKER_READY" in open(f"/root/relay_worker_d{a.demo}.log").read():
            break
        time.sleep(2.0)
    else:
        print("RELAY_ABORT worker never became ready", flush=True)
        worker.kill(); os._exit(1)
    print("RELAY worker ready; policy in control", flush=True)

    frames = [grab()]
    steps, chunks = 0, 0
    act_norms = []
    while steps < a.budget and not toggled():
        o = get_obs()
        if o["pro"] is None or o["zed"] is None:
            print("RELAY_ABORT obs walk failed", flush=True)
            break
        np.savez(REQ_TMP, im0=im224(o["zed"]),
                 im1=(im224(o["l"]) if o["l"] is not None
                      else np.zeros((224, 224, 3), np.uint8)),
                 im2=(im224(o["r"]) if o["r"] is not None
                      else np.zeros((224, 224, 3), np.uint8)),
                 state=o["pro"].astype(np.float32))
        os.replace(REQ_TMP + ".npz" if os.path.exists(REQ_TMP + ".npz")
                   else REQ_TMP, REQ)
        t1 = time.time()
        while not os.path.exists(ACT):
            if time.time() - t1 > 120:
                print("RELAY_ABORT infer timeout", flush=True)
                worker.kill(); os._exit(1)
            time.sleep(0.01)
        chunk = np.load(ACT); os.remove(ACT)
        act_norms.append(float(np.linalg.norm(np.diff(chunk[:a.execute], axis=0))))
        for k in range(min(a.execute, len(chunk))):
            wrapper.env.step(np.asarray(chunk[k], np.float32))
            steps += 1
            if toggled():
                break
        chunks += 1
        rp = _np(radio.get_position_orientation()[0])
        ag = rob._ag_obj_constraint_params.get("right") is not None
        print(f"RELAY chunk={chunks} steps={steps} radio_z={rp[2]:.3f} ag={ag} "
              f"tg={toggled()} dnorm={act_norms[-1]:.3f}", flush=True)
        img = grab(); frames.append(img)
        imageio.imwrite(f"{out}/ch{chunks:03d}.png", img)

    rp1 = _np(radio.get_position_orientation()[0])
    s = dict(demo=a.demo, toggled=toggled(), steps=steps, chunks=chunks,
             radio_z_handoff=float(rp0[2]), radio_z_end=float(rp1[2]),
             ag_end=bool(rob._ag_obj_constraint_params.get("right") is not None),
             mean_dnorm=(float(np.mean(act_norms)) if act_norms else None))
    print("RELAY_SUMMARY", json.dumps(s), flush=True)
    json.dump(s, open(f"{out}/relay_d{a.demo}.json", "w"), indent=1)
    imageio.mimsave(f"{out}/relay_d{a.demo}.mp4", frames, fps=4)
    worker.kill()
    print("RELAY_DONE", flush=True)
    os._exit(0)


main()

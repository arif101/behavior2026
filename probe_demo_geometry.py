"""Measure, per demo, the facts the approach factory depends on (no servo, ~5 min):
radio model id; rest attitude (body up-vector in world); the rig's pull (position +
rotation) between closure-K and closure+t0off; the certified hand/finger geometry in
the radio frame at post-pull. Writes /root/factory_clips_approach/d<D>_probe.json.
"""
import argparse, json, os
import numpy as np
os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--demos", type=int, nargs="+", required=True); ap.add_argument("--K", type=int, default=30)
    a = ap.parse_args()
    import omnigibson as og  # noqa: F401
    from omnigibson.macros import gm
    gm.HEADLESS = True; gm.ENABLE_TRANSITION_RULES = False
    from omnigibson.envs.hdf5_data_wrapper import HDF5PlaybackWrapper
    from reverse_curriculum_collect import EVAL_PROPRIO_KEYS, restore_to_frame
    from skill_env_wrapper import _np
    from scipy.spatial.transform import Rotation as R
    d = a.demos[0]  # one scene per process (og.clear forbidden); driver loops demos
    man = json.load(open(f"/root/snapshot_bank_v21/manifest_d{d}.json"))
    closure = next(r for r in man["entries"] if str(r["stage"]) == "G")["grasp_closure_frame"]
    t0off = 6
    try:
        fm = json.load(open(f"/root/factory_clips/d{d:03d}_meta.json")); t0off = int(fm["t0"] - fm["closure"])
    except FileNotFoundError:
        pass
    w = HDF5PlaybackWrapper.create_from_hdf5(
        input_path=f"/root/rawdemos/task-0000/episode_{d:08d}.hdf5", output_path=f"/root/pdg_tmp_{d}.hdf5",
        robot_obs_modalities=("proprio",), robot_proprio_keys=EVAL_PROPRIO_KEYS)
    radio = [o for o in w.scene.objects if "radio" in o.name.lower()][0]; rob = w.scene.robots[0]
    model = dict(name=radio.name, category=getattr(radio, "category", "?"), model=getattr(radio, "model", "?"),
                 prim=str(getattr(radio, "prim_path", "?")))
    def pose(o):
        p, q = o.get_position_orientation(); return _np(p).copy(), R.from_quat(_np(q))
    restore_to_frame(w, 0, closure + t0off)
    pE, RE = pose(rob.eef_links["right"]); pR1, RR1 = pose(radio)
    fingers = [_np(l.get_position_orientation()[0]) for l in rob.finger_links["right"]]
    rel_f = [RR1.inv().apply(f - pR1).tolist() for f in fingers]
    post = dict(radio_pos=pR1.tolist(), radio_up_world=RR1.apply([0, 0, 1]).tolist(),
                rel_p=RR1.inv().apply(pE - pR1).tolist(), rel_R_quat=(RR1.inv() * RE).as_quat().tolist(), rel_f=rel_f,
                hand_z_axis_world=RE.apply([0, 0, 1]).tolist())
    restore_to_frame(w, 0, closure - a.K)
    pR0, RR0 = pose(radio); pE0, RE0 = pose(rob.eef_links["right"])
    rest = dict(radio_pos=pR0.tolist(), radio_up_world=RR0.apply([0, 0, 1]).tolist(),
                hand_to_radio=float(np.linalg.norm(pE0 - pR0)), hand_z_axis_world=RE0.apply([0, 0, 1]).tolist())
    pull = dict(pos=(pR1 - pR0).tolist(), norm=float(np.linalg.norm(pR1 - pR0)),
                rot_deg=float((RR1 * RR0.inv()).magnitude() * 180 / np.pi),
                hand_rot_deg=float((RE * RE0.inv()).magnitude() * 180 / np.pi))
    out = dict(demo=d, closure=closure, t0off=t0off, K=a.K, model=model, post=post, rest=rest, pull=pull)
    json.dump(out, open(f"/root/factory_clips_approach/d{d:03d}_probe.json", "w"), indent=1)
    print("PROBE", json.dumps(dict(demo=d, model=model["model"], name=model["name"], pull_m=round(pull["norm"], 3),
          pull_rot_deg=round(pull["rot_deg"], 1), hand_rot_deg=round(pull["hand_rot_deg"], 1),
          rest_up=[round(x, 2) for x in rest["radio_up_world"]], post_up=[round(x, 2) for x in post["radio_up_world"]],
          rel_fm=[round(x, 3) for x in np.mean(np.array(rel_f), axis=0)])), flush=True)
    os._exit(0)

main()

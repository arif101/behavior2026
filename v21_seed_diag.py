"""Grasp-rung seed diagnostic: replay demo actions from playback restore at the
grasp anchor (d30 sG) and print per-step what the success predicate would see:
object z vs lift threshold, AG dict truthiness, dwell. No AG re-pin (round-2 rule)."""
import json, os
import numpy as np
os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")

def main():
    import omnigibson as og
    from omnigibson.macros import gm
    gm.HEADLESS = True; gm.ENABLE_TRANSITION_RULES = False
    import h5py
    from omnigibson.envs.hdf5_data_wrapper import HDF5PlaybackWrapper
    from reverse_curriculum_collect import EVAL_PROPRIO_KEYS, restore_to_frame
    from skill_env_wrapper import _np, P
    wrapper = HDF5PlaybackWrapper.create_from_hdf5(
        input_path="/root/rawdemos/task-0000/episode_00000030.hdf5",
        output_path="/root/seeddiag_tmp.hdf5",
        robot_obs_modalities=("proprio",), robot_proprio_keys=EVAL_PROPRIO_KEYS)
    radio = [o for o in wrapper.scene.objects if "radio" in o.name.lower()][0]
    rob = wrapper.scene.robots[0]
    try:
        order = list(rob.controller_order)
    except Exception:
        order = list(rob.controllers.keys())
    dims = {}
    for cname in order:
        c = rob.controllers[cname]
        c = c[0] if isinstance(c, tuple) else c
        try:
            dims[cname] = int(c.command_dim)
        except Exception:
            dims[cname] = -1
    print(f"SD controller_order={order} dims={dims}", flush=True)
    man = json.load(open("/root/snapshot_bank_v21/manifest_d30.json"))
    g = [r for r in man["entries"] if str(r["stage"]) == "G"][0]
    base, arm = g["lift_z"], g["active_arm"]
    f0 = max(0, g["grasp_closure_frame"] - 20)   # pre-closure start (arm-fix diag)
    margin = 0.05
    print(f"SD entry start={f0} (anchor={g['frame']}, closure={g['grasp_closure_frame']}) "
          f"lift_z={base:.4f} active_arm={arm} arm_names={list(rob.arm_names)}", flush=True)
    restore_to_frame(wrapper, 0, f0)
    z0 = float(_np(radio.get_position_orientation()[0])[2])
    print(f"SD post-restore radio_z={z0:.4f} (base+margin={base+margin:.4f})", flush=True)
    def geom():
        rp = _np(radio.get_position_orientation()[0])
        out = {}
        for a_ in rob.arm_names:
            ep = _np(rob.eef_links[a_].get_position_orientation()[0])
            out[a_] = float(np.linalg.norm(ep - rp))
        return rp, out
    rp, dists = geom()
    print(f"SD geom post-restore radio={np.round(rp,3)} eef_dist={ {k: round(v,3) for k,v in dists.items()} }", flush=True)
    with h5py.File(wrapper.input_hdf5.filename, "r") as f:
        key = sorted(k for k in f["data"].keys() if k.startswith("demo_"))[0]
        acts = f[f"data/{key}/action"][f0:f0+240]
    ag_ever, lift_ever, dwell, succ_step = False, False, 0, None
    max_z = z0
    def q61():
        o = wrapper.env.get_obs()[0]
        def find(node, sub):
            if isinstance(node, dict):
                for k2, v2 in node.items():
                    r2 = find(v2, sub)
                    if r2 is not None: return r2
                    if sub in str(k2): return v2
            return None
        return _np(find(o, "proprio")).reshape(-1)
    def converged(cmd, q, tol=0.03):
        return (np.abs(q[3:10] - cmd[7:14]).max() < tol and
                np.abs(q[28:35] - cmd[15:22]).max() < tol and
                np.abs(q[53:57] - cmd[3:7]).max() < tol)
    total_substeps = 0
    import torch as th
    GRIP_CH = {"left": 14, "right": 22}[arm]
    root_link_name = next(k for k, v in radio.links.items() if v is radio.root_link)
    established = False
    for i, a in enumerate(np.asarray(acts)):
        cmd = np.asarray(a, np.float32)
        for sub in range(8):   # closed-loop: hold target until tracked (soft-gain env)
            wrapper.env.step(cmd)
            total_substeps += 1
            if converged(cmd, q61()):
                break
        # magnetic AG (collection-rig behavior): on grasp command, establish the
        # constraint anchored at the palm — PhysX pulls the object in, as the demos show
        if (not established and cmd[GRIP_CH] < 0
                and not rob._ag_obj_constraint_params.get(arm)):
            palm = _np(rob.eef_links[arm].get_position_orientation()[0])
            rpn = _np(radio.get_position_orientation()[0])
            if np.linalg.norm(palm - rpn) < 0.45:
                jt = rob._get_assisted_grasp_joint_type(radio, root_link_name)
                rob._establish_grasp(radio, root_link_name, arm,
                                     th.as_tensor(palm, dtype=th.float32), jt)
                established = True
                print(f"SD MAGNETIC-AG established at step {i} (palm-obj "
                      f"{float(np.linalg.norm(palm-rpn)):.3f} m)", flush=True)
        z = float(_np(radio.get_position_orientation()[2 - 2])[2]) if False else \
            float(_np(radio.get_position_orientation()[0])[2])
        max_z = max(max_z, z)
        ag = rob._ag_obj_constraint_params.get(arm)
        inhand = {k: bool(v) for k, v in rob._ag_obj_in_hand.items()}
        ag_ever = ag_ever or bool(ag)
        lifted = z > base + margin
        lift_ever = lift_ever or lifted
        dwell = dwell + 1 if (ag and lifted) else 0
        if dwell >= 15 and succ_step is None:
            succ_step = i
            print(f"SD SUCCESS at step {i}", flush=True)
            break
        if i % 20 == 0:
            rp2, d2 = geom()
            obs61 = None
            try:
                o = wrapper.env.get_obs()[0]
                def find(node, sub):
                    if isinstance(node, dict):
                        for k2, v2 in node.items():
                            r2 = find(v2, sub)
                            if r2 is not None: return r2
                            if sub in str(k2): return v2
                    return None
                obs61 = _np(find(o, "proprio")).reshape(-1)
            except Exception:
                pass
            cmd = np.asarray(a)
            tL = float(np.abs(obs61[P["left"]["arm_qpos"]] - cmd[7:14]).max()) if obs61 is not None else -1
            tR = float(np.abs(obs61[P["right"]["arm_qpos"]] - cmd[15:22]).max()) if obs61 is not None else -1
            print(f"SD step{i:03d} z={z:.4f} ag={bool(ag)} dwell={dwell} "
                  f"eefd={ {k: round(v,3) for k,v in d2.items()} } "
                  f"trackL={tL:.3f} trackR={tR:.3f} gripcmdR={cmd[22]:.2f}", flush=True)
    print(f"SD_VERDICT success_step={succ_step} ag_ever={ag_ever} "
          f"lift_ever={lift_ever} max_z={max_z:.4f} need={base+margin:.4f} "
          f"total_substeps={total_substeps}", flush=True)
    os._exit(0)

main()

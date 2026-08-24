"""V2 commit-window env (SKILL_TRAINER_V2_SPEC.md D2/D4/D9).

Deltas vs v1 (skill_env_wrapper.py):
- Resets load validated SNAPSHOTS (snapshot_bank/*.npz, serialized dump_state) instead of
  HDF5 playback + settle (day-0 profiler: 0.6-1.1 s vs 34-71 s, dz 0.33 mm, AG 5/5).
  All radio instances overlay ONE house scene, so a single booted env serves EVERY scene:
  multi-scene round-robin = load a different snapshot.
- L2 slot FILLED (D2): synthesized EE-frame occupancy of the target's oriented bbox,
  4x4x4 = 64 dims over a 0.4 m cube ahead of the active gripper (2 cm cells pooled 5x).
  Synthesized from privileged sim geometry in the SAME format the real mapper emits;
  fidelity check vs the real mapper (cosine >= 0.9 on >= 100 frames) is a separate
  pre-registered validation (v2_l2_fidelity.py, day 3) - until it passes, L2-on results
  carry a [synthesized-L2] caveat. --l2 off gives the blind control arm (B3).
- Goal/family code (D9): 4-dim one-hot [press, pick_up_from, place_in, place_on] appended
  to obs. Radio = press. NEVER a task ID.
- Target object from entry metadata (no "radio" hardcode); success predicate per family.
- hold_act synthesized from restored joint state (stationary command), not demo actions.
"""

import glob
import json
import math

import numpy as np

from skill_env_wrapper import (  # v1 constants/helpers reused verbatim
    A_GRIP, A_TORSO, A_ARM, BUDGET, P, PER_FRAME, STACK, q2r, _np)

FAMILIES = ["press", "pick_up_from", "place_in", "place_on"]
L2_GRID = 4          # 4x4x4 = 64 dims
L2_CUBE = 0.4        # metres, cube ahead of the gripper


def load_snapshot_bank(bank_dir="/root/snapshot_bank", exclude_demos=()):
    """Manifest rows with OK snapshots, flattened across demos."""
    entries = []
    for mf in sorted(glob.glob(f"{bank_dir}/manifest_d*.json")):
        m = json.load(open(mf))
        if m["demo"] in exclude_demos:
            continue
        for r in m["entries"]:
            if r["status"] == "OK":
                entries.append({"demo": m["demo"], "stage": r["stage"],
                                "frame": r["frame"], "holding_arm": r["holding_arm"],
                                "active_arm": r["active_arm"], "snapshot": r["snapshot"],
                                "family": r.get("family", "press"),
                                "lift_z": r.get("lift_z"),
                                "grasp_closure_frame": r.get("grasp_closure_frame"),
                                "target_name_sub": "radio"})
    return entries


def load_harvest_bank(path="/root/harvest_bank.json", exclude_demos=()):
    """V2.1-b: VLA-visited boundary states (press-in-place band) into the rotation."""
    try:
        h = json.load(open(path))
    except FileNotFoundError:
        return []
    if isinstance(h, dict):
        h = h.get("entries", [])
    return [{"demo": e["demo"], "stage": "B", "frame": e.get("step", -1),
             "holding_arm": e.get("holding_arm"), "active_arm": e.get("active_arm", "left"),
             "snapshot": e["state_file"], "family": "press", "lift_z": None,
             "target_name_sub": "radio"}
            for e in h if e["demo"] not in exclude_demos]


class SkillCommitEnvV2:
    def __init__(self, wrapper, entries, l2_on=True, shaping=True, seed=0, randomize=True,
                 scale_path="/root/skill_wrapper_scale.json"):
        self.w = wrapper
        self.env = wrapper.env
        self.robot = wrapper.scene.robots[0]
        self.entries = entries
        self.l2_on = l2_on
        self.ag_assist_range = 0.0   # annealed AG assist radius (trainer-set per scene)
        self._assist_used = False
        self.shaping = shaping
        self.randomize = randomize
        self.rng = np.random.default_rng(seed)
        sc = json.load(open(scale_path))
        self.scale = np.asarray(sc["scale"], np.float32)
        self.grip_center, self.grip_half = sc["grip_center"], sc["grip_half"]
        self._p_off = np.asarray(json.load(
            open("/root/behavior2026/labels/metalink_offset.json"))["offset_pos_root_frame"])
        self._snap_cache = {}
        self.obs_dim = STACK * PER_FRAME + 3 + 1 + L2_GRID ** 3 + len(FAMILIES)

    # ---- reset from snapshot ------------------------------------------------------------
    def reset(self, entry=None):
        import omnigibson as og
        from omnigibson.object_states import ToggledOn
        e = entry or self.entries[int(self.rng.integers(len(self.entries)))]
        self.entry = e
        self._assist_used = False
        self._ag_had = False
        self._lift_prog = 0.0
        self._release_ctr = 0
        snap = self._snap_cache.get(e["snapshot"])
        if snap is None:
            import torch as th
            z = np.load(e["snapshot"])
            snap = th.as_tensor(z["state"])   # OG deserialize expects torch (.clone())
            if len(self._snap_cache) < 200:
                self._snap_cache[e["snapshot"]] = snap
        og.sim.load_state(snap, serialized=True)
        self.target = [o for o in self.w.scene.objects
                       if e["target_name_sub"] in o.name.lower()][0]
        if e.get("holding_arm"):
            # HOLD REPAIR (v21l): v21-bank snapshots were dumped with the hold welded to
            # the WRONG (left) arm (pre-audit metadata) and snapshots serialize AG
            # natively — so first ensure the audited holder holds (coincident establish,
            # guarded), then release any stray weld on the other arm.
            self._reestablish_ag(e["holding_arm"])
            for _a in self.robot.arm_names:
                if _a != e["holding_arm"] and \
                        self.robot._ag_obj_constraint_params.get(_a):
                    try:
                        self.robot._release_grasp(arm=_a)
                    except Exception:  # noqa: BLE001
                        pass
        # success predicate must start false (snapshots were validated cleared; re-assert)
        try:
            if self.target.states[ToggledOn].get_value():
                self.target.states[ToggledOn].set_value(False)
        except KeyError:
            pass
        self.active_arm = e["active_arm"]
        obs61 = self._proprio61()
        self.hold_act = self._stationary_act(obs61)
        # settle 5 steps under stationary command (profiler: snapshot needs no long settle)
        for _ in range(5):
            self.env.step(self.hold_act)
        obs61 = self._proprio61()
        # LIVE target (2026-08-17 fix): for held objects the button moves with the hand,
        # so the target vector is recomputed from the object's CURRENT pose every step;
        # DR noise becomes a per-episode bias, not a frozen world point.
        self._point_bias = np.zeros(3)
        self.aff_conf = 0.75
        if self.randomize:
            self._point_bias = self.rng.normal(0, 0.026, 3)
            self.aff_conf = float(np.clip(self.rng.normal(0.75, 0.05), 0.5, 1.0))
        self.target_base = self._live_target_base()
        self._hist = [self._frame_feat(obs61)] * STACK
        self.steps = 0
        self.dwell = 0
        self.dwell_bonus = 0.0
        self.contact_made = False
        self._lift_dwell = 0
        self._ep_acts = []   # per-episode 23-d command log (flywheel export replay)
        # anti-drift anchor (v21q): where the held object should STAY during pressing
        self.hold_anchor = (_np(self.target.get_position_orientation()[0]).copy()
                            if e.get("holding_arm") else None)
        return self._obs()

    def _stationary_act(self, s):
        act = np.zeros(23, np.float32)
        for arm in ("left", "right"):
            act[A_ARM[arm]] = s[P[arm]["arm_qpos"]]
            act[A_GRIP[arm]] = float(np.mean(s[P[arm]["grip_qpos"]]))
        act[A_TORSO] = s[P["trunk_qpos"]]
        return act

    def _reestablish_ag(self, arm):
        r = self.robot
        if getattr(r, "_ag_obj_constraint_params", {}).get(arm):
            return True   # already held (snapshots restore AG natively) — never double-pin
        try:
            in_hand = r._calculate_in_hand_object(arm=arm)
        except Exception:  # noqa: BLE001
            in_hand = None
        if in_hand is not None:
            obj, link = in_hand
        else:
            obj, link = self.target, list(self.target.links.keys())[0]
        jt = r._get_assisted_grasp_joint_type(obj, link)
        if jt is None:
            return False
        try:
            cp = r._find_finger_contact_position(arm, obj.links[link].prim_path)
        except Exception:  # noqa: BLE001
            cp = None
        if cp is None:
            import torch as th
            cp = th.stack([l.get_position_orientation()[0]
                           for l in r.finger_links[arm]]).mean(dim=0)
        r._establish_grasp(obj, link, arm, cp, jt)
        return getattr(r, "_ag_obj_constraint_params", {}).get(arm) is not None

    # ---- step (v1 semantics, family-generic success) -----------------------------------
    def step(self, a):
        a = np.clip(np.asarray(a, np.float32), -1, 1)
        s = self._proprio61()
        arm = self.active_arm
        act23 = self.hold_act.copy()
        act23[0:3] = 0.0
        act23[A_ARM[arm]] = s[P[arm]["arm_qpos"]] + a[:7] * self.scale[:7]
        act23[A_TORSO] = s[P["trunk_qpos"]] + a[7:11] * self.scale[7:11]
        act23[A_GRIP[arm]] = self.grip_center + a[11] * self.grip_half
        # Grip latch (v21i): while the active arm holds via AG, keep the command closed
        # unless the policy DELIBERATELY releases (a[11] > 0.8 for 5 consecutive steps).
        # Without it a stochastic grip channel jitters open within a few steps and OG
        # releases the weld — v21h telemetry: assist fired every episode, ag=False at
        # every episode end, lift reached 4.4 cm then dropped.
        if (self.entry.get("family") == "pick_up_from"
                and getattr(self.robot, "_ag_obj_constraint_params",
                            {}).get(self.active_arm)):
            if a[11] > 0.8:
                self._release_ctr += 1
            else:
                self._release_ctr = 0
            if self._release_ctr < 5:
                act23[A_GRIP[arm]] = -1.0
        self._ep_acts.append(act23.copy())
        self.env.step(act23)
        # Annealed AG assist (v21g): grant the teleop rig's ranged engage when the policy
        # COMMITS (closes the gripper) within ag_assist_range of the target, then shrink
        # the range per-scene toward 0 = native eval physics (contact ∧ raycast ∧ 0.3 s).
        # Aligns rollout physics with the magnetic seeds so the critic sees ONE world;
        # flywheel export gates on range==0 episodes only.
        if (self.ag_assist_range > 0.0 and not self._assist_used
                and self.entry.get("family") == "pick_up_from"
                and act23[A_GRIP[arm]] < 0):
            rob = self.w.scene.robots[0]
            if not rob._ag_obj_constraint_params.get(arm):
                palm = _np(rob.eef_links[arm].get_position_orientation()[0])
                tpos = _np(self.target.get_position_orientation()[0])
                d = float(np.linalg.norm(palm - tpos))
                # STICKY RUNG (v21m): at radius <= 0.10 the assist requires ACTUAL
                # finger contact — the hand must genuinely reach and touch. Ranged
                # engage is only the bootstrap rung; radius 0 = native eval AG.
                contact_ok = (self.ag_assist_range > 0.045) or self._finger_contact()
                if d < self.ag_assist_range and contact_ok:
                    import torch as th
                    # PULL-IN goes INTO THE FINGER CAGE (v21m; was 10 cm off the wrist
                    # — the filmed "air grasp"): place the object at the fingertip
                    # midpoint so the held state is a real grasp pose from rung one.
                    fingers = _np(th.stack(
                        [l.get_position_orientation()[0]
                         for l in rob.finger_links[arm]]).mean(dim=0))
                    if float(np.linalg.norm(tpos - fingers)) > 0.03:
                        _, tq = self.target.get_position_orientation()
                        self.target.set_position_orientation(
                            position=th.as_tensor(fingers, dtype=th.float32),
                            orientation=tq)
                        tpos = fingers
                    _rl = next(k for k, v in self.target.links.items()
                               if v is self.target.root_link)
                    _jt = rob._get_assisted_grasp_joint_type(self.target, _rl)
                    rob._establish_grasp(self.target, _rl, arm,
                                         th.as_tensor(tpos, dtype=th.float32), _jt)
                    self._assist_used = True
        self.steps += 1
        s = self._proprio61()
        self._hist = (self._hist + [self._frame_feat(s)])[-STACK:]
        success = self._success()
        contact = self._finger_contact()
        r = 0.0
        ag_now = bool(getattr(self.robot, "_ag_obj_constraint_params",
                              {}).get(self.active_arm))
        if self.shaping:
            _d = self._dist(s)
            r -= 0.01 * _d
            if self.entry.get("family") == "pick_up_from" and _d < 0.10:
                r -= 0.05 * _d   # steepened terminal-zone gradient (sticky escalation)
            if (self.entry.get("family") == "press"
                    and getattr(self, "hold_anchor", None) is not None):
                r -= 0.01 * self._dist(s)   # transport contexts: double approach pull
            if contact and not self.contact_made:
                r += 0.5   # sticky-rung escalation: contact is the scarce event
                self.contact_made = True
            if contact:
                self.dwell += 1
                if self.dwell <= 5 and self.dwell_bonus < 0.299:
                    r += 0.02
                    self.dwell_bonus += 0.02
            else:
                self.dwell = 0
            # grasp-family shaping (v21h): the press-era terms end at contact, leaving a
            # sparse desert between "grasp acquired" and "lifted +5cm held 15" — the exact
            # gap where v21g stalled. One-time acquisition bonus + potential-based lift
            # progress (pays only NEW height, unfarmable by hovering).
            if (self.entry.get("family") == "press"
                    and getattr(self, "hold_anchor", None) is not None):
                _disp = float(np.linalg.norm(
                    _np(self.target.get_position_orientation()[0]) - self.hold_anchor))
                _ex = max(0.0, _disp - 0.05)
                if not hasattr(self, "_drift_pot"):
                    self._drift_pot = 0.0
                r -= 1.5 * (_ex - self._drift_pot)   # potential-based: pays going BACK
                self._drift_pot = _ex
            if self.entry.get("family") == "pick_up_from":
                if ag_now and not self._ag_had:
                    r += 0.3
                    self._ag_had = True
                if ag_now and self.entry.get("lift_z") is not None:
                    tzp = _np(self.target.get_position_orientation()[0])
                    prog = min(max(float(tzp[2]) - self.entry["lift_z"], 0.0), 0.10)
                    if prog > self._lift_prog:
                        r += 3.0 * (prog - self._lift_prog)
                        self._lift_prog = prog
                    # stillness shaping: once lifted, swinging the object costs
                    prev = getattr(self, "_tp_prev_shape", None)
                    if prev is not None and prog >= 0.05:
                        r -= 2.0 * max(0.0, float(np.linalg.norm(tzp - prev)) - 0.005)
                    self._tp_prev_shape = tzp.copy()
        if success:
            r += 1.0
        done = success or self.steps >= BUDGET
        return self._obs(), r, done, {"success": success, "steps": self.steps,
                                      "dist": self._dist(s), "demo": self.entry["demo"],
                                      "stage": self.entry["stage"],
                                      "assist_fired": self._assist_used, "ag": ag_now,
                                      "lift_prog": round(self._lift_prog, 3)}

    # ---- obs ----------------------------------------------------------------------------
    def _live_target_base(self):
        tp, tq = self.target.get_position_orientation()
        meta_world = _np(tp) + q2r(_np(tq)) @ self._p_off
        return self._world_to_base(meta_world) + self._point_bias

    def _obs(self):
        s = self._proprio61()
        arm = self.active_arm
        self.target_base = self._live_target_base()
        v_base = self.target_base - s[P[arm]["eef_pos"]]
        v_ee = q2r(s[P[arm]["eef_quat"]]).T @ v_base
        l2 = self._l2_features(s) if self.l2_on else np.zeros(L2_GRID ** 3, np.float32)
        goal = np.zeros(len(FAMILIES), np.float32)
        goal[FAMILIES.index(self.entry["family"])] = 1.0
        return np.concatenate([np.concatenate(self._hist), v_ee, [self.aff_conf],
                               l2, goal]).astype(np.float32)

    def _l2_features(self, s):
        """Synthesized EE-frame occupancy of the target's oriented bbox (D2).
        Same 64-dim grid format the real mapper pools to; fidelity check pending."""
        arm = self.active_arm
        ee_p = s[P[arm]["eef_pos"]]
        R_ee = q2r(s[P[arm]["eef_quat"]])
        tp, tq = self.target.get_position_orientation()
        tp = _np(tp)
        R_t = q2r(_np(tq))
        # target OBB corners/samples in world (aabb extent approximated from native bbox)
        try:
            ext = _np(self.target.native_bbox)
        except Exception:  # noqa: BLE001
            ext = np.array([0.2, 0.12, 0.12])
        n = 6
        lin = [np.linspace(-ext[i] / 2, ext[i] / 2, n) for i in range(3)]
        pts = np.stack(np.meshgrid(*lin, indexing="ij"), -1).reshape(-1, 3) @ R_t.T + tp
        # world -> base -> EE frame
        bp, bq = self.robot.get_position_orientation()
        pts_base = (pts - _np(bp)) @ q2r(_np(bq))
        pts_ee = (pts_base - ee_p) @ R_ee
        # occupancy over cube [0, L2_CUBE] ahead in +x of EE, centered in y/z
        g = np.zeros((L2_GRID,) * 3, np.float32)
        cell = L2_CUBE / L2_GRID
        idx = np.floor((pts_ee - np.array([0.0, -L2_CUBE / 2, -L2_CUBE / 2])) / cell)
        ok = np.all((idx >= 0) & (idx < L2_GRID), axis=1)
        for i, j, k in idx[ok].astype(int):
            g[i, j, k] = 1.0
        return g.reshape(-1)

    def _proprio61(self):
        obs = self.env.get_obs()[0]

        def find(node, sub):
            if isinstance(node, dict):
                for k, v in node.items():
                    r = find(v, sub)
                    if r is not None:
                        return r
                    if sub in str(k):
                        return v
            return None
        return _np(find(obs, "proprio")).reshape(-1).astype(np.float64)

    def _frame_feat(self, s):
        arm = self.active_arm
        return np.concatenate([s[P[arm]["arm_qpos"]], s[P[arm]["arm_qvel"]],
                               s[P[arm]["grip_qpos"]], s[P[arm]["grip_qvel"]],
                               s[P["trunk_qpos"]], s[P["trunk_qvel"]]])

    def _dist(self, s):
        return float(np.linalg.norm(self.target_base - s[P[self.active_arm]["eef_pos"]]))

    def _world_to_base(self, p):
        bp, bq = self.robot.get_position_orientation()
        return q2r(_np(bq)).T @ (np.asarray(p) - _np(bp))

    def _success(self):
        fam = self.entry["family"]
        if fam == "press":
            try:
                from omnigibson.object_states import ToggledOn
                if not bool(self.target.states[ToggledOn].get_value()):
                    return False
                # ON-STATION press (v22f): a held-object press only counts if the object
                # is within 10 cm of its hold anchor — ToggledOn-after-an-orbit is the
                # same fraud the grasp stillness gate killed, one phase later
                # (the -16.26-return "success" was a swinging press; user film review #5)
                anch = getattr(self, "hold_anchor", None)
                if anch is not None:
                    tp = _np(self.target.get_position_orientation()[0])
                    if float(np.linalg.norm(tp - anch)) > 0.10:
                        return False
                return True
            except Exception:  # noqa: BLE001
                return False
        if fam == "pick_up_from":
            ag = getattr(self.robot, "_ag_obj_constraint_params", {}).get(self.active_arm)
            tp = _np(self.target.get_position_orientation()[0])
            base = self.entry.get("lift_z")
            margin = self.entry.get("lift_success", 0.05)
            lifted = base is not None and tp[2] > base + margin
            # QUASI-STATIC hold (v21u): 'held' means STILL — the orbit exploit satisfied
            # AG∧height∧dwell while swinging the object in circles (filmed), and poisoned
            # bridge states with momentum. Dwell counts only below ~0.15 m/s.
            prev = getattr(self, "_tp_prev", None)
            still = prev is not None and float(np.linalg.norm(tp - prev)) < 0.005
            self._tp_prev = tp.copy()
            if ag and lifted and still:
                self._lift_dwell = getattr(self, "_lift_dwell", 0) + 1
            else:
                self._lift_dwell = 0
            return self._lift_dwell >= 15          # held STILL ≥15 steps
        raise NotImplementedError(fam)

    def _finger_contact(self):
        try:
            contacts, _ = self.robot._find_gripper_contacts(arm=self.active_arm)
            return any(self.target.name in c for c in contacts)
        except Exception:  # noqa: BLE001
            return False

    # ---- demo-replay seeding (D3) -------------------------------------------------------
    def seed_from_demo(self, entry, max_steps=None, chain_press_entry=None):
        """Replay the demo's own actions from the TRUE demo frame (playback restore —
        NOT a snapshot: snapshots are settled ~90 steps past the frame, so frame-indexed
        actions miss; root-caused 2026-08-19). Returns (transitions, success).

        Do NOT re-pin AG here: playback restore carries the grasp natively, and a second
        constraint lands with disjointed transforms — the snap perturbs the held object
        and the frame-indexed replay misses (v21b failure mode, 2026-08-19 PM)."""
        import h5py
        from omnigibson.object_states import ToggledOn
        from reverse_curriculum_collect import restore_to_frame
        # Grasp seeds must restore BEFORE the gripper-closure transition: the challenge
        # recordings' state stream does not carry the AG constraint, so restoring at the
        # anchor (post-closure) yields closed-fingers-with-no-weld and the lift leaves the
        # object behind (d30 diag, 2026-08-19). Replaying through approach->closure lets
        # AG engage naturally, exactly as it did during demo collection.
        start = entry["frame"]
        if entry.get("family") == "pick_up_from" and entry.get("grasp_closure_frame"):
            start = max(0, entry["grasp_closure_frame"] - 20)
        if max_steps is None:
            if chain_press_entry is not None and entry.get("press_frame"):
                # CHAIN seed: replay grasp THROUGH press as one planted trajectory
                max_steps = (entry["press_frame"] - start) + 80
            elif entry.get("family", "press") == "press" and entry.get("press_frame"):
                max_steps = (entry["press_frame"] - start) + 60
            else:
                max_steps = (entry["frame"] - start) + 240  # closure->lift+5cm + 15-dwell
        epid = int(sorted(k.split("_")[1] for k in self.w.input_hdf5["data"].keys()
                          if k.startswith("demo_"))[0])
        restore_to_frame(self.w, epid, start)
        self.entry = entry
        self.target = [o for o in self.w.scene.objects
                       if entry["target_name_sub"] in o.name.lower()][0]
        # Held-object entries (press-in-hand scenes): recordings carry no AG, so the
        # object falls out of the phantom grip on the first physics step. Re-create the
        # hold at the object's CURRENT pose — coincident anchors, no pull, no snap
        # (grip-fidelity principle; NOT the recorded-pose re-pin that broke v21b).
        if entry.get("holding_arm"):
            import torch as th
            _rob = self.w.scene.robots[0]
            if not _rob._ag_obj_constraint_params.get(entry["holding_arm"]):
                _rl = next(k for k, v in self.target.links.items()
                           if v is self.target.root_link)
                _jt = _rob._get_assisted_grasp_joint_type(self.target, _rl)
                _tp = _np(self.target.get_position_orientation()[0])
                _rob._establish_grasp(self.target, _rl, entry["holding_arm"],
                                      th.as_tensor(_tp, dtype=th.float32), _jt)
        if entry.get("family", "press") == "press":   # family-routed predicate reset
            try:
                if self.target.states[ToggledOn].get_value():
                    self.target.states[ToggledOn].set_value(False)
            except KeyError:
                pass
        self.active_arm = entry["active_arm"]
        s = self._proprio61()
        self.hold_act = self._stationary_act(s)
        self._point_bias = np.zeros(3)
        self.aff_conf = 0.75
        self.target_base = self._live_target_base()
        self._hist = [self._frame_feat(s)] * STACK
        self.steps = 0
        self.dwell = 0
        self.dwell_bonus = 0.0
        self.contact_made = False
        self._lift_dwell = 0
        with h5py.File(f"/root/rawdemos/task-0000/episode_{entry['demo']:08d}.hdf5", "r") as f:
            key = sorted(k for k in f["data"].keys() if k.startswith("demo_"))[0]
            acts = f[f"data/{key}/action"][start:start + max_steps]
        trans = []
        obs = self._obs()
        rob = self.w.scene.robots[0]
        established = False
        root_link_name = next(k for k, v in self.target.links.items()
                              if v is self.target.root_link)

        def _tracked(cmd, q, tol=0.03):
            return (np.abs(q[P["left"]["arm_qpos"]] - cmd[7:14]).max() < tol and
                    np.abs(q[P["right"]["arm_qpos"]] - cmd[15:22]).max() < tol and
                    np.abs(q[P["trunk_qpos"]] - cmd[A_TORSO]).max() < tol)

        for act23 in acts:
            s_pre = self._proprio61()
            # inverse-map the demo's absolute command to the 12-d policy action (pre-step state)
            a12 = np.zeros(12, np.float32)
            arm = self.active_arm
            a12[:7] = np.clip((np.asarray(act23)[A_ARM[arm]] - s_pre[P[arm]["arm_qpos"]])
                              / self.scale[:7], -1, 1)
            a12[7:11] = np.clip((np.asarray(act23)[A_TORSO] - s_pre[P["trunk_qpos"]])
                                / self.scale[7:11], -1, 1)
            a12[11] = np.clip((np.asarray(act23)[A_GRIP[arm]] - self.grip_center)
                              / self.grip_half, -1, 1)
            cmd = np.asarray(act23, np.float32)
            # closed-loop replay: this env's gains are softer than the collection rig's,
            # so hold each absolute target until tracked or the trajectory cuts corners
            # (the grasp-approach dip gets skipped entirely; diag 2026-08-19)
            for _sub in range(8):
                self.env.step(cmd)
                if _tracked(cmd, self._proprio61()):
                    break
            # magnetic AG, collection-rig behavior: demos command grasp from up to ~30 cm
            # and the rig establishes the constraint (recordings carry no AG state, and
            # finger-raycast AG can never fire at that range)
            if (entry.get("family") == "pick_up_from" and not established
                    and cmd[A_GRIP[arm]] < 0
                    and not rob._ag_obj_constraint_params.get(arm)):
                palm = _np(rob.eef_links[arm].get_position_orientation()[0])
                tpos = _np(self.target.get_position_orientation()[0])
                d = float(np.linalg.norm(palm - tpos))
                if d < 0.45:
                    import torch as th
                    # pull-in to the FINGER CAGE (v21m, same as RL assist path)
                    fingers = _np(th.stack(
                        [l.get_position_orientation()[0]
                         for l in rob.finger_links[arm]]).mean(dim=0))
                    if float(np.linalg.norm(tpos - fingers)) > 0.03:
                        _, tq = self.target.get_position_orientation()
                        self.target.set_position_orientation(
                            position=th.as_tensor(fingers, dtype=th.float32),
                            orientation=tq)
                        tpos = fingers
                    jt = rob._get_assisted_grasp_joint_type(self.target, root_link_name)
                    rob._establish_grasp(self.target, root_link_name, arm,
                                         th.as_tensor(np.asarray(tpos, np.float32)), jt)
                    established = True
                    if chain_press_entry is not None:
                        # goal flip at the grasp moment — the seed teaches BOTH
                        # conditioned phases along one demo trajectory
                        self.entry = dict(chain_press_entry)
                        self.active_arm = chain_press_entry["active_arm"]
                        self.hold_act = self._stationary_act(self._proprio61())
                        self._chain_flip_step = len(trans)
                        self.last_chain_rungs = []
                        self.dwell = 0
                        self.dwell_bonus = 0.0
                        self.contact_made = False
                        self._lift_dwell = 0
                        self.target_base = self._live_target_base()
            self.steps += 1
            if (chain_press_entry is not None and established
                    and getattr(self, "_chain_flip_step", None) is not None
                    and len(getattr(self, "last_chain_rungs", [])) < 3
                    and (len(trans) - self._chain_flip_step) % 2 == 1):
                # demo-transport rung: the human's left arm is partway across —
                # snapshot as a reverse-curriculum start for bridge-press
                import omnigibson as _og
                import os as _os
                _st = _og.sim.dump_state(serialized=True)
                _st = _st.cpu().numpy() if hasattr(_st, "cpu") else np.asarray(_st)
                _os.makedirs("/root/transport_rungs", exist_ok=True)
                _bp = (f"/root/transport_rungs/d{entry['demo']}"
                       f"_{len(self.last_chain_rungs)}.npz")
                np.savez_compressed(_bp, state=_st)
                self.last_chain_rungs.append(_bp)
            s = self._proprio61()
            self._hist = (self._hist + [self._frame_feat(s)])[-STACK:]
            nobs = self._obs()
            succ = self._success()
            trans.append((obs, a12, 1.0 if succ else 0.0, nobs, float(succ)))
            obs = nobs
            if succ:
                return trans, True
        return trans, False

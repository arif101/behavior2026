"""Commit-window env wrapper per CONTACT_SKILL_SPEC_v1.md §1/§2/§4 (+ [A-2026-08-13]).

Owns: restore → AG re-establishment → validity check → domain randomization → skill-step
loop (12-d tanh action → 23-d env action) → reward → reset. The learner (rlpd_sac.py) sees
only [-1,1] actions and the obs layout from /root/skill_buffer_prior/skill_buffer_meta.json.

RESTORE VALIDITY ([A-2026-08-13], probe-driven): og.sim.load_state does not re-create
assisted-grasp constraints. At every reset with a holding arm we call
robot._establish_grasp(...) (documented externally-callable) and then verify
AG engaged ∧ |Δ target z| < 1 mm over 30 settle steps; failing states are discarded from
the bank (collector outcome-filter pattern).

Action scaling: a in [-1,1]^12 → physical deltas via SCALE (arm 7 + torso 4) + gripper
absolute command. SCALE derivation pinned in build_start_bank.py (2x prior-buffer p99,
clipped to [0.01, 0.08] rad; gripper span from demo actions); written to
/root/skill_wrapper_scale.json so learner-side prior normalization CANNOT drift from the
wrapper (single source of truth).

Reward (§4): sparse ToggledOn +1 terminal; shaping (training-only, ablatable flags):
-0.01*dist(EE, frozen pt)/step, +0.1 on first finger contact with the target object,
+dwell-counter progress (0.02 per consecutive contact step, capped at 5 steps).
"""

import json

import numpy as np

STACK = 5
PER_FRAME = 26
BUDGET = 300


def q2r(q):
    x, y, z, w = [float(v) for v in q]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _np(v):
    return v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)


# 61-d eval proprio slices (EVAL_PROPRIO_KEYS order)
P = {"left": {"arm_qpos": slice(3, 10), "arm_qvel": slice(10, 17), "eef_pos": slice(17, 20),
              "eef_quat": slice(20, 24), "grip_qpos": slice(24, 26), "grip_qvel": slice(26, 28)},
     "right": {"arm_qpos": slice(28, 35), "arm_qvel": slice(35, 42), "eef_pos": slice(42, 45),
               "eef_quat": slice(45, 49), "grip_qpos": slice(49, 51), "grip_qvel": slice(51, 53)},
     "trunk_qpos": slice(53, 57), "trunk_qvel": slice(57, 61)}
A_TORSO = slice(3, 7)
A_ARM = {"left": slice(7, 14), "right": slice(14, 21)}
A_GRIP = {"left": 21, "right": 22}


class SkillCommitEnv:
    """One (demo hdf5, start-state bank) pair. Isaac env-reuse leaks -> chunk per demo like
    the collector; the training driver cycles processes."""

    def __init__(self, wrapper, bank_entries, meta_path="/root/skill_buffer_prior/skill_buffer_meta.json",
                 scale_path="/root/skill_wrapper_scale.json", shaping=True, seed=0,
                 randomize=True):
        import omnigibson  # noqa: F401  (env must be built by caller inside behavior env)
        self.w = wrapper
        self.env = wrapper.env
        self.robot = wrapper.scene.robots[0]
        self.target = [o for o in wrapper.scene.objects if "radio" in o.name.lower()][0]
        self.bank = bank_entries          # dicts: demo, frame, holding_arm|None, active_arm
        self.meta = json.load(open(meta_path))
        sc = json.load(open(scale_path))
        self.scale = np.asarray(sc["scale"], np.float32)          # 12-d
        self.grip_center, self.grip_half = sc["grip_center"], sc["grip_half"]
        self.shaping = shaping
        self.randomize = randomize
        self.rng = np.random.default_rng(seed)
        self.l2_dim = self.meta["obs_layout"]["l2_geometry"][1] - self.meta["obs_layout"]["l2_geometry"][0]
        self._hist = []
        self._demo_actions = None

    # ---- AG re-establishment ([A-2026-08-13]) ---------------------------------------------
    def _reestablish_ag(self, arm):
        r = self.robot
        in_hand = None
        try:
            in_hand = r._calculate_in_hand_object(arm=arm)
        except Exception:  # noqa: BLE001
            pass
        if in_hand is not None:
            obj, link = in_hand
        else:
            obj, link = self.target, list(self.target.links.keys())[0]
        jt = r._get_assisted_grasp_joint_type(obj, link)
        if jt is None:
            return False
        cp = None
        try:
            cp = r._find_finger_contact_position(arm, obj.links[link].prim_path)
        except Exception:  # noqa: BLE001
            pass
        if cp is None:
            import torch as th
            cp = th.stack([l.get_position_orientation()[0]
                           for l in r.finger_links[arm]]).mean(dim=0)
        r._establish_grasp(obj, link, arm, cp, jt)
        params = getattr(r, "_ag_obj_constraint_params", {})
        return params.get(arm) is not None

    def _ag_ok(self, arm):
        return getattr(self.robot, "_ag_obj_constraint_params", {}).get(arm) is not None

    # ---- reset ----------------------------------------------------------------------------
    def reset(self, entry=None, max_tries=8):
        """Restore a bank state, fix AG, validate, randomize, build obs. Returns obs or
        raises RuntimeError if max_tries bank entries fail validity (all get discarded)."""
        from reverse_curriculum_collect import restore_to_frame
        import h5py
        for _ in range(max_tries):
            e = entry or self.bank[int(self.rng.integers(len(self.bank)))]
            if self._demo_actions is None:
                with h5py.File(self.w.input_hdf5.filename, "r") as f:
                    key = sorted(k for k in f["data"].keys() if k.startswith("demo_"))[0]
                    self._demo_actions = f[f"data/{key}/action"][:]
            epid = int(sorted(k.split("_")[1] for k in self.w.input_hdf5["data"].keys()
                              if k.startswith("demo_"))[0])
            restore_to_frame(self.w, epid, e["frame"])
            hold_act = self._demo_actions[min(e["frame"], len(self._demo_actions) - 1)]

            ok = True
            if e.get("holding_arm"):
                self._reestablish_ag(e["holding_arm"])
                # 60-step settle rides out the ~3-7 mm attachment-snap transient
                # (probe_ag_fix), THEN a 30-step <1 mm stability window is the check
                for _ in range(60):
                    self.env.step(hold_act)
                z0 = float(_np(self.target.get_position_orientation()[0])[2])
                for _ in range(30):
                    self.env.step(hold_act)
                dz = abs(float(_np(self.target.get_position_orientation()[0])[2]) - z0)
                ok = self._ag_ok(e["holding_arm"]) and dz < 0.001
                if not ok:
                    e["invalid"] = True
                    self.bank = [b for b in self.bank if not b.get("invalid")]
                    if not self.bank:
                        raise RuntimeError("start bank exhausted: all entries failed AG validity")
                    entry = None
                    continue
            else:
                # no holding arm expected — still verify the target is actually supported
                # (a missed mid-carry state would free-fall here): 30-step settle,
                # |Δz| < 5 mm or the entry is discarded
                z0 = float(_np(self.target.get_position_orientation()[0])[2])
                for _ in range(30):
                    self.env.step(hold_act)
                dz = abs(float(_np(self.target.get_position_orientation()[0])[2]) - z0)
                if dz > 0.005:
                    e["invalid"] = True
                    self.bank = [b for b in self.bank if not b.get("invalid")]
                    if not self.bank:
                        raise RuntimeError("start bank exhausted: all entries failed validity")
                    entry = None
                    continue

            # Pre-press states can restore with ToggledOn ALREADY TRUE (the demo's toggle
            # precedes the distance-minimum press anchor, or the EE restores inside the
            # toggle zone) -> 1-step false successes that poison training. Clear it and
            # verify it stays cleared; a re-latch means the state is past the press —
            # discard the entry.
            try:
                from omnigibson.object_states import ToggledOn
                if self.target.states[ToggledOn].get_value():
                    self.target.states[ToggledOn].set_value(False)
                    self.env.step(hold_act)
                if self.target.states[ToggledOn].get_value():
                    e["invalid"] = True
                    self.bank = [b for b in self.bank if not b.get("invalid")]
                    if not self.bank:
                        raise RuntimeError("start bank exhausted: all entries re-latch ToggledOn")
                    entry = None
                    continue
            except KeyError:
                pass  # object has no ToggledOn state; reward path will report success=False

            self.active_arm = e["active_arm"]
            self.hold_act = hold_act
            obs61 = self._proprio61()
            # frozen affordance target ([§2]: no re-targeting mid-attempt): metalink in base
            # frame + measured head-error noise (sigma 2.6 cm, conf-conditioned)
            tp, tq = self.target.get_position_orientation()
            p_off = np.asarray(json.load(open("/root/behavior2026/labels/metalink_offset.json"))
                               ["offset_pos_root_frame"]) \
                if not hasattr(self, "_p_off") else self._p_off
            self._p_off = p_off
            meta_world = _np(tp) + q2r(_np(tq)) @ p_off
            self.target_base = self._world_to_base(meta_world)
            self.aff_conf = 0.75
            if self.randomize:
                self.target_base = self.target_base + self.rng.normal(0, 0.026, 3)
                self.aff_conf = float(np.clip(self.rng.normal(0.75, 0.05), 0.5, 1.0))
                jitter = self.rng.uniform(-0.03, 0.03, 7)
                q = _np(self.robot.get_joint_positions()).copy()
                arm_idx = [i for i, nm in enumerate(self.robot.dof_names_ordered)
                           if "arm" in nm.lower() and self.active_arm in nm.lower()]
                for k, i in enumerate(arm_idx[:7]):
                    q[i] += jitter[k]
                self.robot.set_joint_positions(_to_th(q))
                import omnigibson as og
                og.sim.render()
                obs61 = self._proprio61()
            self._hist = [self._frame_feat(obs61)] * STACK
            self.steps = 0
            self.dwell = 0
            self.contact_made = False
            return self._obs()
        raise RuntimeError("reset failed after max_tries")

    # ---- step -----------------------------------------------------------------------------
    def step(self, a):
        a = np.clip(np.asarray(a, np.float32), -1, 1)
        obs61 = self._proprio61()
        act23 = np.zeros(23, np.float32)
        arm = self.active_arm
        act23[:] = self.hold_act  # non-active channels hold the restore-frame command
        act23[0:3] = 0.0
        act23[A_ARM[arm]] = obs61[P[arm]["arm_qpos"]] + a[:7] * self.scale[:7]
        act23[A_TORSO] = obs61[P["trunk_qpos"]] + a[7:11] * self.scale[7:11]
        act23[A_GRIP[arm]] = self.grip_center + a[11] * self.grip_half
        self.env.step(act23)
        self.steps += 1

        obs61 = self._proprio61()
        self._hist = (self._hist + [self._frame_feat(obs61)])[-STACK:]
        toggled = self._toggled_on()
        contact = self._finger_contact()
        r = 0.0
        if self.shaping:
            r -= 0.01 * self._dist(obs61)
            if contact and not self.contact_made:
                r += 0.1
                self.contact_made = True
            self.dwell = self.dwell + 1 if contact else 0
            r += 0.02 * min(self.dwell, 5)
        if toggled:
            r += 1.0
        done = toggled or self.steps >= BUDGET
        return self._obs(), r, done, {"success": toggled, "steps": self.steps,
                                      "dist": self._dist(obs61)}

    # ---- internals ------------------------------------------------------------------------
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
            return None
        v = find(obs, "proprio")
        return _np(v).reshape(-1).astype(np.float64)

    def _frame_feat(self, s):
        arm = self.active_arm
        return np.concatenate([s[P[arm]["arm_qpos"]], s[P[arm]["arm_qvel"]],
                               s[P[arm]["grip_qpos"]], s[P[arm]["grip_qvel"]],
                               s[P["trunk_qpos"]], s[P["trunk_qvel"]]])

    def _obs(self):
        s = self._proprio61()
        arm = self.active_arm
        v_base = self.target_base - s[P[arm]["eef_pos"]]
        v_ee = q2r(s[P[arm]["eef_quat"]]).T @ v_base
        return np.concatenate([np.concatenate(self._hist), v_ee, [self.aff_conf],
                               np.zeros(self.l2_dim)]).astype(np.float32)

    def _dist(self, s):
        return float(np.linalg.norm(self.target_base - s[P[self.active_arm]["eef_pos"]]))

    def _world_to_base(self, p_world):
        bp, bq = self.robot.get_position_orientation()
        return q2r(_np(bq)).T @ (np.asarray(p_world) - _np(bp))

    def _toggled_on(self):
        try:
            from omnigibson.object_states import ToggledOn
            return bool(self.target.states[ToggledOn].get_value())
        except Exception:  # noqa: BLE001
            return False

    def _finger_contact(self):
        try:
            contacts, _ = self.robot._find_gripper_contacts(arm=self.active_arm)
            return any(self.target.name in c for c in contacts)
        except Exception:  # noqa: BLE001
            return False


def _to_th(x):
    import torch as th
    return th.as_tensor(x, dtype=th.float32)

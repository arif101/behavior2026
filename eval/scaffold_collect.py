"""SCAFFOLD ENGINE (PREP e) — corrective-data generator, wraps the policy at COLLECTION time.

Never runs in a submission. The dual-control car: the policy drives; on measured stall
signatures the scaffold intervenes structurally, returns the robot to a demo-like state, and
lets the policy retry. Everything is recorded; only outcome-successful segments become RaC
(rewind-then-correct) training clips for Run 2. Brittleness is acceptable by design — sloppy
thresholds cost collection YIELD, never data QUALITY (the outcome filter is the firewall).

STALL DETECTORS (from the measured legal-baseline failures):
  frozen   : arm-command std < FREEZE_STD over WINDOW steps            (run 2: 0.021)
  no-prog  : min EE-target distance improves < PROG_EPS over WINDOW    (runs 4/5: flat 0.57m)
  bad-grip : gripper closed while dist > 0.25 m for WINDOW steps       (runs 2/3 signature)

INTERVENTIONS (joint-space, from the robot's own recent history — no IK, no object state):
  rewind   : interpolate joints back to the posture REWIND_STEPS ago, then add a torso
             perturbation (the dive axis — the dimension failures never explore) and,
             optionally, a ledger-directed base/arm azimuth nudge
  wrist-orient : small wrist-joint sweep until the affordance point enters the wrist FOV
  force-open   : command grippers open for OPEN_STEPS (clears closed-on-nothing)

RECORDING: per-step ring buffer of (proprio, action, dist, conf, phase-tag). Post-episode,
segments [re-entry state -> progress event] are cut and saved as RaC clips
(/root/rac_clips/ep{N}_seg{k}.npz) iff the outcome filter passes:
  progress = dist drop > 0.10 m sustained, OR stage advance, OR success.
The INTERVENTION steps themselves are tagged and EXCLUDED from clips (we distill the
policy's own recovery, not the scaffold's scripted motion).

Usage (collection campaigns only):
  --env-wrapper behavior2026_eval.scaffold_collect.ScaffoldCollectWrapper
"""

import json
import time

import numpy as np

from behavior2026_eval.affordance_map_fullres import AffordanceMapFullRes

WINDOW = 240            # ~8 s of sim
FREEZE_STD = 0.03
PROG_EPS = 0.04
REWIND_STEPS = 150      # ~5 s back
REWIND_DUR = 60         # interpolation length
TORSO_KICK = 0.35       # rad perturbation on the dive axis, sign-alternating
OPEN_STEPS = 20
MAX_INTERVENTIONS = 6
ARM_SLICE = slice(7, 21)
TORSO_SLICE = slice(3, 7)
GRIP_SLICE = slice(21, 23)


class ScaffoldCollectWrapper(AffordanceMapFullRes):
    def __init__(self, env):
        super().__init__(env=env)
        self._hist = []            # per-step dicts (proprio, action, dist, conf, tag)
        self._mode = "policy"      # policy | rewind | open
        self._mode_left = 0
        self._rewind_from = None
        self._rewind_to = None
        self._n_interventions = 0
        self._kick_sign = 1.0
        self._ledger = np.zeros(12)   # approach-sector visits (spatially-directed retry)

    # -------------------------------------------------------------- stall detection
    def _stalled(self):
        if len(self._hist) < WINDOW or self._n_interventions >= MAX_INTERVENTIONS:
            return None
        w = self._hist[-WINDOW:]
        acts = np.array([h["action"][ARM_SLICE] for h in w])
        dists = np.array([h["dist"] for h in w if h["dist"] is not None])
        if acts.std(0).mean() < FREEZE_STD:
            return "frozen"
        if len(dists) > WINDOW // 2 and (dists[: len(dists) // 2].min() - dists[len(dists) // 2:].min()) < PROG_EPS:
            grip = np.array([h["action"][GRIP_SLICE] for h in w])
            if (grip < 0).any() and (dists.min() > 0.25):
                return "bad-grip"
            if dists.min() > 0.18:
                return "no-progress"
        return None

    # -------------------------------------------------------------- interventions
    def _begin_rewind(self, reason):
        past = self._hist[-min(REWIND_STEPS, len(self._hist))]
        self._rewind_from = np.array(self._hist[-1]["proprio_cmdspace"])
        self._rewind_to = np.array(past["proprio_cmdspace"])
        # torso kick on the dive axis + ledger-directed sign
        self._rewind_to[TORSO_SLICE.start + 1] += TORSO_KICK * self._kick_sign
        self._kick_sign *= -1.0
        self._mode, self._mode_left = "rewind", REWIND_DUR
        self._n_interventions += 1
        self._tag = f"intervene:{reason}"

    def _scripted_action(self, policy_action):
        a = np.array(policy_action, dtype=np.float64)
        if self._mode == "rewind":
            f = 1.0 - self._mode_left / REWIND_DUR
            tgt = self._rewind_from + (self._rewind_to - self._rewind_from) * f
            a[TORSO_SLICE] = tgt[TORSO_SLICE]
            a[ARM_SLICE] = tgt[ARM_SLICE]
            a[GRIP_SLICE] = 1.0                      # open while retreating
            a[0:3] = 0.0                             # base still
        elif self._mode == "open":
            a[GRIP_SLICE] = 1.0
        self._mode_left -= 1
        if self._mode_left <= 0:
            self._mode = "policy"
        return a

    # -------------------------------------------------------------- step hook
    def step(self, action, n_render_iterations=1):
        tag = "policy"
        if self._mode != "policy":
            action = self._scripted_action(action)
            tag = self._mode
        else:
            reason = self._stalled()
            if reason == "bad-grip":
                self._mode, self._mode_left = "open", OPEN_STEPS
                self._n_interventions += 1
                tag = "intervene:open"
            elif reason:
                self._begin_rewind(reason)
                tag = self._tag

        out = self.env.step(action, n_render_iterations=n_render_iterations)
        obs = out[0] if isinstance(out, tuple) else out
        obs = self._inject(obs)

        # record
        prop = None
        node = obs.get(self._robot.name) if isinstance(obs, dict) else None
        if isinstance(node, dict) and "proprio" in node:
            prop = np.asarray(node["proprio"], np.float64).reshape(-1)
        d = None
        if self._last is not None:
            d = float(min(np.linalg.norm(self._last[0]), np.linalg.norm(self._last[1])))
        self._hist.append({
            "action": np.asarray(action, np.float64),
            "proprio": prop,
            # command-space posture for rewind targets: torso+arm joint targets ≈ last action
            "proprio_cmdspace": np.asarray(action, np.float64),
            "dist": d,
            "conf": self._stats["conf"][-1] if self._stats["conf"] else None,
            "tag": tag,
        })
        if isinstance(out, tuple):
            return (obs, *out[1:])
        return obs

    # -------------------------------------------------------------- RaC cutting
    def _cut_rac_clips(self, success):
        clips = []
        i = 0
        H = self._hist
        while i < len(H):
            if H[i]["tag"].startswith("intervene") or H[i]["tag"] in ("rewind", "open"):
                j = i
                while j < len(H) and H[j]["tag"] != "policy":
                    j += 1
                start = j                                  # re-entry: policy resumes
                d0 = next((h["dist"] for h in H[start:start + 30] if h["dist"]), None)
                end, ok = start, False
                for k in range(start, min(start + 900, len(H))):
                    dk = H[k]["dist"]
                    if d0 is not None and dk is not None and (d0 - dk) > 0.10:
                        end, ok = k + 60, True             # progress + a tail
                        break
                if success and not ok:
                    end, ok = min(start + 900, len(H)), True
                if ok and end - start > 40:
                    clips.append((start, min(end, len(H))))
                i = j + 1
            else:
                i += 1
        return clips

    def _dump_rac(self, success):
        import os
        os.makedirs("/root/rac_clips", exist_ok=True)
        clips = self._cut_rac_clips(success)
        for k, (s, e) in enumerate(clips):
            seg = self._hist[s:e]
            np.savez_compressed(
                f"/root/rac_clips/clip_{int(time.time())}_{k}.npz",
                actions=np.array([h["action"] for h in seg]),
                proprio=np.array([h["proprio"] for h in seg if h["proprio"] is not None]),
                dist=np.array([h["dist"] if h["dist"] is not None else np.nan for h in seg]),
                success=success,
            )
        with open("/root/scaffold_stats.json", "w") as f:
            json.dump({"interventions": self._n_interventions,
                       "clips_kept": len(clips), "steps": len(self._hist)}, f)
        return len(clips)

    def reset(self):
        if self._hist:
            n = self._dump_rac(success=False)   # success wiring: read result json post-hoc
            print(f"[scaffold] episode end: {self._n_interventions} interventions, {n} RaC clips")
        self._hist = []
        self._mode = "policy"
        self._n_interventions = 0
        return super().reset()

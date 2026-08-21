"""RaC collector v2 — the v1 scaffold engine + TRAINING-FORMAT obs capture.

WHY (measured 2026-08-06): the 95 banked v1 clips are UNCONVERTIBLE — the converter's
episode gate is literally `bool(d["success"]) and "head_rgb" in d.files`, v1 recorded only
(proprio, action, dist, conf, tag) by design, and the campaign archived no frames and no
sim state (one viz mp4/run). Training-format RaC data therefore requires RE-COLLECTION.
v2 subclasses the UNTOUCHED v1 wrapper (same stall detectors, interventions, outcome
filter) and adds env-side per-step capture of everything the converter's RaC branch
(convert_clips_to_parquet.py, `base_pose`-gated) consumes:

  per step, paired obs_t -> action_t (obs_t = the obs returned by the PREVIOUS env.step —
  the observation the policy computed action_t from; poses are read from the sim at the
  top of step(), i.e. the same state that rendered obs_t):
    head/left/right_rgb    JPEG q92 bytes (zed 720 / realsense 480 = converter native res)
    head/left/right_depth  uint16-mm PNG bytes (converter PIL-decodes to uint16 mm)
    objpose_radio_89       radio world pos+quat xyzw. LITERAL converter-contract key; the
                           actual scene object name is stored in meta_object_name.
    base_pose              robot base world pos+quat xyzw -> exact per-frame world->base
                           (live episodes MOVE the base; the splice-path static-prefix
                           Kabsch would be invalid here)
    campose_head/left/right  camera pose in the BASE frame, 7-vec [pos, quat xyzw] — the
                           same convention as the robot2cam_pose columns / project()
  per clip:
    success=True           converter accept semantics = "outcome-filtered keeper"
    episode_success        the episode's actual terminated flag (honest record)
    radio_rest_z           radio z right after reset — the stage-label lift reference
                           (clip frame 0 is NOT at rest for mid-recovery clips)
    task_instance_id       RAC2_TASK_INSTANCE_ID env (campaign instance 301), else -1

Frames spool to disk per step (RAC2_SPOOL, ~2 GB/episode, deleted after cutting) so RAM
stays flat under the 25 GB cgroup; kept clips land in RAC2_OUT as rac_<epid>_<k>.npz
(object arrays of encoded bytes -> np.load(allow_pickle=True), the RC-collector format).

Smoke hooks (NEVER set in real collection):
  RAC2_FORCE_KICK=<step>  force one rewind intervention at that hist step
  RAC2_ACCEPT_ANY=1       if the outcome filter keeps nothing, keep one post-re-entry
                          segment anyway (pipeline exercise only)

Usage (mirrors campaign_run1.sh; see campaign_rac2.sh):
  --env-wrapper behavior2026_eval.scaffold_collect_v2.ScaffoldCollectV2Wrapper
"""

import io
import json
import os
import pickle
import shutil
import time

import numpy as np
from PIL import Image

from behavior2026_eval.scaffold_collect import ScaffoldCollectWrapper

OUT_DIR = os.environ.get("RAC2_OUT", "/root/rac2_clips")
STATS_DIR = os.environ.get("RAC2_STATS", "/root/rac2_stats")
SPOOL = os.environ.get("RAC2_SPOOL", "/root/rac2_spool")
FORCE_KICK = int(os.environ.get("RAC2_FORCE_KICK", "0"))
ACCEPT_ANY = os.environ.get("RAC2_ACCEPT_ANY", "0") == "1"
TASK_INSTANCE_ID = int(os.environ.get("RAC2_TASK_INSTANCE_ID", "-1"))

FRAME_KEYS = ("head_rgb", "left_rgb", "right_rgb", "head_depth", "left_depth", "right_depth")


def _np(v):
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    return np.asarray(v)


def _q2r(q):
    x, y, z, w = [float(v) for v in q]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _qmul(a, b):
    """Hamilton product, xyzw."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def _rel_pose(parent7, child7):
    """Child pose expressed in the parent frame, 7-vec [pos, quat xyzw]."""
    R = _q2r(parent7[3:7])
    pos = R.T @ (np.asarray(child7[:3], np.float64) - np.asarray(parent7[:3], np.float64))
    qp = np.asarray(parent7[3:7], np.float64)
    qinv = np.array([-qp[0], -qp[1], -qp[2], qp[3]]) / max(1e-12, float(qp @ qp))
    quat = _qmul(qinv, np.asarray(child7[3:7], np.float64))
    return np.concatenate([pos, quat])


def _enc_jpg(rgb):
    rgb = _np(rgb)[..., :3]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=92)
    return buf.getvalue()


def _enc_png16(depth_m):
    mm = np.clip(np.nan_to_num(_np(depth_m).astype(np.float64), nan=0.0) * 1000.0,
                 0, 65535).astype(np.uint16)
    buf = io.BytesIO()
    Image.fromarray(mm, mode="I;16").save(buf, "PNG", compress_level=3)
    return buf.getvalue()


class ScaffoldCollectV2Wrapper(ScaffoldCollectWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._ep_id = int(time.time())
        self._spool = None          # per-episode spool dir
        self._pending = None        # encoded frames of the obs the NEXT action pairs with
        self._poses = []            # per-step dict rows, aligned with self._hist
        self._radio = None
        self._rest_z = None
        self._pose7 = lambda o: np.concatenate([_np(o.get_position_orientation()[0]).astype(np.float64).reshape(3),
                                                _np(o.get_position_orientation()[1]).astype(np.float64).reshape(4)])
        os.makedirs(OUT_DIR, exist_ok=True)
        os.makedirs(STATS_DIR, exist_ok=True)

    # ---------------------------------------------------------------- capture helpers
    def _resolve_radio(self):
        tgt = getattr(self, "_targets", None)
        if tgt:
            return tgt[0]
        for o in self.env.scene.objects:
            if "radio" in getattr(o, "name", "").lower():
                return o
        return None

    def _capture_frames(self, obs):
        """Encode the 3-cam RGB-D of a (possibly tuple-wrapped) obs; None if unavailable."""
        if isinstance(obs, tuple):
            obs = obs[0]
        node = obs.get(self._robot.name) if isinstance(obs, dict) else None
        if not isinstance(node, dict):
            return None
        rec = {}
        for k, v in node.items():
            if not isinstance(v, dict):
                continue
            kl = k.lower()
            if "zed" in kl:
                short = "head"
            elif "left_realsense" in kl:
                short = "left"
            elif "right_realsense" in kl:
                short = "right"
            else:
                continue
            if "rgb" in v:
                rec[f"{short}_rgb"] = _enc_jpg(v["rgb"])
            if "depth_linear" in v:
                rec[f"{short}_depth"] = _enc_png16(v["depth_linear"])
        return rec if "head_rgb" in rec else None

    def _capture_poses(self):
        """Sim poses of the CURRENT state (== the state that rendered self._pending)."""
        row = {}
        try:
            base7 = self._pose7(self._robot)
            row["base_pose"] = base7
            if self._radio is not None:
                row["objpose_radio_89"] = self._pose7(self._radio)
            for sname, sensor in self._robot.sensors.items():
                snl = sname.lower()
                if "zed" in snl:
                    short = "head"
                elif "left_realsense" in snl:
                    short = "left"
                elif "right_realsense" in snl:
                    short = "right"
                else:
                    continue
                row[f"campose_{short}"] = _rel_pose(base7, self._pose7(sensor))
        except Exception as e:  # noqa: BLE001 — capture must never kill the rollout
            print(f"[rac2] pose capture error at step {len(self._hist)}: {e}", flush=True)
        return row

    # ---------------------------------------------------------------- hooks
    def reset(self):
        out = super().reset()          # dynamic dispatch: leftover hist dumps via OUR _dump_rac
        self._ep_id = int(time.time())
        self._spool = os.path.join(SPOOL, f"ep_{self._ep_id}")
        shutil.rmtree(self._spool, ignore_errors=True)
        os.makedirs(self._spool, exist_ok=True)
        self._poses = []
        self._radio = self._resolve_radio()
        if self._radio is None:
            print("[rac2] WARNING: no radio object resolved — objpose/rest_z absent", flush=True)
            self._rest_z = None
        else:
            self._rest_z = float(self._pose7(self._radio)[2])
            print(f"[rac2] radio={self._radio.name} rest_z={self._rest_z:.4f} "
                  f"spool={self._spool}", flush=True)
        self._pending = self._capture_frames(out)
        return out

    def step(self, action, n_render_iterations=1):
        # obs_t (rendered at the end of the previous step) pairs with THIS action; the sim
        # has not advanced since, so poses read now belong to the same state.
        t = len(self._hist)
        if self._pending is not None and self._spool is not None:
            with open(os.path.join(self._spool, f"s{t:05d}.pkl"), "wb") as f:
                pickle.dump(self._pending, f, protocol=pickle.HIGHEST_PROTOCOL)
            self._poses.append(self._capture_poses())
        else:
            self._poses.append({})

        if (FORCE_KICK > 0 and t == FORCE_KICK and self._mode == "policy"
                and self._n_interventions == 0):
            print(f"[rac2] SMOKE: forcing rewind intervention at step {t}", flush=True)
            self._begin_rewind("forced")

        out = super().step(action, n_render_iterations=n_render_iterations)
        self._pending = self._capture_frames(out)
        return out

    # ---------------------------------------------------------------- dump (replaces v1's)
    def _dump_rac(self, success):
        clips = self._cut_rac_clips(success)
        if ACCEPT_ANY and not clips:
            # SMOKE ONLY: keep one post-re-entry segment so the pipeline can be exercised.
            H = self._hist
            for i, h in enumerate(H):
                if h["tag"] != "policy" and any(g["tag"] == "policy" for g in H[i:]):
                    s = next(j for j in range(i, len(H)) if H[j]["tag"] == "policy")
                    if len(H) - s > 40:
                        clips = [(s, min(s + 300, len(H)))]
                    break
        kept = 0
        for k, (s, e) in enumerate(clips):
            e = min(e, len(self._hist), len(self._poses))
            if e - s <= 40:
                continue
            frames = {fk: [] for fk in FRAME_KEYS}
            missing = 0
            for t in range(s, e):
                p = os.path.join(self._spool or "", f"s{t:05d}.pkl")
                try:
                    with open(p, "rb") as f:
                        rec = pickle.load(f)
                    for fk in FRAME_KEYS:
                        frames[fk].append(rec.get(fk, b""))
                except Exception:
                    missing += 1
                    for fk in FRAME_KEYS:
                        frames[fk].append(b"")
            if missing > (e - s) // 10:
                print(f"[rac2] clip {k}: {missing}/{e - s} spool frames missing — SKIP", flush=True)
                continue
            seg = self._hist[s:e]
            prop0 = next((h["proprio"] for h in seg if h["proprio"] is not None), None)
            if prop0 is None:
                print(f"[rac2] clip {k}: no proprio — SKIP", flush=True)
                continue
            prop, last = [], prop0
            for h in seg:
                last = h["proprio"] if h["proprio"] is not None else last
                prop.append(last)
            pz = self._poses[s:e]
            def pose_arr(key, dim=7):
                rows = [r.get(key) for r in pz]
                fill = next((r for r in rows if r is not None), None)
                if fill is None:
                    return None
                out_rows, lastp = [], fill
                for r in rows:
                    lastp = r if r is not None else lastp
                    out_rows.append(lastp)
                return np.asarray(out_rows, np.float64).reshape(-1, dim)
            arrs = {
                "actions": np.asarray([h["action"] for h in seg], np.float64),
                "proprio": np.asarray(prop, np.float64),
                "dist": np.asarray([h["dist"] if h["dist"] is not None else np.nan for h in seg]),
                "tags": np.asarray([h["tag"] for h in seg]),
                "success": np.bool_(True),          # converter accept semantics: kept clip
                "episode_success": np.bool_(bool(success)),
                "task_instance_id": np.int64(TASK_INSTANCE_ID),
                "start_frame": np.int64(-1),        # no demo timeline: RaC family marker
            }
            for fk in FRAME_KEYS:
                arrs[fk] = np.array(frames[fk], dtype=object)
            for key in ("base_pose", "objpose_radio_89", "campose_head",
                        "campose_left", "campose_right"):
                a = pose_arr(key)
                if a is not None:
                    arrs[key] = a
            if self._rest_z is not None:
                arrs["radio_rest_z"] = np.float64(self._rest_z)
            if self._radio is not None:
                arrs["meta_object_name"] = np.str_(self._radio.name)
            fp = os.path.join(OUT_DIR, f"rac_{self._ep_id}_{k}.npz")
            np.savez_compressed(fp, **arrs)
            kept += 1
            print(f"[rac2] saved {fp}: {e - s} steps, missing_frames={missing}", flush=True)
        with open(os.path.join(STATS_DIR, f"ep_{self._ep_id}.json"), "w") as f:
            json.dump({"interventions": self._n_interventions, "clips_cut": len(clips),
                       "clips_kept": kept, "steps": len(self._hist),
                       "episode_success": bool(success),
                       "task_instance_id": TASK_INSTANCE_ID}, f)
        # also keep v1's side-file contract so campaign scripts keep working
        with open("/root/scaffold_stats.json", "w") as f:
            json.dump({"interventions": self._n_interventions,
                       "clips_kept": kept, "steps": len(self._hist)}, f)
        self._hist = []                # dump_on_done + reset would otherwise double-dump
        if self._spool is not None:
            shutil.rmtree(self._spool, ignore_errors=True)
        return kept

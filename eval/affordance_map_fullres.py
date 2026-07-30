"""LEGAL point source + online foveated map at eval — the serving glue (FOVEATED_MEMORY_SPEC_v1).

Replaces OraclePointWrapper's sim-state reads with the trained affordance head. The POINT PATH
touches NO object state: head RGB -> DINOv2-S + AffHead (2.6 cm held-out) -> 3x3 soft decode ->
ROBUST depth (5x5 near-mode median of the full-res depth at the predicted pixel; single-pixel
reads bleed onto the background at object edges) -> + learned dz (metalink sits INSIDE the
radio) -> camera frame -> BASE frame via the robot's own kinematic camera transform (the same
quantity the harness ships as cam_rel_poses) -> target_points displacements from proprio EEs.

The online FoveatedMap lives in the ODOMETRY frame: planar dead-reckoning of proprio[0:3]
(body-frame [vx, vy, wz]; drift 0.144 m/episode << L0 cell, calibrate_odometry.py). Geometry is
written EVERY step; the target channel only when conf > TAU. query() -> map_tokens, handed to
the evaluator via the _last_map_tokens attribute (patch_map_passthrough2.py attaches it
post-flatten; the serving-side B1KInputs ignores it at K=0, consumes it at K=8).

Inherits OraclePointFullRes ONLY for the full-res sensor reconfiguration + wrapper plumbing;
_inject is fully overridden so the oracle's object reads are dormant code. Stats dumped to
/root/affordance_wrapper_stats.json — VERIFY n_inject > 0 AND conf stats before believing runs.
"""

import json
import time

import numpy as np

from behavior2026_eval.oracle_point_fullres import OraclePointFullRes
from omnigibson.utils.ui_utils import create_module_logger

logger = create_module_logger(module_name=__name__)

FX, FY, CX, CY = 238.9, 315.8, 364.7, 356.2   # calibrated zed @ 720
IN, P = 518, 37
EEF_LEFT, EEF_RIGHT = slice(17, 20), slice(42, 45)
TAU = 0.5
DT = 1.0 / 30.0
IMNET_M = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_S = np.array([0.229, 0.224, 0.225], np.float32)


def _q2r(q):
    x, y, z, w = [float(v) for v in q]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _yaw_quat(yaw):
    return np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])


class AffordanceMapFullRes(OraclePointFullRes):
    def __init__(self, env):
        super().__init__(env=env)
        import sys
        sys.path.insert(0, "/root")
        import json as _json
        import torch
        from foveated_map import FoveatedMap
        from train_affordance import AffHead

        # wrist intrinsics (extract_intrinsics.py) for the near-field L2 views
        self._wk = {}
        try:
            intr = _json.load(open("/root/camera_intrinsics.json"))
            for side in ("left", "right"):
                for nm, rec in intr.items():
                    if f"{side}_realsense" in nm and isinstance(rec.get("K"), list):
                        self._wk[side] = (rec["K"][0][0], rec["K"][1][1],
                                          rec["K"][0][2], rec["K"][1][2], rec["image_width"])
        except Exception:
            pass

        self._torch = torch
        self._dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").cuda().eval()
        head = AffHead()
        head.load_state_dict(torch.load("/root/aff_out/aff_head.pt", map_location="cuda"))
        self._head = head.cuda().eval()
        self._map = FoveatedMap()
        self._odom = np.zeros(3)                 # x, y, yaw (odometry frame = episode start)
        self._last = None                        # target_points for the evaluator passthrough
        self._last_map_tokens = None
        self._stats = {"n_steps": 0, "n_inject": 0, "conf": [], "ms_aff": [], "ms_map": [],
                       "skips": {}, "skip_debug": None}
        logger.info("AffordanceMapFullRes ready (legal point source + online map)")

    # ------------------------------------------------------------------ affordance
    def _affordance(self, rgb, depth):
        """rgb (720,720,3) uint8, depth (720,720) m -> (point_cam (3,), conf) or (None, conf)."""
        torch = self._torch
        from PIL import Image
        x = np.asarray(Image.fromarray(rgb).resize((IN, IN)), np.float32) / 255.0
        x = (x - IMNET_M) / IMNET_S
        with torch.no_grad():
            t = torch.from_numpy(x.transpose(2, 0, 1))[None].cuda()
            feats = self._dino.forward_features(t)["x_norm_patchtokens"]
            heat, off, dz, conf = self._head(feats)
            prob2 = heat.softmax(-1).reshape(1, P, P)
            idx = int(heat.argmax(-1)[0])
            pu, pv = idx % P, idx // P
            us, vs, ds, ws = 0.0, 0.0, 0.0, 0.0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    qu = min(max(pu + dx, 0), P - 1)
                    qv = min(max(pv + dy, 0), P - 1)
                    w = float(prob2[0, qv, qu])
                    fr = torch.sigmoid(off[0, :, qv, qu])
                    us += (qu + float(fr[0])) * 14 * w
                    vs += (qv + float(fr[1])) * 14 * w
                    ds += float(dz[0, 0, qv, qu]) * w
                    ws += w
            u720 = us / ws * 720 / IN
            v720 = vs / ws * 720 / IN
            dzp = ds / ws
            c = float(torch.sigmoid(conf)[0])
        ui, vi = int(round(u720)), int(round(v720))
        win = depth[max(vi - 2, 0):vi + 3, max(ui - 2, 0):ui + 3].reshape(-1)
        win = win[np.isfinite(win) & (win > 0.05)]
        if not len(win):
            return None, c
        near = win[win < win.min() + 0.3]
        meas = float(np.median(near))
        z = meas + dzp
        xc = (u720 - CX) / FX * z
        yc = (v720 - CY) / FY * z
        return np.array([xc, -yc, -z]), c            # USD: -Z forward, +Y up

    # ------------------------------------------------------------------ injection
    def _inject(self, obs):
        if not isinstance(obs, dict):
            return obs
        try:
            from foveated_map import MapFrame
            t0 = time.perf_counter()
            name = self._robot.name
            node = obs.get(name)
            prop = None
            if isinstance(node, dict) and "proprio" in node:
                prop = np.asarray(node["proprio"], np.float64).reshape(-1)
            elif f"{name}::proprio" in obs:
                prop = np.asarray(obs[f"{name}::proprio"], np.float64).reshape(-1)
            if prop is None or prop.shape[0] < 45:
                self._skip("no_proprio")
                return obs

            # odometry: integrate body twist (proprio[0:3] = [vx, vy, wz], b1k robot config)
            vx, vy, wz = prop[0], prop[1], prop[2]
            self._odom[2] += wz * DT
            cy, sy = np.cos(self._odom[2]), np.sin(self._odom[2])
            self._odom[0] += (cy * vx - sy * vy) * DT
            self._odom[1] += (sy * vx + cy * vy) * DT
            base_w = np.array([self._odom[0], self._odom[1], 0.0, *(_yaw_quat(self._odom[2]))])

            # sensor images. Pre-flatten the obs nests PER SENSOR:
            #   obs[robot]["robot_r1:zed_link:Camera:0"]["rgb" | "depth_linear"]
            # (values may be torch tensors, possibly on GPU)
            def _np_of(v):
                if hasattr(v, "detach"):
                    v = v.detach().cpu().numpy()
                return np.asarray(v)

            rgb = depth = None
            wrist_imgs = {}
            if isinstance(node, dict):
                for k, v in node.items():
                    kl = k.lower()
                    if "zed" in kl:
                        if isinstance(v, dict):
                            if "rgb" in v:
                                rgb = _np_of(v["rgb"])[..., :3]
                            if "depth_linear" in v:
                                depth = _np_of(v["depth_linear"]).astype(np.float32)
                        elif kl.endswith("rgb"):
                            rgb = _np_of(v)[..., :3]
                        elif "depth" in kl:
                            depth = _np_of(v).astype(np.float32)
                    elif "realsense" in kl and isinstance(v, dict):
                        side = "left" if "left" in kl else "right"
                        wr = _np_of(v["rgb"])[..., :3] if "rgb" in v else None
                        wd = (_np_of(v["depth_linear"]).astype(np.float32)
                              if "depth_linear" in v else None)
                        if wr is not None and wd is not None:
                            wrist_imgs[side] = (wr, wd)
            if rgb is None or depth is None:
                self._skip("no_zed_rgbd", obs_keys=[str(k) for k in (node or {})][:8])
                return obs

            # kinematic camera transforms (same info the harness ships as cam_rel_poses)
            bp, bq = self._robot.get_position_orientation()
            bp = np.asarray(bp, np.float64).reshape(3)
            Rb = _q2r(np.asarray(bq).reshape(4))
            def rel(sensor):
                sp, sq = sensor.get_position_orientation()
                sp = np.asarray(sp, np.float64).reshape(3)
                Rs = _q2r(np.asarray(sq).reshape(4))
                Rrel = Rb.T @ Rs
                w = np.sqrt(max(0.0, 1 + Rrel[0, 0] + Rrel[1, 1] + Rrel[2, 2])) / 2
                q = ([(Rrel[2, 1] - Rrel[1, 2]) / (4 * w), (Rrel[0, 2] - Rrel[2, 0]) / (4 * w),
                      (Rrel[1, 0] - Rrel[0, 1]) / (4 * w), w] if w > 1e-6 else [0, 0, 0, 1])
                return np.array([*(Rb.T @ (sp - bp)), *q])

            zed = wl = wr = None
            for sname, sensor in self._robot.sensors.items():
                if "zed" in sname:
                    zed = rel(sensor)
                elif "left_realsense" in sname:
                    wl = rel(sensor)
                elif "right_realsense" in sname:
                    wr = rel(sensor)
            if zed is None:
                self._skip("no_zed_sensor")
                return obs

            def compose(base7, rel7):
                R0 = _q2r(base7[3:7])
                t = base7[:3] + R0 @ rel7[:3]
                R = R0 @ _q2r(rel7[3:7])
                w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
                q = ([(R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w),
                      (R[1, 0] - R[0, 1]) / (4 * w), w] if w > 1e-6 else [0, 0, 0, 1])
                return np.array([*t, *q])

            cam_w = compose(base_w, zed)
            wrists_w = {}
            if wl is not None:
                wrists_w["left"] = compose(base_w, wl)
            if wr is not None:
                wrists_w["right"] = compose(base_w, wr)

            # affordance -> legal point
            p_cam, conf = self._affordance(rgb.astype(np.uint8), depth)
            ms_aff = (time.perf_counter() - t0) * 1000
            t1 = time.perf_counter()

            p_base = None
            if p_cam is not None and conf > TAU:
                Rz = _q2r(zed[3:7])
                p_base = zed[:3] + Rz @ p_cam

            # online map update: 3-camera fusion (head + both wrist near-fields)
            from foveated_map import CamView
            views = [CamView(rgb.astype(np.uint8), depth, cam_w, (FX, FY, CX, CY),
                             stride=4, d_max=7.0)]
            for side, (wr, wd) in wrist_imgs.items():
                if side in wrists_w and side in self._wk:
                    kfx, kfy, kcx, kcy, knat = self._wk[side]
                    ws = wr.shape[0] / knat
                    views.append(CamView(wr.astype(np.uint8), wd, wrists_w[side],
                                         (kfx * ws, kfy * ws, kcx * ws, kcy * ws),
                                         stride=8, d_max=1.5))
            fr = MapFrame(views, base_w, wrists_w, self._stats["n_steps"])
            self._map.update(fr)
            if p_base is not None:
                Rb0 = _q2r(base_w[3:7])
                self._map.write_target(base_w[:3] + Rb0 @ p_base, conf)
            self._last_map_tokens = self._map.query()
            ms_map = (time.perf_counter() - t1) * 1000

            # target_points injection (same contract as oracle)
            if p_base is not None:
                pts = np.zeros((2, 3), np.float32)
                pts[0] = (p_base - prop[EEF_LEFT]).astype(np.float32)
                pts[1] = (p_base - prop[EEF_RIGHT]).astype(np.float32)
                obs["target_points"] = pts
                obs["target_points_mask"] = np.array([True, True], dtype=bool)
                self._last = pts
                self._stats["n_inject"] += 1
            else:
                self._last = None

            # live-map snapshots for the 3D "what the VLA's map sees" visualization
            if self._stats["n_steps"] % 100 == 0:
                try:
                    import os
                    os.makedirs("/root/map_live", exist_ok=True)
                    snap = self._map.snapshot()
                    out = {"step": self._stats["n_steps"], "odom": base_w}
                    for lv in ("l0", "l1", "l2_left", "l2_right"):
                        if lv in snap and len(snap[lv]["pts"]):
                            out[lv + "_pts"] = snap[lv]["pts"].astype(np.float16)
                            out[lv + "_rgb"] = snap[lv]["rgb"].astype(np.uint8)
                    if snap.get("target") is not None:
                        out["target"] = np.asarray(snap["target"])
                    np.savez_compressed(
                        "/root/map_live/snap_%05d.npz" % self._stats["n_steps"], **out)
                except Exception:
                    pass

            self._stats["n_steps"] += 1
            self._stats["conf"].append(round(conf, 3))
            if p_base is not None:
                self._stats.setdefault("dist_L", []).append(
                    round(float(np.linalg.norm(p_base - prop[EEF_LEFT])), 3))
                self._stats.setdefault("dist_R", []).append(
                    round(float(np.linalg.norm(p_base - prop[EEF_RIGHT])), 3))
            self._stats["ms_aff"].append(round(ms_aff, 1))
            self._stats["ms_map"].append(round(ms_map, 1))
            if self._stats["n_steps"] % 50 == 1:
                self._dump_stats()
        except Exception as e:  # never take down a rollout over conditioning
            logger.warning(f"AffordanceMapFullRes: injection failed ({e})")
        return obs

    def _skip(self, reason, **dbg):
        """No silent early returns — the oracle wrapper's history (n_injections=0 across three
        'conditioned' runs) is exactly one un-counted return away."""
        sk = self._stats["skips"]
        sk[reason] = sk.get(reason, 0) + 1
        if dbg:
            self._stats["skip_debug"] = dbg
        if sum(sk.values()) % 25 == 1:
            self._dump_stats()

    def _dump_stats(self):
        s = self._stats
        out = {"n_steps": s["n_steps"], "n_inject": s["n_inject"],
               "skips": s["skips"], "skip_debug": s["skip_debug"],
               "conf_p50": float(np.median(s["conf"])) if s["conf"] else None,
               "conf_frac_over_tau": float(np.mean(np.array(s["conf"]) > TAU)) if s["conf"] else None,
               "ms_aff_p50": float(np.median(s["ms_aff"])) if s["ms_aff"] else None,
               "ms_map_p50": float(np.median(s["ms_map"])) if s["ms_map"] else None,
               "dist_L_series": s.get("dist_L", [])[::10],
               "dist_L_min": min(s.get("dist_L", [9.9])),
               "dist_R_min": min(s.get("dist_R", [9.9]))}
        with open("/root/affordance_wrapper_stats.json", "w") as f:
            json.dump(out, f, indent=1)

    def reset(self):
        self._map.reset()
        self._odom = np.zeros(3)
        self._last = None
        self._last_map_tokens = None
        return self._inject(self.env.reset())

"""FOVEATED SPATIAL MEMORY v1 — the L0/L1/L2 map module (FOVEATED_MEMORY_SPEC_v1, PREP b).

Pure numpy, no torch: the online driver must run on CPU beside the JAX policy (<20 ms/frame
gate). One class serves both drivers — offline (training-data generation; base pose from
replay-validated pose JSONs + optional odometry-noise injection) and online (integrated
body-frame twist; drift 0.144 m/episode << L0 cell).

Levels (all writes feed-forward; no fitted latents, no refits — spec cuts):
  L0 room   0.50 m  world(odometry)-anchored dense grid, EMA occupancy + RGB + staleness +
                    target-heat. Persistence: "the radio is behind me".
  L1 mid    0.10 m  scrolling window (6.4 x 6.4 x 2.4 m) that follows the base at integer cell
                    offsets (np.roll scroll, wrapped region cleared). Approach-scale geometry.
  L2 fine   0.02 m  0.4 m cube ahead of EACH wrist camera, rebuilt from the current frame only
                    (per-frame overwrite -> no stale-glance blur during grasp), in wrist-cam
                    frame -> registration-free by construction (FK only).

Moved-object handling (demos PICK THE RADIO UP): see-through carving — occupied cells that
project inside the current frustum at depth shallower than the measured ray get their occupancy
decayed; plus the target-heat channel is written fresh every frame and EMA-decays fast.

Camera convention (validated in export_3d_viz.py / project_overlay.py): USD looks down -Z, +Y up.
Poses are 7-vectors [x y z qx qy qz qw].

Token contract — query() returns (8, 72) float32, layout frozen for Run 1:
  T0 target   : pos_base(3) conf staleness range cos/sin bearing, elevation, valid | pad
  T1 room     : L0 polar occupancy 12 bearings x 3 range bins (36) + per-sector staleness (36)
  T2 corridor : 16 samples base->target: L1 max-occ clearance (16) + occupied-height (16) | pad
  T3 EE shell : per gripper, L1 occupancy in 3 radius shells x 8 bearings (24 L + 24 R) | pad
  T4/T6 L/R geo : L2 occupancy 20^3 max-pooled 5x -> 4x4x4 (64) | pad
  T5/T7 L/R app : L2 pooled RGB 2x2x2x3 (24), nearest-occupied offset (3), frac-occ (1),
                  target-in-cube flag + local pos (4) | pad
"""

import numpy as np


def q2r(q):
    x, y, z, w = [float(v) for v in q]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class CamView:
    """One posed RGB-D view. Any camera; N-camera fusion is just a list of these."""

    def __init__(self, rgb, depth, cam_pose_world, intrinsics, stride=4, d_max=7.0):
        self.rgb = rgb                      # (H,W,3) uint8
        self.depth = depth                  # (H,W) float32 meters
        self.cam_pose_world = cam_pose_world
        self.intrinsics = intrinsics        # (fx, fy, cx, cy) at THIS resolution
        self.stride = stride
        self.d_max = d_max


class MapFrame:
    """One time step: a LIST of posed views (head + wrists + future cams) + robot poses.

    views[0] is the PRIMARY (head) view — it drives see-through carving. Wrist views are
    near-field sources (tight d_max): they contribute exactly where the head is occluded by
    the robot's own arms — the contact-range picture."""

    def __init__(self, views, base_pose_world, wrist_poses_world, frame_idx):
        self.views = views                  # list[CamView]
        self.base_pose_world = base_pose_world
        self.wrist_poses_world = wrist_poses_world  # {"left": (7,), "right": (7,)}
        self.frame_idx = frame_idx


class FoveatedMap:
    L0_RES, L0_HALF_XY, L0_ZMAX = 0.5, 8.0, 3.0        # 32 x 32 x 6 cells
    L1_RES, L1_HALF_XY, L1_ZMAX = 0.1, 3.2, 2.4        # 64 x 64 x 24 cells
    L2_RES, L2_CUBE, L2_AHEAD = 0.02, 0.4, 0.25        # 20^3, cube center 0.25 m down -Z
    STRIDE = 4                                          # depth-ray subsample
    D_MAX = 7.0
    EMA = 0.3
    CARVE = 0.4                                         # see-through occupancy multiplier
    THEAT_DECAY = 0.7                                   # per-frame target-heat retention
    TOK_D = 72

    def __init__(self):
        self.reset()

    def reset(self):
        n0xy = int(2 * self.L0_HALF_XY / self.L0_RES)
        n0z = int(self.L0_ZMAX / self.L0_RES)
        self._l0_shape = (n0xy, n0xy, n0z)
        self.l0_occ = np.zeros(self._l0_shape, np.float32)
        self.l0_rgb = np.zeros((*self._l0_shape, 3), np.float32)
        self.l0_seen = np.full(self._l0_shape, -1, np.int32)
        self.l0_theat = np.zeros(self._l0_shape, np.float32)
        self.l0_origin = None                            # set on first update (world xy of cell 0)

        n1xy = int(2 * self.L1_HALF_XY / self.L1_RES)
        n1z = int(self.L1_ZMAX / self.L1_RES)
        self._l1_shape = (n1xy, n1xy, n1z)
        self.l1_occ = np.zeros(self._l1_shape, np.float32)
        self.l1_rgb = np.zeros((*self._l1_shape, 3), np.float32)
        self.l1_seen = np.full(self._l1_shape, -1, np.int32)
        self.l1_theat = np.zeros(self._l1_shape, np.float32)
        self.l1_origin_cell = None                       # integer world-cell of window corner

        self.l2 = {}                                     # per gripper: dict of arrays
        self.target_world = None
        self.target_conf = 0.0
        self.target_seen = -1
        self.t = -1
        self._frame = None

    # ---------------------------------------------------------------- write path
    def update(self, fr: MapFrame):
        import time as _time
        self.prof = {}
        tick = _time.perf_counter
        self.t = fr.frame_idx
        self._frame = fr
        if self.l0_origin is None:
            bp = fr.base_pose_world[:3]
            self.l0_origin = np.array([bp[0] - self.L0_HALF_XY, bp[1] - self.L0_HALF_XY, 0.0])

        t0 = tick()
        P, C, D = [], [], []
        for view in fr.views:
            p_, c_, d_ = self._unproject(view)
            P.append(p_)
            C.append(c_)
            D.append(d_)
        pts = np.concatenate(P)
        cols = np.concatenate(C)
        d_cam = np.concatenate(D)
        self.prof["unproject"] = tick() - t0
        t0 = tick()
        self._scroll_l1(fr.base_pose_world[:3])
        if self.t % 3 == 0:                              # L0 is slow-changing at 0.5 m
            self._write_level(pts, cols, "l0", self.l0_origin, self.L0_RES, self._l0_shape)
        l1_origin = self._l1_origin_world()
        self._write_level(pts, cols, "l1", l1_origin, self.L1_RES, self._l1_shape)
        self.prof["write"] = tick() - t0
        t0 = tick()
        if self.t % 2 == 0:
            self._carve(fr.views[0])           # head view drives carving
        self.prof["carve"] = tick() - t0
        t0 = tick()
        close = d_cam < 1.5
        self._rebuild_l2(pts[close], cols[close], fr)
        self.prof["l2"] = tick() - t0
        self._l0_cells = np.argwhere(self.l0_occ > 0.15)
        self._l1_cells = np.argwhere(self.l1_occ > 0.15)
        self.l0_theat *= self.THEAT_DECAY
        self.l1_theat *= self.THEAT_DECAY

    def write_target(self, pos_world, conf=1.0):
        """Splat target evidence (label offline / affordance head online)."""
        self.target_world = np.asarray(pos_world, np.float64)
        self.target_conf = float(conf)
        self.target_seen = self.t
        for occ, theat, origin, res, shape in (
                (self.l0_occ, self.l0_theat, self.l0_origin, self.L0_RES, self._l0_shape),
                (self.l1_occ, self.l1_theat, self._l1_origin_world(), self.L1_RES, self._l1_shape)):
            if origin is None:
                continue
            c = ((self.target_world - origin) / res).astype(int)
            if all(0 <= c[i] < shape[i] for i in range(3)):
                theat[tuple(c)] = max(theat[tuple(c)], conf)

    _GRID_CACHE = {}

    def _unproject(self, view):
        fx, fy, cx, cy = view.intrinsics
        st = view.stride
        d = view.depth[::st, ::st]
        H, W = d.shape
        key = (H, W, st)
        if key not in self._GRID_CACHE:
            vv, uu = np.mgrid[:H, :W]
            self._GRID_CACHE[key] = (uu * st, vv * st)
        u, v = self._GRID_CACHE[key]
        ok = np.isfinite(d) & (d > 0.05) & (d < view.d_max)
        d, u, v = d[ok], u[ok], v[ok]
        xc = (u - cx) / fx * d
        yc = (v - cy) / fy * d
        p_cam = np.stack([xc, -yc, -d], 1)               # USD: -Z forward, +Y up
        cam_R = q2r(view.cam_pose_world[3:7])
        pts = p_cam @ cam_R.T + view.cam_pose_world[:3]
        cols = view.rgb[::st, ::st].reshape(-1, 3)[ok.ravel()].astype(np.float32)
        return pts, cols, d

    def _write_level(self, pts, cols, lvl, origin, res, shape):
        occ = getattr(self, f"{lvl}_occ")
        rgb = getattr(self, f"{lvl}_rgb")
        seen = getattr(self, f"{lvl}_seen")
        idx = ((pts - origin) / res).astype(np.int32)
        ok = np.all((idx >= 0) & (idx < np.array(shape)), axis=1)
        idx = idx[ok]
        if not len(idx):
            return
        flat = np.ravel_multi_index(idx.T, shape)
        uf, inv = np.unique(flat, return_inverse=True)
        cnt = np.bincount(inv).astype(np.float32)
        c = cols[ok]
        cmean = np.stack([np.bincount(inv, weights=c[:, i]) for i in range(3)], 1) / cnt[:, None]
        of, rf, sf = occ.ravel(), rgb.reshape(-1, 3), seen.ravel()
        of[uf] = (1 - self.EMA) * of[uf] + self.EMA
        rf[uf] = (1 - self.EMA) * rf[uf] + self.EMA * cmean
        sf[uf] = self.t

    def _carve(self, view):
        """Decay occupancy of cells we can currently see THROUGH (object moved away)."""
        fx, fy, cx, cy = view.intrinsics
        cam_R = q2r(view.cam_pose_world[3:7])
        cam_t = view.cam_pose_world[:3]
        H, W = view.depth.shape
        for lvl, origin, res, shape in (("l0", self.l0_origin, self.L0_RES, self._l0_shape),
                                        ("l1", self._l1_origin_world(), self.L1_RES, self._l1_shape)):
            occ = getattr(self, f"{lvl}_occ")
            cells = np.argwhere(occ > 0.15)
            if not len(cells):
                continue
            centers = origin + (cells + 0.5) * res
            q = (centers - cam_t) @ cam_R                # world -> cam
            xc, yc, d = q[:, 0], -q[:, 1], -q[:, 2]
            ok = d > 0.05
            u = np.zeros_like(d)
            v = np.zeros_like(d)
            u[ok] = xc[ok] / d[ok] * fx + cx
            v[ok] = yc[ok] / d[ok] * fy + cy
            ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if not ok.any():
                continue
            meas = view.depth[v[ok].astype(int), u[ok].astype(int)]
            through = np.isfinite(meas) & (d[ok] < meas - 2 * res)
            sel = cells[ok][through]
            occ[sel[:, 0], sel[:, 1], sel[:, 2]] *= self.CARVE

    def _rebuild_l2(self, pts, cols, fr):
        self.l2 = {}
        n = int(self.L2_CUBE / self.L2_RES)
        for side, wp in fr.wrist_poses_world.items():
            R = q2r(wp[3:7])
            # coarse manhattan cull before the rotate: cube fits in a 0.7 m ball around the wrist
            near = np.abs(pts - wp[:3]).max(1) < 0.7
            pn, cn = pts[near], cols[near]
            q = (pn - wp[:3]) @ R                        # world -> wrist-cam frame
            half = self.L2_CUBE / 2
            sel = (np.abs(q[:, 0]) < half) & (np.abs(q[:, 1]) < half) & \
                  (q[:, 2] < -(self.L2_AHEAD - half)) & (q[:, 2] > -(self.L2_AHEAD + half))
            qs = q[sel]
            occ = np.zeros((n, n, n), np.float32)
            rgb = np.zeros((n, n, n, 3), np.float32)
            if len(qs):
                origin = np.array([-half, -half, -(self.L2_AHEAD + half)])
                idx = ((qs - origin) / self.L2_RES).astype(np.int32).clip(0, n - 1)
                flat = np.ravel_multi_index(idx.T, (n, n, n))
                cnt = np.bincount(flat, minlength=occ.size).astype(np.float32)
                hit = cnt > 0
                csum = np.zeros((occ.size, 3), np.float32)
                np.add.at(csum, flat, cn[sel])
                occ.ravel()[hit] = 1.0
                rgb.reshape(-1, 3)[hit] = csum[hit] / cnt[hit, None]
            tl = None
            if self.target_world is not None:
                tw = (self.target_world - wp[:3]) @ R
                if np.abs(tw[0]) < half and np.abs(tw[1]) < half and \
                        -(self.L2_AHEAD + half) < tw[2] < -(self.L2_AHEAD - half):
                    tl = tw
            self.l2[side] = {"occ": occ, "rgb": rgb, "target_local": tl, "pose_world": wp}

    # ---------------------------------------------------------------- read path
    def _l1_origin_world(self):
        if self.l1_origin_cell is None:
            return None
        return np.array([self.l1_origin_cell[0] * self.L1_RES,
                         self.l1_origin_cell[1] * self.L1_RES, 0.0])

    def _scroll_l1(self, base_pos):
        tgt = np.array([int(np.floor((base_pos[0] - self.L1_HALF_XY) / self.L1_RES)),
                        int(np.floor((base_pos[1] - self.L1_HALF_XY) / self.L1_RES))])
        if self.l1_origin_cell is None:
            self.l1_origin_cell = tgt
            return
        shift = tgt - self.l1_origin_cell
        if not shift.any():
            return
        for name in ("l1_occ", "l1_rgb", "l1_seen", "l1_theat"):
            a = getattr(self, name)
            a[:] = np.roll(a, (-shift[0], -shift[1]), axis=(0, 1))
            for ax, s in enumerate(shift):
                if s == 0:
                    continue
                sl = [slice(None)] * a.ndim
                sl[ax] = slice(-s, None) if s > 0 else slice(None, -s)
                a[tuple(sl)] = -1 if name == "l1_seen" else 0
        self.l1_origin_cell = tgt

    def query(self, blind=False):
        """blind=True: suppress all target-derived fields (anti-shortcut token stream) —
        geometry channels are identical, so one map instance serves both streams."""
        toks = np.zeros((8, self.TOK_D), np.float32)
        fr = self._frame
        base_t = fr.base_pose_world[:3]
        base_R = q2r(fr.base_pose_world[3:7])

        # T0 target
        if self.target_world is not None and not blind:
            tb = base_R.T @ (self.target_world - base_t)
            rng = float(np.linalg.norm(tb))
            toks[0, :10] = [tb[0], tb[1], tb[2], self.target_conf,
                            min((self.t - self.target_seen) / 30.0, 10.0), rng,
                            tb[0] / max(rng, 1e-6), tb[1] / max(rng, 1e-6),
                            tb[2] / max(rng, 1e-6), 1.0]

        # T1 room polar (L0)
        cells = self._l0_cells
        if len(cells):
            centers = self.l0_origin + (cells + 0.5) * self.L0_RES
            rel = (centers - base_t) @ base_R
            ang = np.arctan2(rel[:, 1], rel[:, 0])
            rng = np.linalg.norm(rel[:, :2], axis=1)
            bi = ((ang + np.pi) / (2 * np.pi) * 12).astype(int).clip(0, 11)
            ri = np.digitize(rng, [1.5, 3.0]).clip(0, 2)
            occv = self.l0_occ[cells[:, 0], cells[:, 1], cells[:, 2]]
            stale = ((self.t - self.l0_seen[cells[:, 0], cells[:, 1], cells[:, 2]]) / 30.0)
            k = bi * 3 + ri
            np.maximum.at(toks[1], k, occv)
            np.maximum.at(toks[1], 36 + k, np.minimum(stale, 10.0).astype(np.float32))

        # T2 corridor (L1, base -> target)
        if self.target_world is not None and not blind and self.l1_origin_cell is not None:
            o1 = self._l1_origin_world()
            for i, f in enumerate(np.linspace(0.1, 1.0, 16)):
                p = base_t + f * (self.target_world - base_t)
                c = ((p - o1) / self.L1_RES).astype(int)
                if all(0 <= c[j] < self._l1_shape[j] for j in range(2)):
                    col = self.l1_occ[c[0], c[1], :]
                    toks[2, i] = col.max()
                    toks[2, 16 + i] = (np.argmax(col) * self.L1_RES) if col.max() > 0.15 else 0.0

        # T3 EE shells (L1)
        o1 = self._l1_origin_world()
        for gi, side in enumerate(("left", "right")):
            wp = fr.wrist_poses_world.get(side)
            if wp is None or o1 is None:
                continue
            cells = self._l1_cells
            if not len(cells):
                continue
            centers = o1 + (cells + 0.5) * self.L1_RES
            rel = centers - wp[:3]
            rng = np.linalg.norm(rel, axis=1)
            ang = np.arctan2(rel[:, 1], rel[:, 0])
            si = np.digitize(rng, [0.3, 0.6, 1.0])
            bi = ((ang + np.pi) / (2 * np.pi) * 8).astype(int).clip(0, 7)
            occv = self.l1_occ[cells[:, 0], cells[:, 1], cells[:, 2]]
            m = si < 3
            np.maximum.at(toks[3], gi * 24 + si[m] * 8 + bi[m], occv[m])

        # T4-T7 wrist cubes (L2)
        for gi, side in enumerate(("left", "right")):
            g = self.l2.get(side)
            if g is None:
                continue
            n = g["occ"].shape[0]
            p = n // 4
            pooled = g["occ"].reshape(4, p, 4, p, 4, p).max((1, 3, 5))
            toks[4 + 2 * gi, :64] = pooled.ravel()
            rp = n // 2
            rgbp = g["rgb"].reshape(2, rp, 2, rp, 2, rp, 3).mean((1, 3, 5)) / 255.0
            toks[5 + 2 * gi, :24] = rgbp.ravel()
            occ_cells = np.argwhere(g["occ"] > 0)
            if len(occ_cells):
                half = self.L2_CUBE / 2
                origin = np.array([-half, -half, -(self.L2_AHEAD + half)])
                centers = origin + (occ_cells + 0.5) * self.L2_RES
                near = centers[np.argmin(np.linalg.norm(centers, axis=1))]
                toks[5 + 2 * gi, 24:27] = near
                toks[5 + 2 * gi, 27] = len(occ_cells) / g["occ"].size
            if g["target_local"] is not None and not blind:
                toks[5 + 2 * gi, 28] = 1.0
                toks[5 + 2 * gi, 29:32] = g["target_local"]
        return toks

    @staticmethod
    def blind_view(toks):
        """Target-blind token stream from a full query: zero every target-derived field.
        Geometry channels (T1, T3, T4/T6, T5/T7 geometry part) are identical by construction."""
        tb = toks.copy()
        tb[0] = 0.0                # T0 target
        tb[2] = 0.0                # T2 corridor
        tb[5, 28:32] = 0.0         # L2 left target flag+pos
        tb[7, 28:32] = 0.0         # L2 right
        return tb

    def snapshot(self):
        """Occupied-cell clouds per level for visualization."""
        out = {}
        c0 = np.argwhere(self.l0_occ > 0.15)
        out["l0"] = {"pts": self.l0_origin + (c0 + 0.5) * self.L0_RES,
                     "rgb": self.l0_rgb[c0[:, 0], c0[:, 1], c0[:, 2]]}
        o1 = self._l1_origin_world()
        c1 = np.argwhere(self.l1_occ > 0.15)
        out["l1"] = {"pts": o1 + (c1 + 0.5) * self.L1_RES,
                     "rgb": self.l1_rgb[c1[:, 0], c1[:, 1], c1[:, 2]]}
        for side, g in self.l2.items():
            c2 = np.argwhere(g["occ"] > 0)
            half = self.L2_CUBE / 2
            origin = np.array([-half, -half, -(self.L2_AHEAD + half)])
            R = q2r(g["pose_world"][3:7])
            pts = (origin + (c2 + 0.5) * self.L2_RES) @ R.T + g["pose_world"][:3]
            out[f"l2_{side}"] = {"pts": pts, "rgb": g["rgb"][c2[:, 0], c2[:, 1], c2[:, 2]]}
        out["target"] = self.target_world
        return out

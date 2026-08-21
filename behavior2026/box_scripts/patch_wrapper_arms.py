"""Campaign instrumentation patch for AffordanceMapFullRes (protocol Amendment 1).

Adds, env-var controlled so ONE wrapper serves every arm:
  MAP_ARM = B (default) | A (all-zero tokens: presence preserved, content nulled)
          | B0 (blind_view: geometry kept, target fields zeroed)
  (arm C = ScaffoldCollectWrapper, which inherits this and gets the same switches)

  I1: per-step wrist off-axis angles (angle between each wrist camera's optical axis (-Z) and
      the wrist->target ray) -> stats series "wrist_angL/R" (deg). The G1 wrist metric.
  I4: map-fidelity diagnostics — GROUND-TRUTH radio position read via the inherited oracle
      target resolver, LOGGED ONLY (never fed to policy): per-step map-T0-vs-truth error and
      affordance-point-vs-truth error -> "map_err"/"aff_err" series.
"""

import py_compile

P = "/root/behavior2026_eval/affordance_map_fullres.py"
s = open(P).read()
if "MAP_ARM" in s:
    raise SystemExit("already patched")

a = "TAU = 0.5\n"
assert a in s
s = s.replace(a, a + 'import os as _os\nMAP_ARM = _os.environ.get("MAP_ARM", "B").upper()\n', 1)

# arm switch at token attach + I1/I4 instrumentation, inserted before stats increment
a = """            self._last_map_tokens = self._map.query()
            ms_map = (time.perf_counter() - t1) * 1000"""
b = """            _toks = self._map.query()
            if MAP_ARM == "A":
                _toks = np.zeros_like(_toks)
            elif MAP_ARM == "B0":
                from foveated_map import FoveatedMap as _FM
                _toks = _FM.blind_view(_toks)
            self._last_map_tokens = _toks
            ms_map = (time.perf_counter() - t1) * 1000

            # I1: wrist off-axis angles (deg) vs the AFFORDANCE target (legal signal)
            try:
                if p_base is not None:
                    for _side, _key in (("left", "wrist_angL"), ("right", "wrist_angR")):
                        _wp = {"left": wl, "right": wr}.get(_side)
                        if _wp is None:
                            continue
                        _R = _q2r(_wp[3:7])
                        _axis = -_R[:, 2]
                        _ray = p_base - _wp[:3]
                        _n = np.linalg.norm(_ray)
                        if _n > 1e-6:
                            _ang = float(np.degrees(np.arccos(
                                np.clip(_axis @ (_ray / _n), -1, 1))))
                            self._stats.setdefault(_key, []).append(round(_ang, 1))
            except Exception:
                pass

            # I4: map fidelity vs sim GROUND TRUTH — diagnostics only, never fed to policy
            try:
                if self._targets:
                    _tp = np.asarray(self._targets[0].get_position_orientation()[0],
                                     dtype=np.float64).reshape(3)
                    _bp, _bq = self._robot.get_position_orientation()
                    _truth_base = _q2r(np.asarray(_bq).reshape(4)).T @ (
                        _tp - np.asarray(_bp, np.float64).reshape(3))
                    if p_base is not None:
                        self._stats.setdefault("aff_err", []).append(
                            round(float(np.linalg.norm(p_base - _truth_base)), 3))
                    _t0 = self._last_map_tokens[0]
                    if _t0[9] > 0.5:  # T0 valid flag
                        self._stats.setdefault("map_err", []).append(
                            round(float(np.linalg.norm(_t0[:3] - _truth_base)), 3))
            except Exception:
                pass"""
assert a in s, "token-attach anchor not found"
s = s.replace(a, b, 1)

# widen the stats dump with the new series (keep last values + medians)
a = """               "dist_L_series": s.get("dist_L", [])[::10],"""
b = """               "arm": MAP_ARM,
               "wrist_angL_p50": float(np.median(s["wrist_angL"])) if s.get("wrist_angL") else None,
               "wrist_angR_p50": float(np.median(s["wrist_angR"])) if s.get("wrist_angR") else None,
               "aff_err_p50": float(np.median(s["aff_err"])) if s.get("aff_err") else None,
               "map_err_p50": float(np.median(s["map_err"])) if s.get("map_err") else None,
               "wrist_angL_series": s.get("wrist_angL", [])[::10],
               "map_err_series": s.get("map_err", [])[::10],
               "dist_L_series": s.get("dist_L", [])[::10],"""
assert a in s, "stats dump anchor"
s = s.replace(a, b, 1)

compile(s, P, "exec")
open(P, "w").write(s)
py_compile.compile(P, doraise=True)
print("wrapper patched: MAP_ARM switch + I1 wrist angles + I4 map fidelity")

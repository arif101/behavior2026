"""v2 relabel (2026-09-16, PRESS_FIX_SPEC F2 + granular targets): adds THREE NEW columns to a LeRobot-v3 root and leaves
`stage` / `target_points` untouched (A4/S1 stay byte-identical; the press config selects the v2 columns).

  stage_v2   int32  0 APPROACH  1 GRASP  2 TRANSPORT  3 PRESS      (crisp, monotone within an episode)
  progress   f32    (stage_v2 + within-stage fraction) / 4          (piecewise-linear task clock, no demo timeline needed)
  target_points_v2 f32[6]  stage-indexed target: before lift = RAIL grasp point, after lift = toggle button (= target_points)

Per-frame signals (every source has them): target_points = [button - EE_L, button - EE_R] in the base frame (6);
lifted = manufactured: converter stage == 2 (post-lift) | map: button world z (metalink_labels) rises > 3 cm over frame 0.
Rail grasp point = button + (0, 0, +0.141) in the base frame. Derivation: human grasps (19 strict clips) sit at
(-0.013, y, 0.129) in the radio frame with y free along the rail (std 0.065), the button metalink at (0.045, 0.042, -0.012);
the body-frame rail-minus-button vector (-0.058, -0.042, 0.141) has a frame-independent vertical part (radio upright, base
level) and a 7 cm horizontal part that needs the radio yaw, which the eval wrapper cannot know -> the SAME yaw-free target is
used in training and at eval (consistency beats a 7 cm gain the policy could not use). The rail is +-10 cm long anyway.
Rules: holding hand = the hand nearest the button at the first lifted frame; GRASP = not lifted and the holding hand within
--grasp-radius of the rail point; PRESS = lifted and the FREE hand within --press-radius of the button (v1 END frames while
lifted count as press too); classes are made monotone with a cumulative max.
  python relabel_v2.py --root /root/manufactured/b1k_radio_approach_v2
  python relabel_v2.py --root /root/b1k_radio_map --episode-map /root/backup/keys/episode_map.json --metalink-dir /root/backup/metalink_labels
"""
import argparse, glob, json, pathlib
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

RAIL_UP = np.array([0.0, 0.0, 0.141], np.float32)

def relabel_episode(tp, stage_v1, lifted, grasp_r, press_r, reach_r=0.25, toggle_idx=None, press_anchor='closest'):
    n = len(tp); dL = np.linalg.norm(tp[:, 0:3] + RAIL_UP, axis=1); dR = np.linalg.norm(tp[:, 3:6] + RAIL_UP, axis=1)
    bL = np.linalg.norm(tp[:, 0:3], axis=1); bR = np.linalg.norm(tp[:, 3:6], axis=1)
    lift_idx = int(np.argmax(lifted)) if lifted.any() else None
    ref = lift_idx if lift_idx is not None else n - 1
    hold_R = bR[ref] <= bL[ref]                       # holding hand = nearest the button when the lift starts
    d_hold = dR if hold_R else dL; b_free = bL if hold_R else bR
    raw = np.zeros(n, np.int32)
    raw[(~lifted) & (d_hold <= grasp_r)] = 1
    raw[lifted] = 2
    # PRESS = the free hand's FINAL reach to the button. Anchor = the TOGGLE frame when known (human demos: the recorded
    # reward flips to 1 there; matches the bank press frames within 5 frames), else the closest approach within the
    # last 40% of the episode (policy-press episodes), else none (factory/approach clips end 60 frames before the press).
    # Start = the last lifted frame before the anchor where the free hand was still > reach_r away (humans hover the free
    # hand near the carried radio early in the transport, so a plain radius test fires ~160 frames early). Everything
    # from the reach on is press, incl. the post-toggle tail (the human keeps recording ~25-30% of the demo after success).
    press_start = None
    if lift_idx is not None:
        anchor = None
        if toggle_idx is not None:
            anchor = int(toggle_idx)
        elif press_anchor == "closest":
            lo = int(0.6 * n); cand = np.flatnonzero(lifted[lo:]) + lo
            if len(cand): tm = int(cand[np.argmin(b_free[cand])]); anchor = tm if b_free[tm] <= press_r else None
        if anchor is not None:
            far = np.flatnonzero((b_free[:anchor] > reach_r) & lifted[:anchor])
            press_start = int(far[-1]) + 1 if len(far) else max(lift_idx, anchor - 60)
            raw[press_start:] = 3
    st = np.maximum.accumulate(raw)                    # monotone 0 -> 1 -> 2 -> 3
    prog = np.zeros(n, np.float32)
    for c in range(4):
        idx = np.flatnonzero(st == c)
        if len(idx): prog[idx] = (c + (np.arange(len(idx)) + 0.5) / len(idx)) / 4.0
    tp2 = tp.copy()
    pre = st <= 1
    tp2[pre, 0:3] += RAIL_UP; tp2[pre, 3:6] += RAIL_UP
    return st, prog, tp2, ("R" if hold_R else "L"), lift_idx, press_start

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--episode-map"); ap.add_argument("--metalink-dir"); ap.add_argument("--rawdemos", help="dir with episode_XXXXXXXX.hdf5 (toggle frame from the recorded reward)")
    ap.add_argument("--press-anchor", choices=("closest", "none"), default="closest", help="press anchor when no toggle frame: closest free-hand approach in the last 40%% (policy-press episodes) or none (clips that end before the press)")
    ap.add_argument("--grasp-radius", type=float, default=0.12); ap.add_argument("--press-radius", type=float, default=0.12)
    ap.add_argument("--lift-dz", type=float, default=0.03); ap.add_argument("--reach-radius", type=float, default=0.25); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(); root = pathlib.Path(a.root)
    emap = json.load(open(a.episode_map))["mapping"] if a.episode_map else None
    files = sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True)); assert files
    counts = np.zeros(4, np.int64); n_eps = 0; n_press = 0; n_lift = 0; hands = {"L": 0, "R": 0}; nonmono = 0; ep_report = []; n_tog = 0; tog_minus_lift = []; tog_minus_ps = []; n_toggled_frames = 0
    for f in files:
        t = pq.read_table(f); cols = t.schema.names
        ep = np.asarray(t.column("episode_index").to_pylist()); tp_all = np.asarray(t.column("target_points").to_pylist(), np.float32).reshape(len(t), 6)
        st1_all = np.asarray(t.column("stage").to_pylist()).reshape(len(t), -1)[:, 0].astype(int)
        st2 = np.zeros(len(t), np.int32); pr = np.zeros(len(t), np.float32); tp2 = tp_all.copy(); tog_all = np.zeros(len(t), np.int32)
        for e in np.unique(ep):
            idx = np.flatnonzero(ep == e); tp = tp_all[idx]; st1 = st1_all[idx]
            if emap is not None:
                lab = np.load(pathlib.Path(a.metalink_dir) / f"ep{emap[str(int(e))]}.npz")["meta_world"]
                assert len(lab) in (len(idx), len(idx) + 1), f"ep {e}: labels {len(lab)} vs rows {len(idx)}"   # labels cover states (N+1), rows cover actions (N)
                lab = lab[:len(idx)]
                lifted = (lab[:, 2] - lab[0, 2]) > a.lift_dz
                lifted = np.maximum.accumulate(lifted)      # once lifted, stays "lifted" for staging purposes
            else:
                lifted = st1 >= 2
            tog = None
            if a.rawdemos:
                import h5py
                with h5py.File(pathlib.Path(a.rawdemos) / f"episode_{int(emap[str(int(e))]):08d}.hdf5", "r") as hf:
                    rw = hf["data/demo_0/reward"][:]
                hit = np.flatnonzero(rw >= 0.5); tog = int(hit[0]) if len(hit) else None
            s, p, t2, hand, li, ps = relabel_episode(tp, st1, lifted, a.grasp_radius, a.press_radius, a.reach_radius, toggle_idx=tog, press_anchor=a.press_anchor)
            tg = np.zeros(len(idx), np.int32)
            if tog is not None: tg[tog:] = 1
            tog_all[idx] = tg; n_tog += int(tog is not None)
            if tog is not None and li is not None: tog_minus_lift.append(tog - li); tog_minus_ps.append(tog - ps if ps is not None else None)
            st2[idx] = s; pr[idx] = p; tp2[idx] = t2
            counts += np.bincount(s, minlength=4); n_eps += 1; n_press += int((s == 3).any()); n_lift += int(li is not None); hands[hand] += 1
            nonmono += int((np.diff(s) < 0).any())
            ep_report.append((int(e), hand, li, ps, [int((s == c).sum()) for c in range(4)]))
        if not a.dry_run:
            for name in ("stage_v2", "progress", "target_points_v2", "toggled"):
                if name in cols: t = t.drop([name])
            t = t.append_column("stage_v2", pa.array(st2, type=pa.int32()))            # scalar, like the existing `stage`
            t = t.append_column("progress", pa.array(pr, type=pa.float32()))
            t = t.append_column("target_points_v2", pa.array(tp2.tolist(), type=pa.list_(pa.float32(), 6)))
            t = t.append_column("toggled", pa.array(tog_all, type=pa.int32()))
            pq.write_table(t, f)
        n_toggled_frames += int(tog_all.sum())
    tot = counts.sum()
    print(f"RELABEL_V2 {root.name}: {n_eps} eps, {tot} frames | stage_v2 counts approach {counts[0]} ({100*counts[0]/tot:.1f}%) grasp {counts[1]} ({100*counts[1]/tot:.1f}%) "
          f"transport {counts[2]} ({100*counts[2]/tot:.1f}%) press {counts[3]} ({100*counts[3]/tot:.1f}%) | eps with lift {n_lift}, with press {n_press}, with toggle {n_tog} | post-toggle frames {n_toggled_frames} | holding hand {hands} | non-monotone eps {nonmono}")
    if tog_minus_lift:
        q = lambda x: np.percentile(x, [5, 50, 95]).round(0).tolist()
        print(f"   toggle - lift frames p5/50/95 {q(tog_minus_lift)} | toggle - press_start p5/50/95 {q([v for v in tog_minus_ps if v is not None])}")
    for r in ep_report[:6]: print("   ep (idx, hand, lift_frame, press_start, counts)", r)
    if not a.dry_run:
        ip = root / "meta" / "info.json"; info = json.loads(ip.read_text())
        info["features"]["stage_v2"] = {"dtype": "int32", "shape": [1], "names": None}
        info["features"]["progress"] = {"dtype": "float32", "shape": [1], "names": None}
        info["features"]["target_points_v2"] = {"dtype": "float32", "shape": [6], "names": None}
        info["features"]["toggled"] = {"dtype": "int32", "shape": [1], "names": None}
        ip.write_text(json.dumps(info, indent=4)); print(f"registered stage_v2/progress/target_points_v2/toggled in {ip}")

if __name__ == "__main__":
    main()

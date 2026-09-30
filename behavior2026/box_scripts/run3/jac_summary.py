"""Corrective-field probe summary (same definitions as jac_analyze.py, parametrized by output dir).
restoring = -((disp - disp_base_of_state) . off_dir): + = the 16-step displacement pushes back against the imposed offset.
Hand-offset conditions y/z +-3/6 cm; vision-only conditions vis_x/vis_y +-3 (radio moved, proprio unchanged); yaw +-10 deg.
  python jac_summary.py /root/jacobian_probe_4dall_off [/root/jacobian_probe_4dpd_off ...]"""
import sys, json, glob, collections, numpy as np
for D in sys.argv[1:]:
    rows = [json.loads(l) for f in sorted(glob.glob(f"{D}/near_tr*.jsonl")) for l in open(f)]
    print(f"\n=== {D}: {len(rows)} trials, states {dict(collections.Counter(r['tag'] for r in rows))}, tracebacks n/a, grasps {sum(bool(r.get('grasp')) for r in rows)}")
    hand, vis, mags, per_state = [], [], [], {}
    for tag in sorted(set(r["tag"] for r in rows)):
        rs = [r for r in rows if r["tag"] == tag]; by = collections.defaultdict(list)
        for r in rs: by[r["cond"]].append(r)
        if not by.get("base"): print(f"{tag}: no base"); continue
        db = np.mean([np.array(r["disp"]) for r in by["base"]], axis=0)
        line = []
        for c in sorted(by):
            if c == "base" or by[c][0].get("off_dir") is None: continue
            v = [float(-((np.array(r["disp"]) - db) @ np.array(r["off_dir"]))) for r in by[c]]
            line.append(f"{c} {np.mean(v):+.3f}")
            if c.startswith("vis"): vis += v
            elif c[0] in "yz": hand += v; mags += [abs(float(c[1:])) / 100] * len(v)
        oe = [r["oerr_end"] for r in rs if r.get("oerr_end") is not None]
        print(f"{tag}: base |disp| {np.linalg.norm(db):.3f} progress {np.mean([r['progress'] for r in by['base']]):+.3f} | " + "  ".join(line) + f" | oerr_end mean {np.mean(oe):.2f}")
    hand, vis, mags = np.array(hand), np.array(vis), np.array(mags)
    if len(hand):
        print(f"HAND-OFFSET pooled (n={len(hand)}): restoring {hand.mean():+.4f} m +- {hand.std()/np.sqrt(len(hand)):.4f} per 16 steps; "
              f"corr(restoring, offset) {np.corrcoef(hand, mags)[0,1]:.3f}; frac > 0 {np.mean(hand > 0):.2f}; by magnitude: "
              + ", ".join(f"{m*100:.0f}cm {hand[mags==m].mean():+.4f}" for m in sorted(set(mags))))
    if len(vis): print(f"VISION-ONLY pooled (n={len(vis)}): restoring {vis.mean():+.4f} m +- {vis.std()/np.sqrt(len(vis)):.4f}; frac > 0 {np.mean(vis > 0):.2f}")
    print("JAC_SUMMARY_DONE")

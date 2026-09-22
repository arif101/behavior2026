import json, glob, numpy as np, collections
rows=[json.loads(l) for f in sorted(glob.glob("/root/jacobian_probe/near_tr*_b.jsonl")) for l in open(f)]
print("trials", len(rows), "by state:", collections.Counter(r["tag"] for r in rows))
for tag in sorted(set(r["tag"] for r in rows)):
    rs=[r for r in rows if r["tag"]==tag]; by=collections.defaultdict(list)
    for r in rs: by[r["cond"]].append(r)
    base=by.get("base",[]); 
    if not base: continue
    db=np.mean([np.array(r["disp"]) for r in base],axis=0); dprog=np.mean([r["progress"] for r in base]); oe_b=np.mean([r["oerr_end"] for r in base])
    print(f"\n{tag}: dist_start {np.mean([r['dist_start'] for r in rs]):.3f} | BASE disp {np.round(db,3).tolist()} |disp| {np.linalg.norm(db):.3f} progress {dprog:+.3f} oerr_end {oe_b:.2f}")
    # position-offset conditions from the first run for the same state (hand moved): pull from the non-_b file
    try:
        prev=[json.loads(l) for l in open(f"/root/jacobian_probe/{tag[:-2]}.jsonl")]; pby=collections.defaultdict(list)
        for r in prev: pby[r["cond"]].append(r)
        pdb=np.mean([np.array(r["disp"]) for r in pby["base"]],axis=0)
        for c in ("y+3","y-3"):
            v=[float(-((np.array(r["disp"])-pdb) @ np.array(r["off_dir"]))) for r in pby[c]]
            print(f"  HAND moved {c} (run 1): restoring {np.mean(v):+.4f} (n={len(v)})  [image AND proprio change]")
    except Exception as e: print("  (no run-1 data)", e)
    for c in ("vis_y+3","vis_y-3","vis_x+3","vis_x-3"):
        v=[float(-((np.array(r["disp"])-db) @ np.array(r["off_dir"]))) for r in by.get(c,[])]
        if v: print(f"  RADIO moved {c}: restoring {np.mean(v):+.4f} +- {np.std(v)/np.sqrt(len(v)):.4f} (n={len(v)}) [image changes, proprio does not]  | mean |disp-base| {np.mean([np.linalg.norm(np.array(r['disp'])-db) for r in by[c]]):.4f}")
    for c in ("yaw+10","yaw-10"):
        rs2=by.get(c,[])
        if rs2: print(f"  WRIST {c}: oerr_start {np.mean([r['oerr_start'] for r in rs2]):.3f} -> oerr_end {np.mean([r['oerr_end'] for r in rs2]):.3f} rad (base: 0.00 -> {oe_b:.2f}); yaw restore = {np.mean([r['oerr_start']-r['oerr_end'] for r in rs2]):+.3f} rad")
    # response magnitude vs offset direction: is the vis response correlated with the offset direction at all?
    vis=[r for r in rs if r["cond"].startswith("vis_")]
    if vis:
        proj=[float(-((np.array(r["disp"])-db) @ np.array(r["off_dir"]))) for r in vis]; mag=[np.linalg.norm(np.array(r["disp"])-db) for r in vis]
        print(f"  vision-only pooled: restoring {np.mean(proj):+.4f} (n={len(vis)}), mean deviation from base {np.mean(mag):.4f} m; sample noise (base-base) {np.mean([np.linalg.norm(np.array(r['disp'])-db) for r in base]):.4f} m")

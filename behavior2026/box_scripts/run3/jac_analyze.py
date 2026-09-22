import json, glob, numpy as np, collections
rows=[json.loads(l) for f in sorted(glob.glob("/root/jacobian_probe/near_tr*.jsonl")) for l in open(f)]
print("trials", len(rows), "conds", sorted(set(r["cond"] for r in rows)), "states", sorted(set(r["tag"] for r in rows)))
by=collections.defaultdict(list)
for r in rows: by[(r["tag"], r["cond"])].append(r)
print("\nservo placement quality (dist_start = |EE - grasp target| at handover; intended 0.10): ")
for tag in sorted(set(r["tag"] for r in rows)):
    ds=[r["dist_start"] for r in rows if r["tag"]==tag]; oe=[r["oerr_start"] for r in rows if r["tag"]==tag]
    print(f"  {tag}: dist_start mean {np.mean(ds):.3f} (min {np.min(ds):.3f} max {np.max(ds):.3f}) oerr_start mean {np.mean(oe):.2f} rad; servo_ok {np.mean([r['servo_ok'] for r in rows if r['tag']==tag]):.2f}")
print("\nper state x condition: baseline-subtracted restoring displacement (m, + = pushes back against the offset), progress toward target (m), wrist error after chunk (rad)")
summary=collections.defaultdict(list)
for tag in sorted(set(r["tag"] for r in rows)):
    base=by.get((tag,"base"),[])
    if not base: continue
    db=np.mean([np.array(r["disp"]) for r in base],axis=0)
    line=[]
    for cond in ["y+3","y-3","y+6","y-6","z+3","z-3","z+6","z-6"]:
        rs=by.get((tag,cond),[])
        if not rs: continue
        rest=[float(-((np.array(r["disp"])-db) @ np.array(r["off_dir"]))) for r in rs]
        summary[cond]+=rest
        line.append(f"{cond}:{np.mean(rest):+.3f}")
    prog=np.mean([r["progress"] for r in rows if r["tag"]==tag]); oe=np.mean([r["oerr_end"] for r in rows if r["tag"]==tag and r["oerr_end"] is not None])
    print(f"  {tag}: {' '.join(line)} | progress {prog:+.3f} | oerr_end {oe:.2f} | base disp {np.round(db,3).tolist()}")
print("\nPOOLED restoring displacement by condition (mean +- sem over states x samples; offset magnitude for reference):")
for cond in ["y+3","y-3","y+6","y-6","z+3","z-3","z+6","z-6"]:
    v=np.array(summary[cond]); print(f"  {cond}: {v.mean():+.4f} +- {v.std(ddof=1)/np.sqrt(len(v)):.4f} (n={len(v)}) offset={abs(float(cond[1:]))/100:.2f}")
allr=np.concatenate([np.array(summary[c]) for c in summary]); print(f"\nALL: mean restoring {allr.mean():+.4f} m per 16 steps vs offsets 0.03-0.06 m; frac trials restoring>0: {(allr>0).mean():.2f}")
# vs. proportional: correlation of restoring with offset magnitude
mags=np.concatenate([[abs(float(c[1:]))/100]*len(summary[c]) for c in summary]); print("corr(restoring, offset magnitude):", round(float(np.corrcoef(allr,mags)[0,1]),3))
g=[r for r in rows if r.get("grasp")]; print("grasps during chunks:", len(g))

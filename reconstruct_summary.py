#!/usr/bin/env python
"""Rebuild grasp_summary_<arm>.json from per-rollout jsons (fallback if the in-run write is lost)."""
import sys, json, glob, os
import numpy as np
d = sys.argv[1]; arm = sys.argv[2] if len(sys.argv)>2 else os.path.basename(d.rstrip('/'))
rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d,"json","*.json")))]
if not rows: print("no rows in", d); sys.exit(1)
summ = {"arm": arm, "n": len(rows),
        "grasp_completed": int(sum(x.get("grasp_completed",False) for x in rows)),
        "success": int(sum(x.get("success",False) for x in rows)),
        "fingertip_min_median": round(float(np.median([x["fingertip_min_m"] for x in rows])),4),
        "fingertip_min_best": round(float(min(x["fingertip_min_m"] for x in rows)),4),
        "ee_button_min_best": round(float(min(x["ee_button_min_m"] for x in rows)),4)}
json.dump({"summary": summ, "rows": rows}, open(os.path.join(d,f"grasp_summary_{arm}.json"),"w"), indent=2)
print(json.dumps(summ, indent=2))

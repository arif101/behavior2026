import numpy as np, glob, json, collections, os
def traj(R):
    o = []
    for f in sorted(glob.glob(f"{R}/map_live/snap_*.npz")):
        z = np.load(f); o.append((int(z["step"]), z["odom"][:3].astype(float)))
    return o
RADIO = np.array([3.7303, 4.7257, 0.534])   # instance-301 radio position (world) from the harvested near-grasp states
ref = traj("/root/run3_eval/full/run_3"); ref_grasp = [p for s, p in ref if s <= 1700][-1]; ref0 = ref[0][1]
print("full/run_3 (grasp at 1727): start", np.round(ref0, 2), "base pose at ~1700", np.round(ref_grasp, 2), "base-radio dist", round(float(np.linalg.norm(ref_grasp[:2] - RADIO[:2])), 2))
for r in range(1, 8):
    t = traj(f"/root/run3_eval/4dall_final_ptroff/run_{r}"); p0 = t[0][1]; pe = t[-1][1]
    settle = [s for s, p in t if np.linalg.norm(p[:2] - pe[:2]) < 0.3][0]
    print(f"4dall run_{r}: travel {np.linalg.norm(pe[:2] - p0[:2]):.2f} m, settled by step {settle}, final base-radio dist {np.linalg.norm(pe[:2] - RADIO[:2]):.2f} m, dist to full/run_3 grasp pose {np.linalg.norm(pe[:2] - ref_grasp[:2]):.2f} m")
for R in ["full/run_1", "full_ptoff/run_1", "full_ptoff/run_2", "a4/run_1", "full_s2stage/run_1", "full_s2stage/run_2"]:
    t = traj("/root/run3_eval/" + R); pe = t[-1][1]; print(f"{R}: final base-radio dist {np.linalg.norm(pe[:2] - RADIO[:2]):.2f} m, to grasp pose {np.linalg.norm(pe[:2] - ref_grasp[:2]):.2f} m")
print("== full_s2stage (pointer ON, head-driven stage, full stack, 0/25): head votes per run")
for r in range(1, 26):
    f = f"/root/run3_eval/full_s2stage/run_{r}/stage_head_log.jsonl"
    if not os.path.exists(f): continue
    rows = [json.loads(l) for l in open(f)]; c = collections.Counter(x["served"] for x in rows); tr = collections.Counter(x["tracker"] for x in rows)
    first1 = next((x["step"] for x in rows if x["served"] >= 1), None)
    print(f"  run_{r}: served {dict(sorted(c.items()))} first served>=1 at {first1}; tracker {dict(sorted(tr.items(), key=lambda kv: str(kv[0])))}")
print("== 4dall run_1 wrapper stats keys:", list(json.load(open("/root/run3_eval/4dall_final_ptroff/run_1/affordance_wrapper_stats.json")).keys()))

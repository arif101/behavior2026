import glob, json, statistics
fs = sorted(glob.glob("/root/eval_phaseA_weighted_30000_N10/json/*.json"))
rows = [json.load(open(f)) for f in fs]
new = [r for r in rows if r["steps"] > 600]
old = [r for r in rows if r["steps"] <= 600]
print(f"result files: {len(rows)}   FULL-TIME (>600 steps): {len(new)}   stale 501-step: {len(old)}")
print()
for r in sorted(new, key=lambda x: x["instance_id"]):
    nb = r["normalized_agent_distance"]["base"]
    print(f"  inst {r['instance_id']}  success={r['success']}  q={r['q_score']['final']}  "
          f"steps={r['steps']}  norm_base={nb:.2f}")
if new:
    qs = [r["q_score"]["final"] for r in new]
    nb = [r["normalized_agent_distance"]["base"] for r in new]
    print()
    print(f"FULL-TIME:  success {sum(1 for r in new if r['success'])}/{len(new)}   "
          f"mean_q {statistics.mean(qs):.4f}   nonzero_q {sum(1 for q in qs if q>0)}")
    print(f"  norm_base median {statistics.median(nb):.2f}  (range {min(nb):.2f}-{max(nb):.2f})")
    print(f"  all hit timeout: {all(r['steps'] >= 3200 for r in new)}")
    print()
    print("COMPARISON at 501 steps (15.5% budget): 0/10, mean_q 0.0000, nonzero 0")

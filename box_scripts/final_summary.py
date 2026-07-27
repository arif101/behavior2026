import glob, json, statistics

print("=" * 74)
print("A. PROMPT FIX: arm1@30k, 10 instances, prompt now matches training")
print("=" * 74)
fs = sorted(glob.glob("/root/eval_phaseA_weighted_30000_N10/json/*.json"))
rows = [json.load(open(f)) for f in fs]
if rows:
    qs = [r["q_score"]["final"] for r in rows]
    print(f"success {sum(1 for r in rows if r['success'])}/{len(rows)}   mean_q {statistics.mean(qs):.4f}   nonzero_q {sum(1 for q in qs if q>0)}")
    print("(baseline with MISMATCHED prompt was: 0/10, mean_q 0.0000, nonzero 0)")

print()
print("=" * 74)
print("B. CONTACT A/B: commit vs ordinary frame loss (the treatment's own test)")
print("=" * 74)
print(f"{'checkpoint':<26} {'commit':>9} {'ordinary':>9} {'commit-ord':>11}")
for f in sorted(glob.glob("/root/score_*.json")):
    d = json.load(open(f))
    for name, r in d.get("strata", {}).items():
        ck = "/".join(d["ckpt"].rstrip("/").split("/")[-2:])
        print(f"{ck:<26} {r['commit']:>9.5f} {r['ordinary']:>9.5f} {r['commit_minus_ordinary']:>11.5f}")

print()
print("=" * 74)
print("C. LAYER PROBES: weighted vs uniform @10k (3000 frames each)")
print("=" * 74)
try:
    w = json.load(open("/root/probe_w10k_big_results.json"))
    u = json.load(open("/root/probe_u10k_results.json"))
    keys = sorted(set(w) & set(u))
    targets = ["gripper_cmd", "action_t", "phase_frac", "future_ee_delta"]
    print(f"{'target':<18} {'weighted':>10} {'uniform':>10} {'diff':>10}")
    for t in targets:
        wv = [w[k][t]["mean_r2"] for k in keys if t in w.get(k, {})]
        uv = [u[k][t]["mean_r2"] for k in keys if t in u.get(k, {})]
        if wv and uv:
            print(f"{t:<18} {statistics.mean(wv):>10.4f} {statistics.mean(uv):>10.4f} {statistics.mean(wv)-statistics.mean(uv):>+10.4f}")
except Exception as e:
    print("probe compare failed:", e)

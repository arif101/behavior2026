import glob, json, statistics

print("=" * 68)
print("CLOSED-LOOP: turning_on_radio, oracle points, 10 public_test instances")
print("=" * 68)
for d in sorted(glob.glob("/root/eval_*_N10")) + ["/root/eval_w10k_N10"]:
    fs = sorted(glob.glob(d + "/json/*.json"))
    if not fs:
        continue
    rows = [json.load(open(f)) for f in fs]
    qs = [r["q_score"]["final"] for r in rows]
    nb = [r["normalized_agent_distance"]["base"] for r in rows]
    succ = sum(1 for r in rows if r["success"])
    print(f"{d.split('/')[-1]:<34} n={len(rows):<3} success={succ}/{len(rows)}  "
          f"mean_q={statistics.mean(qs):.4f}  nonzero_q={sum(1 for q in qs if q>0)}  "
          f"norm_base med={statistics.median(nb):.1f}")

print()
print("=" * 68)
print("POINT-CHANNEL LIVENESS (same frames, same noise, deterministic)")
print("=" * 68)
print(f"{'checkpoint':<28} {'A_vs_C':>8} {'A_vs_B':>8}   verdict")
for f in sorted(glob.glob("/root/liveness_*.json")):
    d = json.load(open(f))
    parts = d["ckpt"].rstrip("/").split("/")
    name = f"{parts[-2]}/{parts[-1]}"
    print(f"{name:<28} {d['rel_A_vs_C']:>8.4f} {d['rel_A_vs_B']:>8.4f}   {d['verdict']}")

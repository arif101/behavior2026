"""Per-rollout motion diagnostics from the action log (commanded base velocity + arm command magnitude) and the evaluator json.
Usage: rollout_diag.py <run_dir> [<run_dir> ...]"""
import sys, json, glob, os, numpy as np
for R in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f"{R}/action_log.jsonl")]
    bt = np.array([r["base_trans_mag"] for r in rows]); by = np.array([r["base_yaw_mag"] for r in rows]); am = np.array([r["arm_mag"] for r in rows])
    n = len(rows); q = [slice(i * n // 4, (i + 1) * n // 4) for i in range(4)]
    js = glob.glob(f"{R}/json/*.json"); d = json.load(open(js[0])) if js else {}
    ad = d.get("agent_distance", {}); nad = d.get("normalized_agent_distance", {})
    moving = (bt > 0.05).mean()
    last_move = int(np.max(np.where(bt > 0.05)[0])) if (bt > 0.05).any() else -1
    print(f"{os.path.basename(os.path.dirname(R))}/{os.path.basename(R)}: steps {n} | base_trans sum {bt.sum():.1f} (per quarter {[round(float(bt[s].sum()),1) for s in q]}), frac steps |v|>0.05 {moving:.2f}, last moving step {last_move} | yaw sum {by.sum():.1f} | arm_mag mean per quarter {[round(float(am[s].mean()),2) for s in q]} std {[round(float(am[s].std()),2) for s in q]} | evaluator agent_distance {ad} normalized {nad} success {d.get('success')}")

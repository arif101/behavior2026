"""Post-rollout behavioral battery (T2). Reads a step-trace JSONL, emits a verdict JSON.

Answers automatically what we previously needed human video-watching for:
  - did the policy ENGAGE (reach + gripper-close attempts)?
  - when was the first attempt; how many; how deep?
  - did behavior DEGRADE after failed attempts (off-distribution spiral)?
  - classification: never-engaged | near-miss | spiral-after-attempt | steady-no-attempt

Usage: python analyze_rollout.py --trace traces.jsonl [--out verdict.json]
Action layout (R1Pro 23-D): base 0:3 · torso 3:7 · armL 7:14 · gripL 14 · armR 15:22 · gripR 22
"""
import argparse
import json


def analyze(trace_path: str) -> dict:
    # A crash mid-write leaves a malformed line; appends after it start a new episode.
    # Split on malformed lines and analyze the last clean segment only.
    segs, cur = [], []
    for line in open(trace_path):
        try:
            cur.append(json.loads(line))
        except json.JSONDecodeError:
            if cur:
                segs.append(cur)
            cur = []
    if cur:
        segs.append(cur)
    rows = segs[-1] if segs else []
    n = len(rows)
    if n < 10:
        return {"error": "trace too short", "steps": n}
    acts = [r["action"] for r in rows]

    # --- gripper-close events (attempt proxy) ---
    closes = []
    for i in range(1, n):
        for g, side in ((14, "L"), (22, "R")):
            drop = acts[i - 1][g] - acts[i][g]
            if drop > 0.15:
                closes.append({"step": i, "side": side, "to": round(acts[i][g], 2)})
    deep_closes = [c for c in closes if c["to"] < 0.0]

    # --- proximity gating (when TARGET_CATS pose data present in trace) ---
    # objs logged every 10 steps (forward-fill), ee logged per step, both world-frame.
    REACH_M = 0.35
    obj_snaps = [(i, r["objs"]) for i, r in enumerate(rows) if r.get("objs")]
    have_prox = bool(obj_snaps) and any(r.get("ee") for r in rows)
    min_target_dist = None
    if have_prox:
        def targets_at(i):
            best = None
            for s, objs in obj_snaps:
                if best is None or abs(s - i) < abs(best[0] - i):
                    best = (s, objs)
            return best[1]
        def dist_at(i):
            ee = rows[i].get("ee") or {}
            ds = [sum((a - b) ** 2 for a, b in zip(e, o)) ** 0.5
                  for e in ee.values() for o in targets_at(i).values()]
            return min(ds) if ds else None
        all_d = [d for d in (dist_at(i) for i in range(n)) if d is not None]
        min_target_dist = round(min(all_d), 3) if all_d else None
        for c in closes:
            d = dist_at(c["step"])
            c["target_dist"] = round(d, 3) if d is not None else None
        target_closes = [c for c in closes if c["target_dist"] is not None and c["target_dist"] < REACH_M]
        air_closes = [c for c in closes if c not in target_closes]
        closes_for_cls = target_closes
    else:
        target_closes, air_closes, closes_for_cls = None, None, closes

    # --- activity curves in quarters ---
    def seg_stats(lo, hi):
        seg = acts[lo:hi]
        arm = sum(sum(abs(x) for x in a[7:14]) + sum(abs(x) for x in a[15:22]) for a in seg) / max(1, len(seg))
        base = sum(max(abs(a[0]), abs(a[1]), abs(a[2])) for a in seg) / max(1, len(seg))
        return round(arm, 2), round(base, 3)
    q = n // 4
    quarters = [seg_stats(i * q, (i + 1) * q) for i in range(4)]
    arm_escalation = quarters[3][0] / max(0.01, quarters[0][0])

    # --- classification (proximity-gated when pose data available) ---
    if not closes_for_cls:
        cls = "never-engaged" if quarters[0][0] < 1.0 else "steady-no-attempt"
    elif arm_escalation > 1.8:
        cls = "spiral-after-attempt"
    else:
        cls = "near-miss"

    out = {
        "steps": n,
        "attempts": len(closes),
        "deep_closes": len(deep_closes),
        "first_attempt_step": closes[0]["step"] if closes else None,
        "first_attempt_sec_at_30fps": round(closes[0]["step"] / 30.0, 1) if closes else None,
        "close_events": closes[:20],
        "quarter_activity_arm_base": quarters,
        "arm_escalation_ratio": round(arm_escalation, 2),
        "classification": cls,
    }
    if have_prox:
        out["proximity_gated"] = True
        out["target_attempts"] = len(target_closes)
        out["air_attempts"] = len(air_closes)
        out["min_target_dist_m"] = min_target_dist
        out["first_target_attempt_step"] = target_closes[0]["step"] if target_closes else None
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    v = analyze(args.trace)
    print(json.dumps(v, indent=1))
    if args.out:
        json.dump(v, open(args.out, "w"), indent=1)

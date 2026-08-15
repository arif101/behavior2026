"""PER-SCENE FRONTIER SCHEDULER for train_skill.py — prepped 2026-08-15 (Arif/Claude).

STATUS: READY-IF-NEEDED. Do NOT deploy while the stage-2 volume experiment is running.
This is the pre-built branch for the Saturday-morning readout: IF cold scenes do not
convert with 2-3x practice volume at a uniform global rung, switch the pass from
"every scene at stage S" to "every scene at its own frontier":

  - each scene trains at its OWN rung; promotion/demotion is per-scene, evidence-driven
  - warm scenes climb without waiting; cold scenes practice where they can earn reward
  - composes with (does not replace) chunk-level frontier/mastered mixing, episode caps,
    and quarantine — this only changes WHICH (demo, stage) chunk runs next

Rules (defaults mirror measured thresholds; all flaggable):
  promote:  chunk rolling20 >= --promote (0.5) -> stage+1 (capped at --max-stage)
  demote:   two consecutive chunks rolling20 < --demote (0.1) -> stage-1 (floor 0)
  resume:   newest ckpt whose stats show rolling20 >= 0.3 (mirrors the cd55ccc
            quarantine rule: failure-dominated chunks never seed the lineage)
  schedule: round-robin over scenes, skipping scenes mastered at --max-stage

State: --state (default /root/skill_frontier.json), seeded on first run from
--seed-stages JSON ({demo: stage}) or uniformly from --seed-stage.

Usage (after Saturday branch decision only):
  python -u skill_frontier_driver.py --demos 20 30 40 ... --seed-stages /root/seed.json \
      --bank /root/skill_start_bank.json --out /root/skill_ckpts_frontier
"""

import argparse
import glob
import json
import os
import subprocess
import sys

def newest_healthy_ckpt(out_dir, min_rolling=0.3):
    best = (None, -1.0)
    for sf in glob.glob(f"{out_dir}/stats_d*_s*.json"):
        try:
            st = json.load(open(sf))
        except Exception:
            continue
        if st.get("rolling20", 0.0) < min_rolling:
            continue
        pats = sorted(glob.glob(f"{out_dir}/skill_d{st['demo']}_s{st['stage']}_ep*.pt"),
                      key=os.path.getmtime)
        if pats and os.path.getmtime(pats[-1]) > best[1]:
            best = (pats[-1], os.path.getmtime(pats[-1]))
    return best[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", type=int, nargs="+", required=True)
    ap.add_argument("--state", default="/root/skill_frontier.json")
    ap.add_argument("--seed-stages", default=None, help="JSON file {demo: stage} initial frontiers")
    ap.add_argument("--seed-stage", type=int, default=1)
    ap.add_argument("--max-stage", type=int, default=3)
    ap.add_argument("--promote", type=float, default=0.5)
    ap.add_argument("--demote", type=float, default=0.1)
    ap.add_argument("--max-episodes", type=int, default=25)
    ap.add_argument("--out", default="/root/skill_ckpts")
    ap.add_argument("--train-script", default="/root/train_skill.py")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--rounds", type=int, default=4, help="full round-robins before exiting")
    a = ap.parse_args()

    if os.path.exists(a.state):
        S = json.load(open(a.state))
    else:
        seed = json.load(open(a.seed_stages)) if a.seed_stages else {}
        S = {str(d): {"stage": int(seed.get(str(d), a.seed_stage)), "dry": 0, "mastered": False}
             for d in a.demos}
    json.dump(S, open(a.state, "w"), indent=1)

    for rnd in range(a.rounds):
        for d in a.demos:
            s = S[str(d)]
            if s["mastered"]:
                continue
            resume = newest_healthy_ckpt(a.out)
            cmd = [a.python, "-u", a.train_script, "--demo-id", str(d),
                   "--stage", str(s["stage"]), "--max-episodes", str(a.max_episodes),
                   "--out", a.out]
            if resume:
                cmd += ["--resume", resume]
            print(f"[frontier] round {rnd} demo {d} stage {s['stage']} resume={resume}",
                  flush=True)
            subprocess.run(cmd, check=False)
            try:
                st = json.load(open(f"{a.out}/stats_d{d}_s{s['stage']}.json"))
                rolling = st.get("rolling20", 0.0)
            except Exception:
                rolling = 0.0
            if rolling >= a.promote:
                if s["stage"] >= a.max_stage:
                    s["mastered"] = True
                else:
                    s["stage"] += 1
                s["dry"] = 0
            elif rolling < a.demote:
                s["dry"] += 1
                if s["dry"] >= 2 and s["stage"] > 0:
                    s["stage"] -= 1
                    s["dry"] = 0
            else:
                s["dry"] = 0
            json.dump(S, open(a.state, "w"), indent=1)
        print(f"[frontier] round {rnd} complete: " +
              " ".join(f"d{d}:s{S[str(d)]['stage']}{'*' if S[str(d)]['mastered'] else ''}"
                       for d in a.demos), flush=True)

if __name__ == "__main__":
    main()

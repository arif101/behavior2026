"""Offline Opus consult: episode-start strategy from a situation briefing.

Sends the deliberative tier's situation report (NO human ground truth included) to
Claude Opus with the typed decision schema, temperature 0. Output is compared against
the demonstrator's actual opening — the strategy-agreement probe.

Key from file (never inline): ~/.claude/jobs/<job>/tmp/.anthropic_key or $ANTHROPIC_KEY_FILE.
Uses urllib only (no SDK dependency).

Usage:
  python -m director.harness.opus_consult --situations situations.jsonl --out consults.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request

API = "https://api.anthropic.com/v1/messages"
MODEL = "claude-opus-4-8"

SYSTEM = """You are the deliberative tier of a bimanual mobile household robot's controller.
At episode start you receive a situation report: the task goal (BDDL-style literals), the
objects present with their 3D world positions, and pairwise distances. The robot has two
arms (L, R) and a mobile base; one arm can carry an object while the other works; moving
the base across rooms is slow and should be minimized.

{menu}Decide the OPENING STRATEGY. Reply with ONLY a JSON object, no prose:
{
 "strategy": "<one sentence>",
 "first_actions": [{"arm": "L"|"R",
                    "skill": "move to"|"pick up from"|"place in"|"place on"|"place next to"|
                             "open door"|"close door"|"press"|"pull"|"push"|"toggle on",
                    "object": "<name>"}, ...]   // the first 2-5 actions in order
 "rationale": "<one sentence on why this ordering minimizes time>"
}"""

# Neutral pattern menu (names no objects — geometry must do the choosing).
MENU = """Common strategy patterns to weigh against the geometry (pick whatever fits, or none):
 - SHUTTLE: move to each item, bring it to the destination, repeat.
 - BATCH: collect several co-located items in one base position using both arms, then one trip.
 - CARRY-CONTAINER: if the destination/receptacle is itself portable, pick IT up and bring it
   to the items, loading as you go.
 - STAGE: pre-position items near the destination, then finish them all.

"""


def consult(key: str, situation: dict, menu: bool = False) -> dict:
    brief = {k: situation[k] for k in
             ("task", "objects_at_start", "initial_state", "pairwise_distances_m") if k in situation}
    brief["goal"] = situation.get("goal", f"complete the task: {situation['task']}")
    body = json.dumps({
        "model": MODEL,
        "max_tokens": 500,
        "system": SYSTEM.replace("{menu}", MENU if menu else ""),
        "messages": [{"role": "user", "content": f"SITUATION REPORT:\n{json.dumps(brief, indent=1)}"}],
    }).encode()
    req = urllib.request.Request(API, data=body, headers={
        "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            out = json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"API {e.code}: {e.read().decode()[:400]}") from None
    txt = out["content"][0]["text"].strip()
    if txt.startswith("```"):
        txt = txt.strip("`").lstrip("json").strip()
    usage = out.get("usage", {})
    return {"decision": json.loads(txt), "usage": usage}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--situations", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key-file", default=os.environ.get(
        "ANTHROPIC_KEY_FILE", os.path.expanduser("~/.claude/jobs/da71691e/tmp/.anthropic_key")))
    args = ap.parse_args()
    key = open(args.key_file).read().strip()
    with open(args.out, "w") as f:
        for line in open(args.situations):
            s = json.loads(line)
            r = consult(key, s)
            row = {"task": s["task"], "opus": r["decision"], "usage": r["usage"],
                   "human_opening": s["human_opening"]}
            f.write(json.dumps(row) + "\n")
            print(f"== {s['task']} ==")
            print("OPUS strategy:", r["decision"].get("strategy"))
            print("OPUS first_actions:", json.dumps(r["decision"].get("first_actions")))
            print("HUMAN opening:   ", json.dumps([
                {"skill": o["skill"], "object": (o["objects"] or [None])[0]} for o in s["human_opening"]]))
            print("rationale:", r["decision"].get("rationale"))
            it, ot = r["usage"].get("input_tokens", 0), r["usage"].get("output_tokens", 0)
            print(f"tokens: {it} in / {ot} out")


if __name__ == "__main__":
    main()

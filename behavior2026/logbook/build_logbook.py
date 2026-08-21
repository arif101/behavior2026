#!/usr/bin/env python3
"""Regenerate logbook.html from box telemetry. Read-only over training artifacts.

Sources: /root/skill_ckpts/stats_*.json (per-chunk aggregates), stage*_train.log
(per-episode EP lines -> failure terminal-dist buckets), the live train_skill.py
process (current chunk), and events.jsonl in this directory (append-only dated
narrative -- the actual "logbook" entries; regeneration never touches it).

Usage: python3 build_logbook.py   (writes logbook.html next to this file)
Republish: point the Artifact tool at logbook.html (same path = same URL).
"""

import glob
import html
import json
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
CKPTS = "/root/skill_ckpts"
LOGS = ["/root/stage1_train.log", "/root/stage2_train.log"]

EP_RE = re.compile(
    r"EP (\d+) steps=(\d+) success=(True|False) dist=([\d.]+) ep_r=([-\d.]+) "
    r"rolling20=([\d.]+) env_steps=(\d+) elapsed=(\d+)s")

# scenes with zero successes at every rung tried (recomputed below from stats)


def load_stats():
    rows = {}
    for f in sorted(glob.glob(f"{CKPTS}/stats_*.json")):
        d = json.load(open(f))
        rows[(d["demo"], d["stage"])] = d
    return rows


def per_scene(rows):
    scenes = {}
    for (demo, stage), d in sorted(rows.items()):
        s = scenes.setdefault(demo, {"demo": demo, "stages": {}, "eps": 0, "succ": 0,
                                     "steps": 0, "secs": 0})
        r = d["results"]
        s["stages"][stage] = {"eps": len(r), "succ": sum(r),
                              "rate": sum(r) / max(len(r), 1),
                              "steps": d["env_steps"], "secs": d["elapsed_s"]}
        s["eps"] += len(r)
        s["succ"] += sum(r)
        s["steps"] += d["env_steps"]
        s["secs"] += d["elapsed_s"]
    for s in scenes.values():
        s["best"] = max((v["rate"] for v in s["stages"].values()), default=0.0)
        s["status"] = ("cold" if s["succ"] == 0 else
                       "converged" if s["best"] >= 0.6 else "partial")
    return scenes


STEP_CAPS = {1: 3000, 2: 4500}  # per-stage --max-env-steps used by the campaign


def chunk_rows(rows):
    out = []
    for (demo, stage), d in sorted(rows.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        r = d["results"]
        n, s = len(r), sum(r)
        cap = STEP_CAPS.get(stage)
        if n >= 20 and d.get("rolling20", 0) >= 0.95:
            term = "early-stop 0.95"
        elif stage >= 2 and n >= 25 and (cap is None or d["env_steps"] < cap):
            term = "ep-cap 25"
        elif cap is not None and d["env_steps"] >= cap:
            term = f"step-cap {cap}"
        else:
            term = "other"
        out.append(
            f"<tr><td class='num'>s{stage}</td><td class='num'>d{demo}</td>"
            f"<td class='num'>{s}/{n}</td><td class='num'>{d.get('rolling20', 0):.2f}</td>"
            f"<td class='num'>{d['env_steps']:,}</td>"
            f"<td class='num'>{d['elapsed_s'] / 60:.0f}m</td><td>{term}</td></tr>")
    return out


def failure_buckets():
    out = {}
    for log in LOGS:
        if not os.path.exists(log):
            continue
        b = {"succ": 0, "fail_far": 0, "fail_near": 0, "fail_contact": 0, "eps": 0}
        succ_steps = []
        for line in open(log, errors="ignore"):
            m = EP_RE.search(line)
            if not m:
                continue
            b["eps"] += 1
            steps, su, dist = int(m.group(2)), m.group(3) == "True", float(m.group(4))
            if su:
                b["succ"] += 1
                succ_steps.append(steps)
            elif dist > 0.15:
                b["fail_far"] += 1
            elif dist > 0.05:
                b["fail_near"] += 1
            else:
                b["fail_contact"] += 1
        b["succ_med_steps"] = sorted(succ_steps)[len(succ_steps) // 2] if succ_steps else None
        out[os.path.basename(log)] = b
    return out


def live_run():
    try:
        ps = subprocess.run(["ps", "-eo", "pid,etime,args"], capture_output=True,
                            text=True).stdout
    except Exception:
        return None
    for line in ps.splitlines():
        if ("train_skill.py" in line or "train_skill_v2.py" in line) and "grep" not in line:
            m = re.search(r"--demo-id (\d+).*?--stage (\d+)", line)
            info = {"cmd": line.split("python", 1)[-1].strip()[:160],
                    "etime": line.split()[1],
                    "demo": m.group(1) if m else "?", "stage": m.group(2) if m else "?"}
            if "train_skill_v2" in line:
                for fr in ("/root/v2_full/flight_recorder.jsonl",
                           "/root/v2_smoke/flight_recorder.jsonl"):
                    if os.path.exists(fr):
                        r = json.loads(open(fr).readlines()[-1])
                        info["last_ep"] = (f"d{r['demo']} s{r['stage']} "
                                           f"success={r['success']} dist={r['dist']} "
                                           f"env_steps={r['env_steps']:,}")
                        break
                return info
            # v1 path: last EP line of the freshest train log
            newest = max((l for l in LOGS if os.path.exists(l)), key=os.path.getmtime)
            last = None
            for ln in open(newest, errors="ignore"):
                e = EP_RE.search(ln)
                if e:
                    last = e
            if last:
                info["last_ep"] = (f"EP {last.group(1)} success={last.group(3)} "
                                   f"dist={last.group(4)} rolling20={last.group(6)} "
                                   f"env_steps={last.group(7)}")
            return info
    return None


def v2_campaign():
    """Parse V2 flight recorders into a per-scene summary (cold set marked)."""
    import collections
    cold = {30, 50, 70, 110, 130, 160, 220, 240, 320, 340, 350, 370, 410}
    out = []
    for run, path in (("full", "/root/v2_full/flight_recorder.jsonl"),
                      ("smoke", "/root/v2_smoke/flight_recorder.jsonl")):
        if not os.path.exists(path):
            continue
        recs = [json.loads(l) for l in open(path)]
        if not recs:
            continue
        by = collections.defaultdict(list)
        for r in recs:
            by[r["demo"]].append(r)
        rows = []
        for d in sorted(by):
            rs = by[d]
            s = sum(r["success"] for r in rs)
            tag = " <span class='chip chip-cold'>cold-set</span>" if d in cold else ""
            cls = "chip-converged" if s / len(rs) >= 0.5 else \
                  ("chip-partial" if s else "chip-cold")
            rows.append(f"<tr><td class='num'>d{d}{tag}</td>"
                        f"<td><span class='chip {cls}'>{s}/{len(rs)}</span></td>"
                        f"<td class='num'>{sum(r['steps'] for r in rs):,}</td></tr>")
        conv = sum(1 for d in by if d in cold and any(r["success"] for r in by[d]))
        vis = sum(1 for d in by if d in cold)
        out.append(
            f"<h2>V2 campaign — {run} run</h2><div class='panel'>"
            f"<p style='margin-top:0'>episodes {len(recs):,} · env steps "
            f"{recs[-1]['env_steps']:,} · successes {sum(r['success'] for r in recs)} "
            f"({sum(r['success'] for r in recs) / len(recs):.0%}) · scenes {len(by)} · "
            f"<strong>cold-set scenes converted: {conv}/{vis} visited</strong></p>"
            f"<div class='table-wrap'><table><tr><th>scene</th><th>succ/eps</th>"
            f"<th>env steps</th></tr>{''.join(rows)}</table></div></div>")
    return "\n".join(out)


def events():
    p = f"{HERE}/events.jsonl"
    if not os.path.exists(p):
        return []
    return [json.loads(x) for x in open(p) if x.strip()]


def render(scenes, buckets, live, evs, stats_rows):
    tot_eps = sum(s["eps"] for s in scenes.values())
    tot_succ = sum(s["succ"] for s in scenes.values())
    tot_steps = sum(s["steps"] for s in scenes.values())
    tot_h = sum(s["secs"] for s in scenes.values()) / 3600
    cold = sorted(d for d, s in scenes.items() if s["status"] == "cold")
    conv = sorted(d for d, s in scenes.items() if s["status"] == "converged")

    def chip(status):
        label = {"converged": "converged", "partial": "partial", "cold": "cold 0%"}[status]
        return f'<span class="chip chip-{status}">{label}</span>'

    def bar(rate):
        pct = round(rate * 100)
        return (f'<div class="bar"><div class="bar-fill" style="width:{pct}%"></div>'
                f'<span class="bar-val">{pct}%</span></div>')

    scene_rows = []
    for demo in sorted(scenes):
        s = scenes[demo]
        stg = " · ".join(
            f's{k} {v["succ"]}/{v["eps"]}' for k, v in sorted(s["stages"].items()))
        scene_rows.append(
            f"<tr><td class='num'>d{demo}</td><td>{chip(s['status'])}</td>"
            f"<td>{bar(s['best'])}</td><td class='num'>{stg}</td>"
            f"<td class='num'>{s['steps']:,}</td>"
            f"<td class='num'>{s['secs'] / 3600:.1f}h</td></tr>")

    bucket_rows = []
    for name, b in buckets.items():
        fails = b["eps"] - b["succ"]
        bucket_rows.append(
            f"<tr><td>{name}</td><td class='num'>{b['eps']}</td>"
            f"<td class='num'>{b['succ']} ({b['succ'] / max(b['eps'], 1):.0%})</td>"
            f"<td class='num'>{b['fail_far']} ({b['fail_far'] / max(fails, 1):.0%} of fails)</td>"
            f"<td class='num'>{b['fail_near']}</td><td class='num'>{b['fail_contact']}</td>"
            f"<td class='num'>{b['succ_med_steps']}</td></tr>")

    live_html = "<p class='muted'>No skill training process running.</p>"
    if live:
        live_html = (f"<p><span class='chip chip-live'>live</span> "
                     f"<code>{html.escape(live['cmd'])}</code> — up {live['etime']}"
                     + (f"<br>last: <code>{html.escape(live.get('last_ep', ''))}</code>"
                        if live.get("last_ep") else "") + "</p>")

    ev_html = "\n".join(
        f"<div class='event'><div class='event-date'>{html.escape(e['date'])}"
        + (f" <span class='event-tag'>{html.escape(e['tag'])}</span>" if e.get('tag') else "")
        + f"</div><div class='event-body'>{e['html']}</div></div>"
        for e in reversed(evs))

    tpl = open(f"{HERE}/logbook_template.html").read()
    out = (tpl
           .replace("{{TOT_EPS}}", f"{tot_eps:,}")
           .replace("{{TOT_SUCC}}", f"{tot_succ:,}")
           .replace("{{SUCC_RATE}}", f"{tot_succ / max(tot_eps, 1):.0%}")
           .replace("{{TOT_STEPS}}", f"{tot_steps:,}")
           .replace("{{TOT_H}}", f"{tot_h:.1f}")
           .replace("{{THROUGHPUT}}", f"{tot_steps / max(tot_h * 3600, 1):.2f}")
           .replace("{{N_SCENES}}", str(len(scenes)))
           .replace("{{N_CONV}}", str(len(conv)))
           .replace("{{N_COLD}}", str(len(cold)))
           .replace("{{COLD_LIST}}", ", ".join(f"d{d}" for d in cold))
           .replace("{{SCENE_ROWS}}", "\n".join(scene_rows))
           .replace("{{CHUNK_ROWS}}", "\n".join(chunk_rows(stats_rows)))
           .replace("{{BUCKET_ROWS}}", "\n".join(bucket_rows))
           .replace("{{LIVE}}", live_html)
           .replace("{{V2}}", v2_campaign())
           .replace("{{EVENTS}}", ev_html)
           .replace("{{GENERATED}}", subprocess.run(["date", "-u", "+%Y-%m-%d %H:%M UTC"],
                                                    capture_output=True, text=True).stdout.strip()))
    open(f"{HERE}/logbook.html", "w").write(out)
    print(f"wrote {HERE}/logbook.html  ({len(scene_rows)} scenes, {len(evs)} events)")


if __name__ == "__main__":
    rows = load_stats()
    render(per_scene(rows), failure_buckets(), live_run(), events(), rows)

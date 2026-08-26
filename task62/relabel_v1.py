"""v1 CONTEXT labels from sim for task-62 (CONTEXT_STATES_SPEC_62 §3): per demo,
  (a) AG-held object per arm per frame, from the RECORDED states (sequential restore, stride K —
      every restored frame reads robot._ag_obj_in_hand; frames in between inherit);
  (b) progress q(t) from goal_status via per-segment near-anchor replay at QUASI-STATIC anchors
      (Step 0: mid-motion anchors nudge objects; anchors <=120 frames before closure succeed);
      q is piecewise-constant, updated at each scoring segment's end (+slack) — the human's
      spike stream is NOT used;
  (c) context_weight per segment: 1.0 if the replay reproduced the human outcome (segment
      predicate true and the recorded q-step observed), else 0.5.
Output: /root/step0/relabel_v1/ep{raw}.npz with ag_L, ag_R (object name id), q (T,), seg_ok
(per segment), plus a JSON summary. build_t62_dataset.py --v1 merges these over the v0 labels.

Single sim per box. Usage:
  ... python -u task62/relabel_v1.py --demos 620010,620030 --stride 5 --out /root/step0/relabel_v1
"""
import os, sys, json, time, argparse, traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task62.step0_replay_sweep import Sim, load_demo, segments, closure_frames, score, chop_spike, log, PRE, SLACK

ARM_CH = {"left": slice(7, 14), "right": slice(15, 22)}
SCORING = ("chop", "place_knife_sink", "place_half_plate")   # segments that change q


def quasi_static_anchor(act, seg_start, lo_off=150, hi_off=PRE, win=10, tol=0.004):
    """Latest frame in [seg_start-lo_off, seg_start-hi_off] where arm+trunk commands are still over
    a window and the base command is ~0. Falls back to seg_start-hi_off."""
    lo, hi = max(win, seg_start - lo_off), max(win, seg_start - hi_off)
    for t in range(hi, lo, -1):
        w = act[t - win:t + 1]
        if np.abs(np.diff(w[:, 3:14], axis=0)).max() < tol and np.abs(np.diff(w[:, 15:22], axis=0)).max() < tol \
           and np.abs(w[:, 0:3]).max() < 1e-3:
            return t
    return hi


def ag_labels(sim, d, stride):
    """(a): sequential restore over the whole demo, sampling AG per arm every `stride` frames."""
    T = len(d["action"]); agL = np.zeros(T, np.int8); agR = np.zeros(T, np.int8)   # 0 none, 1 egg/half, 2 knife
    names = {}
    def cls(n):
        if not n: return 0
        return 2 if "knife" in n else 1
    sim.ensure_pre_slice_composition()
    spike = chop_spike(d); prev = 0
    for t in range(0, T, stride):
        # post-slice frames need the halves to exist: slice in-episode once, then keep restoring forward
        if spike is not None and t >= spike + 15 and not sim.halves_exist():
            sim.restore_sequential(d, max(0, spike - 60), start=prev + 1 if prev else 0)
            for k in range(max(0, spike - 60), spike + 15): sim.step(d["action"][k])
            prev = spike + 15
        try:
            sim.restore_sequential(d, t, start=prev + 1 if prev else 0, stride=stride); prev = t
        except Exception as e:
            log(f"  restore@{t} failed: {type(e).__name__}"); continue
        ag = sim.ag(); agL[t:t + stride] = cls(ag.get("left")); agR[t:t + stride] = cls(ag.get("right"))
    return agL, agR


def progress_labels(sim, d):
    """(b)+(c): per scoring segment, restore at a quasi-static anchor, replay to end+slack, read q."""
    T = len(d["action"]); act = d["action"]; q = np.zeros(T, np.float32); seg_ok = {}
    spike = chop_spike(d); held = set(); cur_q = 0.0
    for seg in segments(d):
        if seg["tag"] not in SCORING: continue
        anchor = quasi_static_anchor(act, seg["start"]); hi = min(T, seg["end"] + SLACK)
        try:
            sim.ensure_pre_slice_composition()
            if seg["post_slice"]:
                sim.restore_sequential(d, max(0, spike - 60))
                for k in range(max(0, spike - 60), spike + 15): sim.step(act[k])
                sim.restore_sequential(d, anchor, start=spike + 16)
            else:
                sim.restore_sequential(d, anchor)
            q0, sat0 = sim.q()
            for k in range(anchor, hi): sim.step(act[k])
            r = score(seg["tag"], sim, held); q1, sat1 = sim.q()
            gained = sorted(set(sat1) - set(sat0)); held |= set(sat1)
            ok = bool(r.get("success")) or len(gained) > 0
            seg_ok[f"{seg['tag']}#{seg['idx']}"] = dict(ok=ok, anchor=anchor, q0=q0, q1=q1, gained=gained)
            if q1 > cur_q: cur_q = q1
            q[seg["end"]:] = max(cur_q, q1)
            log(f"  {seg['tag']}#{seg['idx']} anchor {anchor} ({seg['start']-anchor} pre): q {q0:.1f}->{q1:.1f} ok={ok}")
        except Exception as e:
            seg_ok[f"{seg['tag']}#{seg['idx']}"] = dict(ok=False, error=repr(e)[:120]); log(f"  {seg['tag']} ERROR {traceback.format_exc()[-300:]}")
    return q, seg_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="620010")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--out", default="/root/step0/relabel_v1")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    sim = Sim(); summary = {}
    for ep in [int(x) for x in a.demos.split(",")]:
        d = load_demo(ep); t0 = time.time(); log(f"ep{ep}: T={len(d['action'])}")
        try:
            agL, agR = ag_labels(sim, d, a.stride)
            q, seg_ok = progress_labels(sim, d)
            np.savez(f"{a.out}/ep{ep}.npz", ag_L=agL, ag_R=agR, q=q)
            summary[ep] = dict(minutes=round((time.time() - t0) / 60, 1), q_final=float(q[-1]),
                               agL_frac=float((agL > 0).mean()), agR_frac=float((agR > 0).mean()), seg_ok=seg_ok)
            log(f"ep{ep} done in {summary[ep]['minutes']} min: q_final={q[-1]:.1f} segs={ {k: v.get('ok') for k, v in seg_ok.items()} }")
        except Exception as e:
            summary[ep] = dict(error=repr(e)[:200]); log(f"ep{ep} FAILED {traceback.format_exc()[-400:]}")
        json.dump(summary, open(f"{a.out}/summary_{a.demos.replace(',', '_')[:40]}.json", "w"), indent=1)
    log("RELABEL_V1_DONE"); os._exit(0)


if __name__ == "__main__":
    main()

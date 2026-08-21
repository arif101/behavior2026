"""T1 diagnostics renderer: evaluation rollout (trace + video) -> annotated debug video.

Failure-analysis instrument for G3 (4-arm policy evaluation). Composites, per frame:
  - HEADER: battery verdict summary (classification, attempts, first-attempt time)
    computed via probes/analyze_rollout (single source of truth for conventions).
  - LIVE PANEL: action-magnitude bars (base/torso/armL/armR), per-arm gripper
    open/close indicators, step counter + wall time, and (when the trace carries
    objs/ee fields) EE-to-target distance readouts + a proximity sparkline.
  - TIMELINE BAR: full-episode strip with per-step gripper states (both arms),
    close-event ticks (target-gated vs air when prox data present), battery
    quarter-activity zones, and a current-position cursor.
  - GROUNDING OVERLAY (--grounding): per-step predicted points drawn on the video.
    Schema (one JSON object per line):
      {"step": int, "arm": "L"|"R"|null, "uv": [u, v], "conf": float, "valid": bool,
       "label": str (optional)}
    uv = pixel coords in the SOURCE video frame. Points are forward-filled for up
    to GROUND_HOLD steps and fade with age; valid=false renders as a gray X.

--contact-sheet emits a PNG grid of the K most informative frames instead of a
video: init, first approach, each close-event cluster (pre/at/post context),
closest-approach (prox traces), final state — with the full timeline strip below.

Video/trace alignment: trace step k <-> video frame k * n_video_frames / n_steps.
Action layout (R1Pro 23-D): base 0:3 · torso 3:7 · armL 7:14 · gripL 14 · armR 15:22 · gripR 22

Usage:
  python render_debug_video.py --trace t.jsonl --video in.mp4 --out dbg.mp4 [--speed 3]
  python render_debug_video.py --trace t.jsonl --video in.mp4 --out sheet.png --contact-sheet
"""
import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "probes"))
from analyze_rollout import analyze, find_closes, load_trace, GRIP_DIMS, REACH_M  # noqa: E402

# ---------------------------------------------------------------- conventions
ACTION_GROUPS = [("base", 0, 3), ("torso", 3, 7), ("armL", 7, 14), ("armR", 15, 22)]
STEP_HZ = 30.0          # trace steps per wall-clock second (matches analyze_rollout)
GROUND_HOLD = 30        # steps a grounding point persists (fading) after its stamp

# ------------------------------------------------------------------- palette
BG = (13, 17, 23)
PANEL_BG = (22, 27, 34)
FG = (230, 237, 243)
DIM = (139, 148, 158)
GRID = (48, 54, 61)
COL_BASE = (88, 166, 255)
COL_TORSO = (210, 153, 34)
COL_ARML = (63, 185, 80)
COL_ARMR = (247, 120, 186)
COL_CLOSED = (248, 81, 73)
COL_OPEN = (63, 185, 80)
COL_TARGET_TICK = (46, 230, 168)
COL_AIR_TICK = (240, 136, 62)
COL_PLAIN_TICK = (255, 255, 255)
COL_CURSOR = (255, 255, 255)
CLS_COLOR = {
    "spiral-after-attempt": (248, 81, 73),
    "never-engaged": (139, 148, 158),
    "steady-no-attempt": (210, 153, 34),
    "near-miss": (88, 166, 255),
}

_FONT_PATHS = [
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Monaco.ttf",
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]


def font(size):
    for p in _FONT_PATHS:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_SM, F_MD, F_LG = font(11), font(13), font(16)


# ============================================================== episode data
class Episode:
    """Everything per-step precomputed from the trace, plus the battery verdict."""

    def __init__(self, trace_path):
        self.rows = load_trace(trace_path)
        self.n = len(self.rows)
        if self.n < 10:
            raise SystemExit(f"trace too short ({self.n} rows): {trace_path}")
        self.acts = np.array([r["action"] for r in self.rows], dtype=np.float32)
        self.verdict = analyze(trace_path)

        # action-magnitude curves (sum |.| per group, gripper dims excluded)
        self.mags = {name: np.abs(self.acts[:, lo:hi]).sum(axis=1)
                     for name, lo, hi in ACTION_GROUPS}
        self.mag_norm = {k: max(1e-3, float(np.percentile(v, 99))) for k, v in self.mags.items()}
        self.grip = {"L": self.acts[:, 14], "R": self.acts[:, 22]}

        # close events (same detector as the battery)
        self.closes = find_closes([list(a) for a in self.acts])

        # proximity (objs every ~10 steps -> nearest snapshot; ee per step; world frame)
        self._build_prox()
        if self.have_prox:
            for c in self.closes:
                d = self.min_dist[c["step"]]
                c["target_dist"] = round(float(d), 3) if np.isfinite(d) else None
        self.first_approach = self._first_approach()

    def _build_prox(self):
        snaps = [(i, r["objs"]) for i, r in enumerate(self.rows) if r.get("objs")]
        self.have_prox = bool(snaps) and any(r.get("ee") for r in self.rows)
        self.dist = {"left": np.full(self.n, np.nan), "right": np.full(self.n, np.nan)}
        self.near_obj = {"left": [None] * self.n, "right": [None] * self.n}
        if not self.have_prox:
            self.min_dist = np.full(self.n, np.nan)
            return
        snap_steps = np.array([s for s, _ in snaps])
        for i, r in enumerate(self.rows):
            ee = r.get("ee") or {}
            if not ee:
                continue
            objs = snaps[int(np.argmin(np.abs(snap_steps - i)))][1]  # nearest snapshot
            for arm, e in ee.items():
                best, best_name = np.inf, None
                for name, o in objs.items():
                    d = math.dist(e, o)
                    if d < best:
                        best, best_name = d, name
                if arm in self.dist and np.isfinite(best):
                    self.dist[arm][i] = best
                    self.near_obj[arm][i] = best_name
        self.min_dist = np.fmin(self.dist["left"], self.dist["right"])

    def _first_approach(self):
        """First step the policy plausibly starts an approach (for the contact sheet)."""
        if self.have_prox:
            idx = np.where(self.min_dist < 0.5)[0]
            if len(idx):
                return int(idx[0])
        arm = self.mags["armL"] + self.mags["armR"]
        k = min(25, self.n)
        smooth = np.convolve(arm, np.ones(k) / k, mode="same")
        thr = 0.5 * np.percentile(smooth, 95)
        idx = np.where(smooth > thr)[0]
        return int(idx[0]) if len(idx) else 0

    def close_tick_color(self, c):
        if not self.have_prox:
            return COL_PLAIN_TICK
        d = c.get("target_dist")
        return COL_TARGET_TICK if (d is not None and d < REACH_M) else COL_AIR_TICK


# ================================================================= grounding
def load_grounding(path):
    """step -> {arm: entry}; latest entry per (step, arm) wins."""
    per_step = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        e = json.loads(line)
        per_step.setdefault(int(e["step"]), {})[e.get("arm") or "?"] = e
    return per_step


def grounding_at(per_step, step):
    """Latest entry per arm within the trailing GROUND_HOLD window, with age."""
    out = {}
    for s in range(max(0, step - GROUND_HOLD), step + 1):
        for arm, e in per_step.get(s, {}).items():
            out[arm] = (e, step - s)
    return list(out.values())


def draw_grounding(draw, entries, scale):
    for e, age in entries:
        u, v = e["uv"][0] * scale, e["uv"][1] * scale
        conf = float(e.get("conf", 1.0))
        fade = max(0.25, 1.0 - age / GROUND_HOLD)
        arm = e.get("arm") or "?"
        col = {"L": (46, 230, 230), "R": (247, 120, 186)}.get(arm, (255, 255, 0))
        col = tuple(int(c * fade) for c in col)
        if not e.get("valid", True):
            g = int(160 * fade)
            for dx in (-6, 6):
                draw.line([u - dx, v - 6, u + dx, v + 6], fill=(g, g, g), width=2)
            continue
        r = 4 + 8 * conf
        draw.ellipse([u - r, v - r, u + r, v + r], outline=col, width=2)
        draw.line([u - r - 4, v, u + r + 4, v], fill=col, width=1)
        draw.line([u, v - r - 4, u, v + r + 4], fill=col, width=1)
        label = f"{arm} {conf:.2f}" + (f" {e['label']}" if e.get("label") else "")
        draw.text((u + r + 5, v - 7), label, font=F_SM, fill=col)


# ================================================================== timeline
class Timeline:
    """Pre-rendered full-episode strip; per frame we paste it and add the cursor."""

    H = 104

    def __init__(self, ep, width):
        self.ep, self.W = ep, width
        self.x0, self.x1 = 34, width - 12
        self.pw = self.x1 - self.x0
        self.base = self._render_base()

    def x(self, step):
        return self.x0 + int(self.pw * step / max(1, self.ep.n - 1))

    def _strip(self, values):
        """Map a per-step scalar array to per-pixel columns (nearest step)."""
        idx = (np.arange(self.pw) * (self.ep.n - 1) / max(1, self.pw - 1)).astype(int)
        return values[idx]

    def _render_base(self):
        ep = self.ep
        img = Image.new("RGB", (self.W, self.H), BG)
        d = ImageDraw.Draw(img)
        y_tick, th_tick = 4, 12          # close-event tick lane
        y_zone, th_zone = 18, 16         # battery quarter zones
        y_gl, y_gr, th_g = 36, 50, 12    # gripper strips
        y_axis = 66

        # battery quarter-activity zones (analyze_rollout's seg_stats quarters)
        quarters = ep.verdict.get("quarter_activity_arm_base", [])
        if quarters:
            qmax = max(0.01, max(q[0] for q in quarters))
            q = ep.n // 4
            for qi, (arm_a, base_a) in enumerate(quarters):
                xa = self.x(qi * q)
                xb = self.x((qi + 1) * q if qi < 3 else ep.n - 1)
                heat = arm_a / qmax
                col = (int(40 + 180 * heat), int(40 + 70 * heat), 40)
                d.rectangle([xa, y_zone, xb, y_zone + th_zone], fill=col)
                d.text((xa + 3, y_zone + 2), f"Q{qi+1} arm {arm_a:g} base {base_a:g}",
                       font=F_SM, fill=FG)
                if qi:
                    d.line([xa, y_zone, xa, y_gr + th_g], fill=BG, width=1)

        # per-step gripper strips: green=open, amber=partial, red=closed
        for side, y in (("L", y_gl), ("R", y_gr)):
            v = self._strip(ep.grip[side])
            strip = np.empty((th_g, self.pw, 3), dtype=np.uint8)
            strip[:] = (33, 70, 45)
            strip[:, (v < 0.5) & (v >= 0.0)] = (181, 137, 0)
            strip[:, v < 0.0] = COL_CLOSED
            img.paste(Image.fromarray(strip), (self.x0, y))
            d.text((10, y), f"g{side}", font=F_SM, fill=DIM)

        # close-event ticks (target vs air vs ungated), deep closes full-height
        for c in ep.closes:
            xc = self.x(c["step"])
            col = ep.close_tick_color(c)
            deep = c["to"] < 0.0
            d.line([xc, y_tick, xc, y_tick + (th_tick if deep else th_tick // 2)],
                   fill=col, width=2 if deep else 1)

        # axis: step labels
        step_grid = 500 if ep.n > 1500 else 100
        for s in range(0, ep.n, step_grid):
            xs = self.x(s)
            d.line([xs, y_axis, xs, y_axis + 4], fill=GRID)
            d.text((xs + 2, y_axis + 4), str(s), font=F_SM, fill=DIM)
        d.text((10, y_axis + 4), "step", font=F_SM, fill=DIM)

        # legend + classification
        cls = ep.verdict.get("classification", "?")
        lx = self.x0
        ly = self.H - 16
        items = [("deep close", COL_PLAIN_TICK)] if not ep.have_prox else \
            [("target close", COL_TARGET_TICK), ("air close", COL_AIR_TICK)]
        items += [("grip closed", COL_CLOSED), ("grip open", (63, 185, 80))]
        for name, col in items:
            d.rectangle([lx, ly + 4, lx + 8, ly + 12], fill=col)
            d.text((lx + 12, ly), name, font=F_SM, fill=DIM)
            lx += 12 + int(d.textlength(name, font=F_SM)) + 14
        d.text((self.x1 - d.textlength(f"zones: battery [{cls}]", font=F_SM), ly),
               f"zones: battery [{cls}]", font=F_SM, fill=CLS_COLOR.get(cls, DIM))
        return img

    def stamp(self, step):
        img = self.base.copy()
        d = ImageDraw.Draw(img)
        xc = self.x(step)
        d.line([xc, 2, xc, 82], fill=COL_CURSOR, width=2)
        d.polygon([(xc - 4, 0), (xc + 4, 0), (xc, 6)], fill=COL_CURSOR)
        return img


# ==================================================================== panel
class Panel:
    W = 304

    def __init__(self, ep, height, grounding=None):
        self.ep, self.H, self.grounding = ep, height, grounding

    def render(self, step):
        ep = self.ep
        img = Image.new("RGB", (self.W, self.H), PANEL_BG)
        d = ImageDraw.Draw(img)
        y = 8
        d.text((12, y), f"step {step:>5d} / {ep.n}", font=F_LG, fill=FG)
        d.text((12, y + 20), f"t = {step / STEP_HZ:6.1f}s  (@{STEP_HZ:g} steps/s)",
               font=F_MD, fill=DIM)
        y += 46

        # action-magnitude bars, normalized to episode p99; red when near max
        d.text((12, y), "action |.|  (bar = episode p99)", font=F_SM, fill=DIM)
        y += 16
        colors = {"base": COL_BASE, "torso": COL_TORSO, "armL": COL_ARML, "armR": COL_ARMR}
        bw = self.W - 110
        for name, _, _ in ACTION_GROUPS:
            v = float(ep.mags[name][step])
            frac = min(1.0, v / ep.mag_norm[name])
            col = COL_CLOSED if frac > 0.9 else colors[name]
            d.text((12, y), name, font=F_MD, fill=FG)
            d.rectangle([62, y + 2, 62 + bw, y + 12], outline=GRID)
            d.rectangle([62, y + 2, 62 + int(bw * frac), y + 12], fill=col)
            d.text((66 + bw, y), f"{v:5.2f}", font=F_MD, fill=FG)
            y += 20
        y += 8

        # gripper indicators
        d.text((12, y), "grippers", font=F_SM, fill=DIM)
        y += 15
        for i, side in enumerate(("L", "R")):
            v = float(ep.grip[side][step])
            closed = v < 0.0
            x0 = 12 + i * 142
            col = COL_CLOSED if closed else (COL_TORSO if v < 0.5 else COL_OPEN)
            d.rectangle([x0, y, x0 + 130, y + 26], outline=col, width=2)
            d.text((x0 + 8, y + 5), f"{side}: {'CLOSED' if closed else 'open':6s} {v:+.2f}",
                   font=F_MD, fill=col)
        y += 38

        # proximity block
        if ep.have_prox:
            d.text((12, y), f"EE -> target dist  (reach {REACH_M}m)", font=F_SM, fill=DIM)
            y += 15
            for arm, tag in (("left", "L"), ("right", "R")):
                dv = ep.dist[arm][step]
                name = ep.near_obj[arm][step] or "-"
                if np.isfinite(dv):
                    col = COL_TARGET_TICK if dv < REACH_M else FG
                    d.text((12, y), f"{tag} {dv:5.2f}m  {name}", font=F_MD, fill=col)
                else:
                    d.text((12, y), f"{tag}   -  ", font=F_MD, fill=DIM)
                y += 17
            y += 4
            y = self._sparkline(d, step, y)
        if self.grounding is not None:
            d.text((12, y), "grounding: points drawn on frame", font=F_SM, fill=DIM)
            y += 14
            for e, age in grounding_at(self.grounding, step):
                ok = "ok" if e.get("valid", True) else "INVALID"
                d.text((12, y), f"{e.get('arm','?')} conf {e.get('conf',0):.2f} "
                                f"{ok} (age {age})", font=F_SM, fill=FG)
                y += 13
        return img

    def _sparkline(self, d, step, y):
        ep, w, h, win = self.ep, self.W - 24, 52, 450
        lo = max(0, step - win)
        seg = ep.min_dist[lo:step + 1]
        d.rectangle([12, y, 12 + w, y + h], outline=GRID)
        ymax = max(1.2, float(np.nanpercentile(ep.min_dist, 95))
                   if np.isfinite(ep.min_dist).any() else 1.2)

        def sy(v):
            return y + h - 2 - (min(v, ymax) / ymax) * (h - 4)

        ty = sy(REACH_M)
        for xx in range(12, 12 + w, 8):  # dashed reach threshold
            d.line([xx, ty, xx + 4, ty], fill=COL_CLOSED)
        pts = [(12 + 2 + (w - 4) * i / max(1, len(seg) - 1), sy(v))
               for i, v in enumerate(seg) if np.isfinite(v)]
        if len(pts) > 1:
            d.line(pts, fill=COL_BASE, width=2)
        if pts:
            cx, cy = pts[-1]
            d.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=FG)
        d.text((16, y + 2), f"min EE-target dist, last {win} steps", font=F_SM, fill=DIM)
        return y + h + 10


# ==================================================================== header
def render_header(ep, width, trace_name, speed):
    H = 46
    img = Image.new("RGB", (width, H), BG)
    d = ImageDraw.Draw(img)
    v = ep.verdict
    cls = v.get("classification", "?")
    parts = [f"{trace_name}", f"[{cls.upper()}]",
             f"attempts {v.get('attempts', 0)} ({v.get('deep_closes', 0)} deep)"]
    if v.get("first_attempt_step") is not None:
        parts.append(f"first@{v['first_attempt_step']} ({v['first_attempt_sec_at_30fps']}s)")
    if v.get("proximity_gated"):
        parts.append(f"target/air {v['target_attempts']}/{v['air_attempts']} "
                     f"min_d {v['min_target_dist_m']}m")
    parts.append(f"esc {v.get('arm_escalation_ratio', '?')}")
    line = "  ·  ".join(parts)
    f = F_LG if d.textlength(line, font=F_LG) <= width - 24 else \
        (F_MD if d.textlength(line, font=F_MD) <= width - 24 else F_SM)
    d.text((12, 6), line, font=f, fill=CLS_COLOR.get(cls, FG))
    sub = f"T1 debug render · {ep.n} steps · battery=analyze_rollout"
    sub += f" · playback {speed}x" if speed > 1 else ""
    d.text((12, 27), sub, font=F_SM, fill=DIM)
    return img


# =============================================================== ffmpeg pipes
def probe_video(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=width,height,r_frame_rate,nb_read_packets",
         "-of", "json", path], capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return int(s["width"]), int(s["height"]), int(s["nb_read_packets"]), float(num) / float(den)


def decode_frames(path, w, h, scale=1.0):
    """Yield RGB frames (numpy) at scaled size via an ffmpeg rawvideo pipe."""
    sw, sh = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24"]
    if scale != 1.0:
        cmd += ["-vf", f"scale={sw}:{sh}"]
    cmd += ["-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    fsz = sw * sh * 3
    try:
        while True:
            buf = proc.stdout.read(fsz)
            if len(buf) < fsz:
                break
            yield np.frombuffer(buf, np.uint8).reshape(sh, sw, 3)
    finally:
        proc.stdout.close()
        proc.wait()


class Encoder:
    def __init__(self, path, w, h, fps=30, crf=26):
        self.proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
             "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)

    def write(self, img):
        self.proc.stdin.write(np.asarray(img, dtype=np.uint8).tobytes())

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


# ============================================================== video render
def render_video(args, ep, grounding):
    vw, vh, nf, fps = probe_video(args.video)
    scale = args.vscale if args.vscale else (0.5 if vw > 1000 else 1.0)
    sw, sh = int(vw * scale) // 2 * 2, int(vh * scale) // 2 * 2
    W = ((sw + Panel.W) // 2) * 2
    H_HDR = 46
    H = ((H_HDR + sh + Timeline.H) // 2) * 2

    timeline = Timeline(ep, W)
    panel = Panel(ep, sh, grounding)
    header = render_header(ep, W, os.path.basename(args.trace), args.speed)
    canvas = Image.new("RGB", (W, H), BG)
    canvas.paste(header, (0, 0))

    enc = Encoder(args.out, W, H, fps=fps, crf=args.crf)
    n_out = 0
    for f, frame in enumerate(decode_frames(args.video, vw, vh, scale)):
        if f % args.speed:
            continue
        step = min(ep.n - 1, round(f * ep.n / nf))
        fr = Image.fromarray(frame)
        if grounding is not None:
            entries = grounding_at(grounding, step)
            if entries:
                draw_grounding(ImageDraw.Draw(fr), entries, scale)
        canvas.paste(fr, (0, H_HDR))
        canvas.paste(panel.render(step), (sw, H_HDR))
        canvas.paste(timeline.stamp(step), (0, H_HDR + sh))
        enc.write(canvas)
        n_out += 1
    enc.close()
    print(f"wrote {args.out}: {n_out} frames {W}x{H} "
          f"({os.path.getsize(args.out)/1e6:.1f} MB)")


# ============================================================= contact sheet
def pick_keyframes(ep, k):
    """(step, label) list: init, first approach, close clusters +/- context,
    closest approach, final."""
    cands = [(0, "init"), (ep.first_approach, "first-approach"), (ep.n - 1, "final")]
    if ep.have_prox and np.isfinite(ep.min_dist).any():
        s = int(np.nanargmin(ep.min_dist))
        cands.append((s, f"closest {np.nanmin(ep.min_dist):.2f}m"))
    clusters = []
    for c in ep.closes:  # cluster closes within 40 steps
        if clusters and c["step"] - clusters[-1][-1]["step"] <= 40:
            clusters[-1].append(c)
        else:
            clusters.append([c])
    for cl in clusters[:4]:
        c0 = cl[0]
        deepest = min(cl, key=lambda c: c["to"])
        tag = f"close {deepest['side']}->{deepest['to']:+.2f}"
        if ep.have_prox and deepest.get("target_dist") is not None:
            tag += f" d={deepest['target_dist']}m"
        cands += [(max(0, c0["step"] - 45), "pre-close"),
                  (deepest["step"], tag.upper()),
                  (min(ep.n - 1, cl[-1]["step"] + 90), "post-close")]
    seen, out = set(), []
    for s, lab in sorted(cands):
        s = min(ep.n - 1, max(0, s))
        if s in seen:
            continue
        seen.add(s)
        out.append((s, lab))
    while len(out) > k:  # drop context frames first, keep the story
        pre = [i for i, (_, l) in enumerate(out) if l in ("pre-close", "post-close")]
        out.pop(pre[-1] if pre else len(out) - 2)
    return out


def render_contact_sheet(args, ep, grounding):
    vw, vh, nf, fps = probe_video(args.video)
    keys = pick_keyframes(ep, args.tiles)
    want = {min(nf - 1, round(s * nf / ep.n)): (s, lab) for s, lab in keys}
    tiles = {}
    for f, frame in enumerate(decode_frames(args.video, vw, vh, 1.0)):
        if f in want:
            tiles[f] = frame.copy()
        if len(tiles) == len(want):
            break

    tw, th, cap = 336, int(336 * vh / vw), 20
    cols = min(4, len(want))
    rows_n = math.ceil(len(want) / cols)
    W = cols * (tw + 8) + 8
    H = 46 + rows_n * (th + cap + 8) + 8 + Timeline.H
    sheet = Image.new("RGB", (W, H), BG)
    sheet.paste(render_header(ep, W, os.path.basename(args.trace), 1), (0, 0))
    d = ImageDraw.Draw(sheet)

    for i, f in enumerate(sorted(want)):
        s, lab = want[f]
        x = 8 + (i % cols) * (tw + 8)
        y = 46 + (i // cols) * (th + cap + 8)
        img = Image.fromarray(tiles[f]).resize((tw, th))
        if grounding is not None:
            entries = grounding_at(grounding, s)
            if entries:
                draw_grounding(ImageDraw.Draw(img), entries, tw / vw)
        sheet.paste(img, (x, y))
        col = COL_CLOSED if lab.startswith("CLOSE") else FG
        d.text((x + 2, y + th + 3), f"s{s} ({s/STEP_HZ:.0f}s)  {lab}", font=F_MD, fill=col)

    tl = Timeline(ep, W)
    strip = tl.base.copy()
    ds = ImageDraw.Draw(strip)
    for s, _ in keys:  # mark chosen frames on the strip
        xs = tl.x(s)
        ds.polygon([(xs - 4, 0), (xs + 4, 0), (xs, 7)], fill=COL_BASE)
    sheet.paste(strip, (0, H - Timeline.H))
    sheet.save(args.out)
    print(f"wrote {args.out}: {len(want)} tiles {W}x{H} "
          f"({os.path.getsize(args.out)/1e6:.1f} MB)")


# ====================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", required=True, help="step-trace JSONL")
    ap.add_argument("--video", required=True, help="matching rollout MP4")
    ap.add_argument("--out", required=True, help="output MP4 (or PNG with --contact-sheet)")
    ap.add_argument("--grounding", default=None,
                    help="per-step predicted-point JSONL: "
                         '{"step","arm","uv":[u,v],"conf","valid"}')
    ap.add_argument("--speed", type=int, default=1, help="emit every Nth frame (N x faster)")
    ap.add_argument("--contact-sheet", action="store_true",
                    help="emit a PNG grid of key frames instead of a video")
    ap.add_argument("--tiles", type=int, default=12, help="max contact-sheet tiles")
    ap.add_argument("--crf", type=int, default=26, help="x264 quality (higher = smaller)")
    ap.add_argument("--vscale", type=float, default=None,
                    help="scale source video (default 1.0; auto 0.5 if width>1000)")
    args = ap.parse_args()

    ep = Episode(args.trace)
    grounding = load_grounding(args.grounding) if args.grounding else None
    print(f"trace: {ep.n} steps · verdict: {ep.verdict.get('classification')}"
          f" · closes: {len(ep.closes)} · prox: {ep.have_prox}")
    if args.contact_sheet:
        render_contact_sheet(args, ep, grounding)
    else:
        render_video(args, ep, grounding)


if __name__ == "__main__":
    main()

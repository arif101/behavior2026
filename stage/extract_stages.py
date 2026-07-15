#!/usr/bin/env python3
"""
extract_stages.py -- Subtask / stage-label extractor for BEHAVIOR-2026 long-horizon tasks.

The "fourth head" data pipeline: derive per-frame subtask labels from PRIVILEGED STATE
(object world positions + R1Pro proprio) so a stage-prediction head can later be distilled
from them.  NO sim, NO rendering, NO GPU -- pure geometry over positions + proprio.

Signals used (all CPU, from positions + proprio):
  1. Per-object world velocity  : finite-difference of object world positions (smoothed).
  2. Gripper events             : gripper_qpos per arm (close = qpos drops).  [SECONDARY/noisy]
  3. EE<->object proximity      : eef_pos transformed to world frame, dist to each object.  [PRIMARY]
  4. Object-at-goal             : object relocated + stationary + not carried after release.

Output:
  * SUBTASK TIMELINE   : list of (active_object, phase, frame-range) segments,
                         phase in {approach, grasp, transport, place, release, idle}.
  * PER-FRAME labels   : {frame, active_object, phase, subtask_index, goal_literals_satisfied_count}.
  * stage_timeline.json + a diagnostic overlay grid on the video.

R1Pro observation.state (61-dim) layout (from OmniGibson PROPRIOCEPTION_INDICES["R1Pro"]):
  base_qvel 0:3 | arm_left_qpos 3:10 | arm_left_qvel 10:17 | eef_left_pos 17:20 |
  eef_left_quat 20:24 | gripper_left_qpos 24:26 | gripper_left_qvel 26:28 |
  arm_right_qpos 28:35 | arm_right_qvel 35:42 | eef_right_pos 42:45 | eef_right_quat 45:49 |
  gripper_right_qpos 49:51 | gripper_right_qvel 51:53 | trunk_qpos 53:57 | trunk_qvel 57:61
eef_*_pos/quat are in the ROBOT BASE frame -> transformed to world via camera M + robot2cam.
"""
import argparse
import json
import os
import subprocess
import numpy as np

# ---- R1Pro observation.state slices ----
SL = {
    "eef_left_pos": (17, 20), "eef_left_quat": (20, 24), "gripper_left_qpos": (24, 26),
    "eef_right_pos": (42, 45), "eef_right_quat": (45, 49), "gripper_right_qpos": (49, 51),
}

# ---- tunable thresholds (documented; the report calls out which need tuning to scale) ----
CFG = dict(
    SPEED_WIN=5,          # frames for finite-difference speed
    V_MOVE=0.004,         # m/frame; object considered "moving" (0.12 m/s @30fps)
    R_NEAR=0.20,          # m; EE-world within this of an object => "in contact/carrying"
    GAP_MERGE=45,         # frames; bridge brief EE-proximity dropouts within one carry
    MIN_CARRY_DISP=0.12,  # m; a carry must displace the object at least this much
    MIN_CARRY_DUR=25,     # frames; minimum carry duration
    GRASP_LEN=12,         # frames labelled "grasp" at pickup
    RELEASE_LEN=12,       # frames labelled "release" at drop
    APPROACH_LEN=90,      # frames of "approach" before grasp (~3s @30fps)
    GRIP_OPEN=0.08,       # gripper qpos-sum above this ~ open (secondary signal only)
    STATIONARY_V=0.002,   # m/frame; "placed & stationary"
    FPS=30,
)


def quat_to_R(q):
    """xyzw quaternion -> 3x3 rotation matrix."""
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = np.array([x, y, z, w]) / np.sqrt(n)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def load_episode(labels_path, parquet_path):
    """Return a dict of per-frame arrays on a common frame index (N = parquet frames)."""
    import pandas as pd
    df = pd.read_parquet(parquet_path, columns=[
        "observation.state", "observation.robot2cam_pose.zed_link_camera_0"])
    st = np.stack(df["observation.state"].values)                       # (N,61)
    r2c = np.stack(df["observation.robot2cam_pose.zed_link_camera_0"].values)  # (N,7) pos+quat(xyzw)
    N = len(st)

    rows = [json.loads(l) for l in open(labels_path)]
    LR = len(rows)
    objnames = list(rows[0]["objs"].keys())

    def lidx(i):  # "last-per-frame" consumer mapping
        return min(int((i + 1) * LR / N) - 1, LR - 1)

    P = {o: np.zeros((N, 3)) for o in objnames}
    M = np.zeros((N, 4, 4))
    for i in range(N):
        r = rows[lidx(i)]
        for o in objnames:
            P[o][i] = r["objs"][o]
        M[i] = np.array(r["M"], dtype=float).reshape(4, 4)

    # EE -> world:  T_world_base = T_world_cam @ inv(T_base_cam)
    eef = {}
    for arm in ("left", "right"):
        a, b = SL[f"eef_{arm}_pos"]
        ep = st[:, a:b]
        ew = np.zeros((N, 3))
        for i in range(N):
            Twc = M[i]
            Tbc = np.eye(4)
            Tbc[:3, :3] = quat_to_R(r2c[i, 3:7])
            Tbc[:3, 3] = r2c[i, :3]
            Twb = Twc @ np.linalg.inv(Tbc)
            ew[i] = Twb[:3, :3] @ ep[i] + Twb[:3, 3]
        eef[arm] = ew
    grip = {arm: st[:, SL[f"gripper_{arm}_qpos"][0]:SL[f"gripper_{arm}_qpos"][1]].sum(1)
            for arm in ("left", "right")}
    return dict(N=N, P=P, M=M, eef=eef, grip=grip, objnames=objnames)


def smooth_speed(Po, win):
    N = len(Po)
    d = np.zeros(N)
    for i in range(win, N):
        d[i] = np.linalg.norm(Po[i] - Po[i - win]) / win
    return d


def classify_objects(objnames, task_targets):
    """Split label objects into targets (to relocate) vs references (surfaces/containers)."""
    tgt_cats = task_targets.get("targets", [])
    targets, refs = [], []
    for o in objnames:
        if any(o.startswith(c) for c in tgt_cats):
            targets.append(o)
        else:
            refs.append(o)
    return targets, refs


def detect_carries(ep, targets, cfg):
    """For each target, find its primary carry window and grasp/place/release events."""
    N = ep["N"]
    carries = []
    for o in targets:
        Po = ep["P"][o]
        sp = smooth_speed(Po, cfg["SPEED_WIN"])
        dmin = np.minimum(np.linalg.norm(ep["eef"]["left"] - Po, axis=1),
                          np.linalg.norm(ep["eef"]["right"] - Po, axis=1))
        armidx = (np.linalg.norm(ep["eef"]["right"] - Po, axis=1) <
                  np.linalg.norm(ep["eef"]["left"] - Po, axis=1)).astype(int)  # 0=L,1=R
        near = dmin < cfg["R_NEAR"]
        # contiguous near-runs, merged across short gaps
        runs = []
        i = 0
        while i < N:
            if near[i]:
                j = i
                while j + 1 < N and near[j + 1]:
                    j += 1
                runs.append([i, j])
                i = j + 1
            else:
                i += 1
        merged = []
        for r in runs:
            if merged and r[0] - merged[-1][1] <= cfg["GAP_MERGE"]:
                merged[-1][1] = r[1]
            else:
                merged.append(r[:])
        # keep runs that actually displace the object; pick the largest-displacement one
        best = None
        best_disp = 0.0
        for a, b in merged:
            disp = np.linalg.norm(Po[b] - Po[a])
            if (b - a) >= cfg["MIN_CARRY_DUR"] and disp >= cfg["MIN_CARRY_DISP"] and disp > best_disp:
                best, best_disp = (a, b), disp
        if best is None:
            continue
        a, b = best
        # grasp = first frame in window with sustained motion
        movef = np.where(sp[a:b + 1] > cfg["V_MOVE"])[0]
        g = a + int(movef[0]) if len(movef) else a
        # place = last frame with motion (arrival), release = end of near-window
        p = a + int(movef[-1]) if len(movef) else b
        r = b
        arm = "right" if np.round(armidx[g:p + 1].mean()) == 1 else "left"
        carries.append(dict(obj=o, arm=arm, approach=int(a), grasp=int(g),
                            place=int(p), release=int(r), disp=float(best_disp),
                            speed=sp, dmin=dmin))
    carries.sort(key=lambda c: c["grasp"])
    return carries


def build_timeline(ep, carries, refs, cfg):
    """Per-frame labels + subtask segments + goal-satisfied count."""
    N = ep["N"]
    # per-frame active-object / phase from subtasks (handle overlap: nearest EE wins)
    per_frame = [dict(frame=i, active_object=None, phase="idle", subtask_index=-1,
                      goal_literals_satisfied_count=0) for i in range(N)]

    subtasks = []
    for k, c in enumerate(carries):
        g, p, r = c["grasp"], c["place"], c["release"]
        appr0 = max(0, g - cfg["APPROACH_LEN"])
        # Only clamp to the previous subtask on the SAME arm (bimanual subtasks can
        # be concurrent -- a nested left-arm carry inside a long right-arm carry must
        # NOT be clamped to the right-arm subtask's release).
        for kk in range(k - 1, -1, -1):
            if carries[kk]["arm"] == c["arm"]:
                appr0 = max(appr0, carries[kk]["release"] + cfg["RELEASE_LEN"])
                break
        appr0 = min(appr0, g)  # never past our own grasp
        spans = [
            ("approach", appr0, g),
            ("grasp", g, min(g + cfg["GRASP_LEN"], p)),
            ("transport", min(g + cfg["GRASP_LEN"], p), p),
            ("place", p, r),
            ("release", r, min(r + cfg["RELEASE_LEN"], N)),
        ]
        subtasks.append(dict(subtask_index=k, active_object=c["obj"], arm=c["arm"],
                             approach_start=appr0, grasp_frame=g, place_frame=p,
                             release_frame=r, spans=spans))

    # window length per subtask; on bimanual overlap the SMALLER (nested, transient)
    # window wins its frames -- a short left-arm pick nested inside a long right-arm
    # carry represents the salient action happening "now".
    win_len = [sub["release_frame"] - sub["approach_start"] for sub in subtasks]
    for k, sub in enumerate(subtasks):
        c = carries[k]
        for phase, s, e in sub["spans"]:
            for i in range(int(s), int(e)):
                cur = per_frame[i]
                if cur["subtask_index"] == -1 or win_len[k] < win_len[cur["subtask_index"]]:
                    cur["active_object"] = sub["active_object"]
                    cur["phase"] = phase
                    cur["subtask_index"] = k

    # goal_literals_satisfied_count: a subtask's literal is "satisfied" once released AND
    # the object stays put afterward (not re-grabbed). Monotone step function over frames.
    n_goal = 0
    sat_frames = []
    for sub in subtasks:
        c = carries[sub["subtask_index"]]
        r = sub["release_frame"]
        after = c["speed"][r:]
        stayed = (after < cfg["STATIONARY_V"]).mean() > 0.6 if len(after) else True
        sat_frames.append(r if stayed else N)  # satisfied from release onward
    sat_frames = sorted(sat_frames)
    for i in range(N):
        per_frame[i]["goal_literals_satisfied_count"] = int(sum(1 for sf in sat_frames if i >= sf))

    return per_frame, subtasks


def flatten_segments(per_frame):
    segs = []
    for f in per_frame:
        key = (f["active_object"], f["phase"], f["subtask_index"])
        if segs and segs[-1]["key"] == key:
            segs[-1]["end"] = f["frame"]
        else:
            segs.append(dict(key=key, start=f["frame"], end=f["frame"]))
    out = []
    for s in segs:
        obj, phase, idx = s["key"]
        out.append(dict(start=s["start"], end=s["end"], n_frames=s["end"] - s["start"] + 1,
                        active_object=obj, phase=phase, subtask_index=idx))
    return out


def project(M_i, pw, fx, fy, cx, cy):
    """world point -> pixel (u,v) using p_cam = R.T@(pw - t)."""
    R = M_i[:3, :3]
    t = M_i[:3, 3]
    pc = R.T @ (np.asarray(pw) - t)
    z = -pc[2]
    if z <= 1e-3:
        return None
    u = cx + fx * pc[0] / z
    v = cy - fy * pc[1] / z
    return (float(u), float(v), float(z))


def make_overlay(ep, carries, subtasks, per_frame, video_path, intr, out_png, cfg):
    from PIL import Image, ImageDraw, ImageFont
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    # choose boundary frames: grasp + place (+ release) of each subtask, spread, cap 8
    cand = []
    for sub in subtasks:
        cand.append((sub["grasp_frame"], "grasp", sub))
        cand.append((sub["place_frame"], "place", sub))
        cand.append((min(sub["release_frame"] + 3, ep["N"] - 1), "release", sub))
    cand.sort(key=lambda x: x[0])
    if len(cand) > 8:
        idxs = np.linspace(0, len(cand) - 1, 8).round().astype(int)
        cand = [cand[i] for i in sorted(set(idxs))]
    frames = [c[0] for c in cand]

    # extract the needed frames in ONE ffmpeg pass
    tmpdir = out_png + ".frames"
    os.makedirs(tmpdir, exist_ok=True)
    sel = "+".join(f"eq(n\\,{k})" for k in frames)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", video_path,
         "-vf", f"select='{sel}'", "-vsync", "0",
         os.path.join(tmpdir, "f_%03d.png")],
        check=True)
    got = sorted(os.listdir(tmpdir))
    imgs = [Image.open(os.path.join(tmpdir, g)).convert("RGB") for g in got]

    # annotate each
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        sfont = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default(); sfont = font
    PHASE_COLOR = dict(approach=(80, 160, 255), grasp=(0, 220, 0), transport=(255, 200, 0),
                       place=(255, 120, 0), release=(255, 0, 120), idle=(160, 160, 160))
    cells = []
    for img, (k, boundary, sub) in zip(imgs, cand):
        d = ImageDraw.Draw(img)
        obj = sub["active_object"]
        pf = per_frame[k]
        col = PHASE_COLOR.get(pf["phase"], (200, 200, 200))
        # project active object
        pr = project(ep["M"][k], ep["P"][obj][k], fx, fy, cx, cy)
        if pr:
            u, v, z = pr
            d.ellipse([u - 16, v - 16, u + 16, v + 16], outline=col, width=4)
            d.line([u - 24, v, u + 24, v], fill=col, width=2)
            d.line([u, v - 24, u, v + 24], fill=col, width=2)
            d.text((u + 18, v - 20), obj, fill=col, font=sfont)
        # project carrying EE
        c = carries[sub["subtask_index"]]
        pe = project(ep["M"][k], ep["eef"][c["arm"]][k], fx, fy, cx, cy)
        if pe:
            ue, ve, _ = pe
            d.rectangle([ue - 8, ve - 8, ue + 8, ve + 8], outline=(0, 255, 255), width=3)
            d.text((ue + 10, ve + 6), f"EE-{c['arm'][0]}", fill=(0, 255, 255), font=sfont)
        # caption bar
        d.rectangle([0, 0, img.width, 58], fill=(0, 0, 0))
        d.text((6, 4), f"f{k}  st#{sub['subtask_index']} {boundary.upper()}", fill=col, font=font)
        eed = c["dmin"][k]
        d.text((6, 30), f"phase={pf['phase']}  goalcnt={pf['goal_literals_satisfied_count']}  EEd={eed:.2f}m",
               fill=(230, 230, 230), font=sfont)
        cells.append(img)

    # grid 4 x 2
    ncol = 4
    nrow = int(np.ceil(len(cells) / ncol))
    W = cells[0].width; H = cells[0].height
    grid = Image.new("RGB", (ncol * W, nrow * H), (20, 20, 20))
    for i, im in enumerate(cells):
        grid.paste(im, ((i % ncol) * W, (i // ncol) * H))
    grid.save(out_png)
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--task", default="tidying_bedroom")
    ap.add_argument("--task_targets", required=True)
    ap.add_argument("--intrinsics", required=True)
    ap.add_argument("--video", default=None)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_overlay", default=None)
    ap.add_argument("--out_perframe", default=None)
    args = ap.parse_args()

    tt_all = json.load(open(args.task_targets))
    # task_targets.json may be flat (one task) or keyed by task name
    task_targets = tt_all if "targets" in tt_all else tt_all[args.task]
    intr = json.load(open(args.intrinsics))
    ep = load_episode(args.labels, args.parquet)
    targets, refs = classify_objects(ep["objnames"], task_targets)
    print(f"[objects] targets={targets}")
    print(f"[objects] references(static/surfaces)={refs}")

    carries = detect_carries(ep, targets, CFG)
    per_frame, subtasks = build_timeline(ep, carries, refs, CFG)
    segs = flatten_segments(per_frame)

    n_relocated = len(carries)
    n_target_inst = len(targets)
    n_goal = task_targets.get("n_goal_literals", None)

    print(f"\n[relocation] detected {n_relocated} objects relocated "
          f"(target instances in scene: {n_target_inst}; goal literals: {n_goal})")

    print("\n===== SUBTASK TIMELINE (event summary) =====")
    for sub in subtasks:
        c = carries[sub["subtask_index"]]
        dur = (sub["release_frame"] - sub["approach_start"]) / CFG["FPS"]
        print(f" subtask#{sub['subtask_index']}  obj={sub['active_object']} arm={sub['arm']}  "
              f"approach@{sub['approach_start']} grasp@{sub['grasp_frame']} "
              f"place@{sub['place_frame']} release@{sub['release_frame']}  "
              f"disp={c['disp']:.2f}m  span={dur:.1f}s")

    print("\n===== PER-PHASE SEGMENT TIMELINE (non-idle) =====")
    for s in segs:
        if s["phase"] == "idle":
            continue
        print(f"  [{s['start']:5d}-{s['end']:5d}] ({s['n_frames']:4d}f "
              f"{s['n_frames']/CFG['FPS']:5.1f}s)  st#{s['subtask_index']} "
              f"{s['active_object']:<16} {s['phase']}")

    print("\n===== PER-SUBTASK PHASE DURATIONS (frames / sec) =====")
    for sub in subtasks:
        durs = {}
        for s in segs:
            if s["subtask_index"] == sub["subtask_index"]:
                durs[s["phase"]] = durs.get(s["phase"], 0) + s["n_frames"]
        parts = "  ".join(f"{p}={n}f/{n/CFG['FPS']:.1f}s" for p, n in durs.items())
        print(f"  st#{sub['subtask_index']} {sub['active_object']:<16}: {parts}")

    # ----- write outputs -----
    out = dict(
        task=args.task,
        episode=os.path.basename(args.labels),
        n_frames=ep["N"], fps=CFG["FPS"],
        targets=targets, references=refs,
        n_relocated=n_relocated, n_target_instances=n_target_inst, n_goal_literals=n_goal,
        thresholds=CFG,
        subtasks=[dict(subtask_index=s["subtask_index"], active_object=s["active_object"],
                       arm=s["arm"], approach_start=s["approach_start"],
                       grasp_frame=s["grasp_frame"], place_frame=s["place_frame"],
                       release_frame=s["release_frame"],
                       displacement_m=round(carries[s["subtask_index"]]["disp"], 3))
                  for s in subtasks],
        segments=segs,
    )
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[write] timeline -> {args.out_json}")

    if args.out_perframe:
        with open(args.out_perframe, "w") as f:
            for pf in per_frame:
                f.write(json.dumps(pf) + "\n")
        print(f"[write] per-frame labels -> {args.out_perframe}")

    if args.video and args.out_overlay:
        frames = make_overlay(ep, carries, subtasks, per_frame, args.video, intr,
                              args.out_overlay, CFG)
        print(f"[write] overlay ({len(frames)} frames {frames}) -> {args.out_overlay}")


if __name__ == "__main__":
    main()

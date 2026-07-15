#!/usr/bin/env python
"""100-task grounding-label sweep driver for BEHAVIOR-2026.

Per task (ordered by task id, skipping the 4 empty-target tasks):
  a. fetch that task's 5 smallest-index rawdata episodes from HF (skipped if already local)
  b. point /root/replay_root/2026-challenge-rawdata at the dir holding task-XXXX
  c. run replay_labeled.py per episode -> videos /root/sweep_out/<task>/ + labels
     /root/sweep_labels/<task>/ (success = .done.json, NEVER exit code)
  d. integrity: label_records/2 vs lerobot meta episode length (+-1) -> status json
  e. delete the task's fetched rawdata
  f. every 10 tasks: upload labels + lerobot meta (not videos) to HF
     arif101/behavior2026-artifacts under sweep_100/; videos uploaded at the very end
  g. append one line per task to /root/sweep_100.log

Resumable: tasks whose status json says ok/partial/skip are skipped; episodes with an
existing .done.json are not re-run.

Run inside the behavior conda env via sweep_100.sh (sets OMP/MKL=16, HF env, taskset).
"""

import argparse
import csv
import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

RAW_LOCAL = "/root/rawdata"            # pre-existing pilot rawdata (task-0000); never deleted
RAW_FETCH = "/root/rawdata_sweep"      # transient fetched rawdata; deleted per task
REPLAY_ROOT = "/root/replay_root"
SYMLINK = os.path.join(REPLAY_ROOT, "2026-challenge-rawdata")
OUT_DIR = "/root/sweep_out"
LBL_DIR = "/root/sweep_labels"
ST_DIR = "/root/sweep_status"
LOG_DIR = "/root/sweep_logs"
TASK_LOG = "/root/sweep_100.log"
TARGETS_JSON = "/root/task_targets.json"
CSV_PATH = "/root/BEHAVIOR-1K/datasets/2026-challenge-task-instances/metadata/B100_task_misc.csv"
OG_DIR = "/root/BEHAVIOR-1K/OmniGibson"
REPLAY_LABELED = "/root/probes/replay_labeled.py"
FILES_CACHE = "/root/sweep_cache_repo_files.json"

REPO_RAW = "behavior-1k/2026-challenge-rawdata"
REPO_ART = "arif101/behavior2026-artifacts"
ART_PREFIX = "sweep_100"

EPS_PER_TASK = 5
MIN_FREE_GB = 15
UPLOAD_EVERY = 10

# Measured on radio/trash pilots: ~3150 bytes of rawdata per frame; effective replay ~6.5
# frames/s + ~140s sim boot + save. Timeout = 3x expected, from the actual hdf5 size.
BYTES_PER_FRAME = 3150.0
TIMEOUT_S_PER_FRAME = 2.0
EP_TIMEOUT_MIN_S = 7200
EP_TIMEOUT_MAX_S = 24 * 3600


def free_gb():
    return shutil.disk_usage("/root").free / 2**30


def log_task(line):
    with open(TASK_LOG, "a") as f:
        f.write(line + "\n")
    print("[sweep]", line, flush=True)


def hf_token():
    with open("/root/.hf_token") as f:
        return f.read().strip()


def get_api():
    from huggingface_hub import HfApi

    return HfApi(token=hf_token())


def rawdata_files(api):
    if os.path.exists(FILES_CACHE):
        with open(FILES_CACHE) as f:
            return json.load(f)
    files = list(api.list_repo_files(REPO_RAW, repo_type="dataset"))
    with open(FILES_CACHE, "w") as f:
        json.dump(files, f)
    return files


def pick_episode_paths(files, task_id, n=EPS_PER_TASK):
    """5 smallest-index episode files for task-XXXX."""
    pat = re.compile(rf"^task-{task_id:04d}/episode_(\d+)\.hdf5$")
    cands = []
    for p in files:
        m = pat.match(p)
        if m:
            cands.append((int(m.group(1)), p))
    cands.sort()
    return cands[:n]  # [(demo_id, repo_path), ...]


def local_episode_demo_ids(task_id, n=EPS_PER_TASK):
    """If pilot rawdata already has >= n episodes for this task, use them (no fetch)."""
    d = os.path.join(RAW_LOCAL, f"task-{task_id:04d}")
    if not os.path.isdir(d):
        return None
    ids = sorted(
        int(m.group(1))
        for m in (re.match(r"episode_(\d+)\.hdf5$", f) for f in os.listdir(d))
        if m
    )
    return ids[:n] if len(ids) >= n else None


def set_symlink(target_dir):
    tmp = SYMLINK + ".tmp"
    if os.path.lexists(tmp):
        os.remove(tmp)
    os.symlink(target_dir, tmp)
    os.replace(tmp, SYMLINK)


def fetch_episodes(api, task_id, ep_paths):
    """hf_transfer-download episode files into RAW_FETCH/task-XXXX/."""
    from huggingface_hub import hf_hub_download

    for _, repo_path in ep_paths:
        dest = os.path.join(RAW_FETCH, repo_path)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            continue
        for attempt in range(3):
            try:
                hf_hub_download(
                    repo_id=REPO_RAW,
                    filename=repo_path,
                    repo_type="dataset",
                    local_dir=RAW_FETCH,
                    token=hf_token(),
                )
                break
            except Exception as e:
                print(f"[sweep] fetch retry {attempt + 1} {repo_path}: {e}", flush=True)
                if attempt == 2:
                    raise
                time.sleep(10 * (attempt + 1))


def episode_timeout(task_id, demo_id):
    """Dynamic timeout from the local hdf5 size (episode lengths span 2k..400k frames)."""
    path = os.path.join(SYMLINK, f"task-{task_id:04d}", f"episode_{demo_id:08d}.hdf5")
    try:
        est_frames = os.path.getsize(path) / BYTES_PER_FRAME
    except OSError:
        est_frames = 20000
    return int(min(max(EP_TIMEOUT_MIN_S, 300 + est_frames * TIMEOUT_S_PER_FRAME), EP_TIMEOUT_MAX_S))


def run_episode(task_name, task_id, demo_id):
    """Run replay_labeled.py for one episode. Success = .done.json (exit code ignored:
    teardown segfaults after LABELS_DONE are benign)."""
    labels_out = os.path.join(LBL_DIR, task_name, f"labels_{demo_id}.jsonl")
    done_path = labels_out + ".done.json"
    if os.path.exists(done_path):
        with open(done_path) as f:
            return json.load(f), True
    os.makedirs(os.path.dirname(labels_out), exist_ok=True)
    os.makedirs(os.path.join(LOG_DIR, task_name), exist_ok=True)
    log_path = os.path.join(LOG_DIR, task_name, f"ep_{demo_id}.log")
    timeout_s = episode_timeout(task_id, demo_id)

    cmd = [
        "taskset", "-c", "0-23",
        "python", REPLAY_LABELED,
        "--data_folder", REPLAY_ROOT,
        "--demo_id", str(demo_id),
        "--lerobot_root_dir", os.path.join(OUT_DIR, task_name),
        "--labels_out", labels_out,
        "--targets_json", TARGETS_JSON,
    ]
    # resume only when a valid local dataset exists; resuming into a missing dir makes
    # LeRobot fall back to a HUB lookup (b1k/<task> -> 404 -> teardown segfault).
    _meta = os.path.join(OUT_DIR, task_name, "b1k", task_name, "meta", "info.json")
    import glob as _glob
    _clean_eps = _glob.glob(os.path.join(LBL_DIR, task_name, "*.done.json"))
    if os.path.exists(_meta) and _clean_eps:
        cmd.append("--resume_lerobot")
    else:
        import shutil as _sh
        _stale = os.path.join(OUT_DIR, task_name, "b1k", task_name)
        if os.path.isdir(_stale):
            _sh.rmtree(_stale)  # partial output from a killed first attempt
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = "16"

    with open(log_path, "w") as lf:
        proc = subprocess.Popen(
            cmd, cwd=OG_DIR, env=env, stdout=lf, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
            proc.wait()
            print(f"[sweep] TIMEOUT ep {demo_id} ({task_name})", flush=True)

    if os.path.exists(done_path):
        with open(done_path) as f:
            return json.load(f), False
    return None, False


def meta_lengths(task_name):
    """Episode lengths from the task's LeRobot meta (episodes parquet)."""
    import pandas as pd

    pats = glob.glob(
        os.path.join(OUT_DIR, task_name, "b1k", task_name, "meta", "episodes", "**", "*.parquet"),
        recursive=True,
    )
    if not pats:
        return []
    df = pd.concat([pd.read_parquet(p) for p in sorted(pats)])
    df = df.sort_values("episode_index")
    return [int(x) for x in df["length"].tolist()]


def integrity_check(ep_results, lengths):
    """label_records/2 must be within +-1 of a lerobot meta episode length (and of the
    source num_samples)."""
    checks = {}
    for demo_id, rec in ep_results.items():
        if rec is None:
            checks[demo_id] = {"ok": False, "reason": "no_done_file"}
            continue
        frames = rec["label_records"] / 2.0
        ok_meta = any(abs(frames - L) <= 1.0 for L in lengths) if lengths else False
        ok_src = abs(frames - rec["num_samples"]) <= 1.0
        checks[demo_id] = {
            "ok": bool(ok_meta and ok_src),
            "frames": frames,
            "num_samples": rec["num_samples"],
            "ok_meta": bool(ok_meta),
            "ok_src": bool(ok_src),
        }
    return checks


def upload_checkpoint(api, done_tasks):
    """Per not-yet-uploaded task: upload labels + lerobot meta, then videos, then DELETE the
    local mp4s (videos total ~132GB across the sweep -- exceeds local disk, so they are
    offloaded to HF at every checkpoint; only deleted after their upload succeeded). Always
    refresh status dir + sweep log. Failures are logged, never fatal. Upload state tracked
    per task in ST_DIR/_uploaded.json."""
    state_path = os.path.join(ST_DIR, "_uploaded.json")
    state = {"meta": [], "videos": []}
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
    kw = dict(repo_id=REPO_ART, repo_type="model", token=hf_token())
    try:
        for t in done_tasks:
            lbl = os.path.join(LBL_DIR, t)
            out = os.path.join(OUT_DIR, t)
            if t not in state["meta"]:
                if os.path.isdir(lbl):
                    api.upload_folder(folder_path=lbl, path_in_repo=f"{ART_PREFIX}/sweep_labels/{t}", **kw)
                if os.path.isdir(out):
                    api.upload_folder(
                        folder_path=out,
                        path_in_repo=f"{ART_PREFIX}/sweep_out/{t}",
                        ignore_patterns=["*.mp4"],
                        **kw,
                    )
                state["meta"].append(t)
            if t not in state["videos"] and os.path.isdir(out):
                mp4s = glob.glob(os.path.join(out, "**", "*.mp4"), recursive=True)
                if mp4s:
                    api.upload_folder(
                        folder_path=out,
                        path_in_repo=f"{ART_PREFIX}/sweep_out/{t}",
                        allow_patterns=["**/*.mp4", "*.mp4"],
                        **kw,
                    )
                    for m in mp4s:  # only after successful upload
                        os.remove(m)
                state["videos"].append(t)
            with open(state_path, "w") as f:
                json.dump(state, f)
        api.upload_folder(
            folder_path=ST_DIR, path_in_repo=f"{ART_PREFIX}/sweep_status",
            ignore_patterns=["_uploaded.json"], **kw,
        )
        if os.path.exists(TASK_LOG):
            api.upload_file(
                path_or_fileobj=TASK_LOG, path_in_repo=f"{ART_PREFIX}/sweep_100.log", **kw
            )
        print(f"[sweep] upload checkpoint done ({len(done_tasks)} tasks)", flush=True)
    except Exception as e:
        print(f"[sweep] UPLOAD_FAIL (non-fatal): {type(e).__name__}: {e}", flush=True)


def load_task_list():
    """[(task_id, task_name)] ordered by id; skip empty-target tasks."""
    with open(TARGETS_JSON) as f:
        targets = json.load(f)
    name2id = {}
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name2id[row["Task"]] = int(row["Task ID"])
    tasks, skipped = [], []
    for name, spec in targets.items():
        if name not in name2id:
            print(f"[sweep] WARNING: no task id for '{name}', skipping", flush=True)
            continue
        if not spec.get("targets"):
            skipped.append((name2id[name], name))
            continue
        tasks.append((name2id[name], name))
    tasks.sort()
    skipped.sort()
    return tasks, skipped


def process_task(api, files, task_id, task_name):
    t0 = time.time()
    status_path = os.path.join(ST_DIR, f"{task_name}.json")

    # ---- locate/fetch rawdata ----
    fetched = False
    demo_ids = local_episode_demo_ids(task_id)
    if demo_ids is not None:
        set_symlink(RAW_LOCAL)
    else:
        ep_paths = pick_episode_paths(files, task_id)
        if not ep_paths:
            return {"status": "fail", "reason": "no_rawdata_in_repo"}
        fetch_episodes(api, task_id, ep_paths)
        demo_ids = [d for d, _ in ep_paths]
        set_symlink(RAW_FETCH)
        fetched = True

    # ---- replay + label each episode ----
    ep_results = {}
    for demo_id in demo_ids:
        rec, cached = run_episode(task_name, task_id, demo_id)
        ep_results[demo_id] = rec
        state = "cached" if cached else ("ok" if rec else "FAIL")
        print(f"[sweep] {task_name} ep {demo_id}: {state}", flush=True)

    # ---- integrity ----
    try:
        lengths = meta_lengths(task_name)
    except Exception as e:
        lengths = []
        print(f"[sweep] meta read failed for {task_name}: {e}", flush=True)
    checks = integrity_check(ep_results, lengths)
    n_ok = sum(1 for c in checks.values() if c["ok"])
    total_frames = int(sum(c.get("frames", 0) for c in checks.values() if c["ok"]))

    status = "ok" if n_ok == len(demo_ids) else ("partial" if n_ok > 0 else "fail")
    st = {
        "task": task_name,
        "task_id": task_id,
        "status": status,
        "n_ok": n_ok,
        "n_eps": len(demo_ids),
        "demo_ids": demo_ids,
        "checks": {str(k): v for k, v in checks.items()},
        "meta_lengths": lengths,
        "total_frames": total_frames,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(status_path, "w") as f:
        json.dump(st, f, indent=1)

    # ---- transient rawdata cleanup (keep videos + labels) ----
    if fetched:
        shutil.rmtree(os.path.join(RAW_FETCH, f"task-{task_id:04d}"), ignore_errors=True)

    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default=None, help="comma-separated task names (validation mode)")
    ap.add_argument("--skip-upload", action="store_true")
    args = ap.parse_args()

    for d in (OUT_DIR, LBL_DIR, ST_DIR, LOG_DIR, RAW_FETCH):
        os.makedirs(d, exist_ok=True)

    api = get_api()
    files = rawdata_files(api)
    tasks, skipped = load_task_list()

    # log empty-target skips once
    for tid, name in skipped:
        sp = os.path.join(ST_DIR, f"{name}.json")
        if not os.path.exists(sp):
            with open(sp, "w") as f:
                json.dump({"task": name, "task_id": tid, "status": "skip", "reason": "empty_targets"}, f)
            log_task(f"TASK_{tid}_{name}_SKIP empty_targets")

    if args.only:
        keep = set(args.only.split(","))
        tasks = [t for t in tasks if t[1] in keep]

    print(f"[sweep] {len(tasks)} tasks to process, free disk {free_gb():.1f} GB", flush=True)

    done_since_upload = 0
    completed_tasks = []  # tasks with any artifacts to upload (ok/partial/fail-with-files)
    for tid, name in tasks:
        status_path = os.path.join(ST_DIR, f"{name}.json")
        if os.path.exists(status_path):
            with open(status_path) as f:
                prev = json.load(f)
            if prev.get("status") in ("ok", "partial", "skip"):
                print(f"[sweep] resume-skip {name} ({prev.get('status')})", flush=True)
                completed_tasks.append(name)
                continue

        if free_gb() < MIN_FREE_GB:
            log_task(f"SWEEP_ABORT_DISK free={free_gb():.1f}GB")
            sys.exit(3)

        try:
            st = process_task(api, files, tid, name)
        except Exception as e:
            st = {"status": "fail", "reason": f"{type(e).__name__}: {e}"}
            with open(status_path, "w") as f:
                json.dump({"task": name, "task_id": tid, **st}, f)

        if st["status"] in ("ok", "partial"):
            log_task(f"TASK_{tid}_{name}_OK frames={st.get('total_frames', 0)} eps={st.get('n_ok')}/{st.get('n_eps')}")
        else:
            log_task(f"TASK_{tid}_{name}_FAIL {st.get('reason', 'integrity_or_replay_failed')}")
        completed_tasks.append(name)

        done_since_upload += 1
        if not args.skip_upload and done_since_upload >= UPLOAD_EVERY:
            upload_checkpoint(api, completed_tasks)
            done_since_upload = 0

    if not args.skip_upload:
        upload_checkpoint(api, completed_tasks)

    log_task("SWEEP_DONE")


if __name__ == "__main__":
    main()

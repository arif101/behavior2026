"""Fleet driver: fetch + labels-only playback across many tasks, N processes in parallel.

Per task it (1) fetches the raw HDF5s it needs, then (2) runs replay_poses_batch.py once — a
single env load amortised over that task's episodes.

Measured on turning_on_radio (RTX PRO 4000): env setup 96 s, playback ~235 s/episode. So a
100-episode task is ~6.6 h single-process; 24 tasks at 4-way parallel is ~40 h, at 8-way ~20 h.
GPU assignment is round-robin over CUDA_VISIBLE_DEVICES.

Resumable: replay_poses_batch.py skips episodes whose output JSON already exists, so re-running
after an interruption costs only the env reloads.

Usage:
  python pose_fleet.py --tasks_json task_objects.json --n_episodes 100 --parallel 4 \
      --data_folder /root/replay_root --out_root /root/poses --gpus 0,1,2,3
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time

BATCH = os.environ.get("BATCH_SCRIPT", "/root/replay_poses_batch.py")
PY = os.environ.get("PYBIN", "/root/miniconda3/envs/behavior/bin/python")
OG_DIR = os.environ.get("OG_DIR", "/root/bw/BEHAVIOR-1K/OmniGibson")


def fetch(task_idx: int, demo_ids: list[int], data_folder: str, token_path: str) -> None:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import hf_hub_download

    tok = open(token_path).read().strip()
    target = os.path.join(data_folder, "2026-challenge-rawdata")
    os.makedirs(target, exist_ok=True)
    for d in demo_ids:
        rel = f"task-{task_idx:04d}/episode_{d:08d}.hdf5"
        if os.path.exists(os.path.join(target, rel)):
            continue
        hf_hub_download("behavior-1k/2026-challenge-rawdata", rel, repo_type="dataset",
                        local_dir=target, token=tok)


def run_task(task: str, spec: dict, a, gpu: int, results: dict, lock: threading.Lock) -> None:
    demo_ids = sorted(spec["episodes"])[: a.n_episodes]
    out_dir = os.path.join(a.out_root, task)
    os.makedirs(out_dir, exist_ok=True)
    log = os.path.join(out_dir, "run.log")

    t0 = time.time()
    try:
        fetch(spec["task_index"], demo_ids, a.data_folder, a.token)
    except Exception as e:  # noqa: BLE001
        with lock:
            results[task] = f"FETCH_FAILED {type(e).__name__}: {e}"
        return

    # THREAD CAP — without this each Isaac process spawns ~425 threads. Measured: 8 procs on a
    # 112-core box drove load average to 373 with GPU utilisation at 0% and ZERO episodes produced
    # in 50 minutes; 4 procs on 48 cores gave load 86. The job is CPU-thrash-bound, not GPU-bound,
    # so parallelism must be sized by cores-per-process, never by VRAM (VRAM said 8 would fit).
    nthreads = str(max(1, a.threads))
    env = dict(os.environ,
               CUDA_VISIBLE_DEVICES=str(gpu),
               XDG_RUNTIME_DIR=f"/tmp/xdg{gpu}",
               OMP_NUM_THREADS=nthreads,
               MKL_NUM_THREADS=nthreads,
               OPENBLAS_NUM_THREADS=nthreads,
               NUMEXPR_NUM_THREADS=nthreads)
    os.makedirs(env["XDG_RUNTIME_DIR"], exist_ok=True)
    cmd = [PY, BATCH,
           "--data_folder", a.data_folder, "--task", task,
           "--episodes", ",".join(str(d) for d in demo_ids),
           "--objects", ",".join(spec["objects"]),
           "--out_dir", out_dir]
    if a.max_steps:
        cmd += ["--max_steps", str(a.max_steps)]
    with open(log, "w") as lf:
        p = subprocess.run(cmd, cwd=OG_DIR, env=env, stdout=lf, stderr=subprocess.STDOUT)

    # Judge by produced files, never the exit code: og.shutdown() teardown segfaults AFTER the
    # data is written are benign (same contract as replay_labeled.py).
    n = len([f for f in os.listdir(out_dir) if f.startswith("ep") and f.endswith(".json")])
    with lock:
        results[task] = f"{n}/{len(demo_ids)} episodes in {(time.time()-t0)/3600:.2f}h rc={p.returncode}"
    print(f"[gpu{gpu}] {task}: {results[task]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks_json", required=True)
    ap.add_argument("--n_episodes", type=int, default=100)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--data_folder", default="/root/replay_root")
    ap.add_argument("--out_root", default="/root/poses")
    ap.add_argument("--token", default="/root/.hf_token")
    ap.add_argument("--only", default="", help="comma-separated subset of task names")
    ap.add_argument("--threads", type=int, default=8,
                    help="OMP/MKL threads per process; total should stay well under nproc")
    ap.add_argument("--max_steps", type=int, default=0,
                    help="truncate playback to this many steps (0 = full episode). Tier-1 tasks "
                         "run 6k-16k steps vs radio's 1957; truncating is the biggest cost lever "
                         "and the navigation phase we care about is early in the episode.")
    a = ap.parse_args()

    specs = json.load(open(a.tasks_json))
    if a.only:
        keep = set(a.only.split(","))
        specs = {k: v for k, v in specs.items() if k in keep}
    gpus = [int(g) for g in a.gpus.split(",")]

    print(f"fleet: {len(specs)} tasks x {a.n_episodes} eps, {a.parallel} parallel on gpus {gpus}",
          flush=True)

    results: dict[str, str] = {}
    lock = threading.Lock()
    sem = threading.Semaphore(a.parallel)
    threads = []

    for i, (task, spec) in enumerate(sorted(specs.items())):
        def worker(task=task, spec=spec, gpu=gpus[i % len(gpus)]):
            with sem:
                run_task(task, spec, a, gpu, results, lock)

        t = threading.Thread(target=worker, daemon=False)
        t.start()
        threads.append(t)
        time.sleep(2)  # stagger Isaac starts; simultaneous launches contend badly

    for t in threads:
        t.join()

    print("\n=== FLEET SUMMARY ===", flush=True)
    for k in sorted(results):
        print(f"  {k:44s} {results[k]}", flush=True)
    json.dump(results, open(os.path.join(a.out_root, "fleet_summary.json"), "w"), indent=1)
    print("FLEET_DONE", flush=True)


if __name__ == "__main__":
    main()

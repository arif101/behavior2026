"""Observation-aliasing readout on OUR data (TEMPORAL_4D_SWEEP §4 null-result protocol, AliasBench criterion): for Q
sampled rows, the nearest neighbour by CURRENT head gist (cosine, other episodes only) — how different is its correct
action chunk (next 32 actions, per-dim standardized)? Then the same with the K-slot HISTORY stack [g_t, g_t-32, ..., g_t-256]
as the embedding. If history-NN chunks are closer than current-NN chunks, temporal context disambiguates the action and the
channel is demandable by BC; if both ≈ random-pair distance the observation carries no action information at this
resolution; if current-NN ≈ history-NN ≪ random, aliasing is not the binding failure.
  python aliasing_rate.py /root/b1k_radio_mix_full [Q=20000]
"""
import glob, sys, time, numpy as np, pyarrow.parquet as pq
root = sys.argv[1]; Q = int(sys.argv[2]) if len(sys.argv) > 2 else 20000; K, S, H = 8, 32, 32
t0 = time.time(); rng = np.random.default_rng(0)
files = sorted(glob.glob(root + "/data/**/*.parquet", recursive=True))
G = []; A = []; EP = []; FR = []; ST = []
for f in files:
    t = pq.read_table(f, columns=["gist_head", "action", "episode_index", "frame_index", "stage_v2"])
    G.append(np.asarray(t.column("gist_head").to_pylist(), np.float16)); A.append(np.asarray(t.column("action").to_pylist(), np.float32))
    EP.append(np.asarray(t.column("episode_index").to_pylist())); FR.append(np.asarray(t.column("frame_index").to_pylist())); ST.append(np.asarray(t.column("stage_v2").to_pylist()).reshape(-1))
G = np.concatenate(G).astype(np.float32); A = np.concatenate(A); EP = np.concatenate(EP); FR = np.concatenate(FR); ST = np.concatenate(ST); N = len(G)
print(f"loaded {N} rows, {G.shape[1]}-D gists (fp16 -> fp32), {time.time()-t0:.0f}s", flush=True)
# episode row ranges (rows are contiguous per episode, frame-ordered)
starts = np.flatnonzero(np.r_[True, EP[1:] != EP[:-1]]); ends = np.r_[starts[1:], N]; ep_of_row = np.repeat(np.arange(len(starts)), ends - starts)
# standardized action chunks (next H actions within the episode; clamp at the episode end)
astd = A.std(0) + 1e-6; An = (A - A.mean(0)) / astd
def chunk_rows(rows):
    out = np.zeros((len(rows), H, A.shape[1]), np.float32)
    for i, r in enumerate(rows):
        e = ep_of_row[r]; end = ends[e]; idx = np.minimum(np.arange(r, r + H), end - 1); out[i] = An[idx]
    return out.reshape(len(rows), -1)
def hist_rows(rows):   # [len, K+1] row indices of the history slots (clamped to the episode start)
    e = ep_of_row[rows]; st = starts[e]
    return np.stack([np.maximum(rows - (K - k) * S, st) for k in range(K)] + [rows], axis=1)
# memory guard (the box shares a 250 GB cgroup with the trainer): random-project each gist 2048 -> 128 (JL, fixed
# Gaussian) for BOTH searches; cosine on the projections approximates cosine on the originals.
Rp = rng.standard_normal((G.shape[1], 128)).astype(np.float32) / np.sqrt(128)
Gp = (G @ Rp).astype(np.float32); del G
Gn = Gp / (np.linalg.norm(Gp, axis=1, keepdims=True) + 1e-6)
q = rng.choice(N, size=min(Q, N), replace=False)
# ---- current-gist NN (exclude the query's own episode)
def nn_search(qemb, emb, qrows, blk=2000):
    best = np.zeros(len(qrows), np.int64)
    for s0 in range(0, len(qrows), blk):
        sims = qemb[s0:s0 + blk] @ emb.T                     # [b, N]
        same = ep_of_row[qrows[s0:s0 + blk]][:, None] == ep_of_row[None, :]
        sims[same] = -9.0
        best[s0:s0 + blk] = np.argmax(sims, axis=1)
    return best
t1 = time.time(); nn_cur = nn_search(Gn[q], Gn, q); print(f"current-gist NN done {time.time()-t1:.0f}s", flush=True)
# ---- history-stack NN: embedding = concat of the K+1 slot gists (normalized per slot)
Hs = hist_rows(np.arange(N)); Gh = Gn[Hs].reshape(N, -1) / np.sqrt(K + 1)
t1 = time.time(); nn_hist = nn_search(Gh[q], Gh, q); print(f"history-stack NN done {time.time()-t1:.0f}s", flush=True)
cq = chunk_rows(q); d_cur = np.linalg.norm(cq - chunk_rows(nn_cur), axis=1); d_hist = np.linalg.norm(cq - chunk_rows(nn_hist), axis=1)
rnd = rng.choice(N, size=len(q)); d_rand = np.linalg.norm(cq - chunk_rows(rnd), axis=1)
# same-episode temporal neighbour (t+1) as the "best case" floor
nxt = np.minimum(q + 1, ends[ep_of_row[q]] - 1); d_next = np.linalg.norm(cq - chunk_rows(nxt), axis=1)
pc = lambda x: np.percentile(x, [25, 50, 75]).round(2).tolist()
print(f"chunk distance (standardized, H={H}): random pairs p25/50/75 {pc(d_rand)} | current-gist NN {pc(d_cur)} | history-stack NN {pc(d_hist)} | next-frame floor {pc(d_next)}")
med = np.median(d_rand)
print(f"aliasing rate (NN chunk distance > 0.5 x median random): current-gist {100*(d_cur > 0.5*med).mean():.1f}%  history-stack {100*(d_hist > 0.5*med).mean():.1f}%")
print(f"NN from a different stage_v2: current-gist {100*(ST[nn_cur] != ST[q]).mean():.1f}%  history-stack {100*(ST[nn_hist] != ST[q]).mean():.1f}%")
for c in range(4):
    m = ST[q] == c
    if m.any(): print(f"  stage {c}: n={m.sum()} current-NN d p50 {np.median(d_cur[m]):.2f} history-NN d p50 {np.median(d_hist[m]):.2f} random p50 {np.median(d_rand[m]):.2f}")
print(f"flow-loss floor proxy (||a-a'||^2 / 4 per dim, standardized): current-NN {np.mean(d_cur**2)/4/cq.shape[1]:.4f} history-NN {np.mean(d_hist**2)/4/cq.shape[1]:.4f} random {np.mean(d_rand**2)/4/cq.shape[1]:.4f}")
print(f"ALIASING_RATE_DONE {time.time()-t0:.0f}s")

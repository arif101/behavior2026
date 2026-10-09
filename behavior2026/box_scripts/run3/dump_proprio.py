"""mix parquet -> proprio.npy [N, 61] + index.npy [N, 2] (episode_index, frame_index) for the FK precompute."""
import glob, numpy as np, pyarrow.parquet as pq
files = sorted(glob.glob("/root/mixes/mix_full/data/**/*.parquet", recursive=True))
P = []; I = []
for f in files:
    t = pq.read_table(f, columns=["observation.state", "episode_index", "frame_index"]).to_pandas()
    P.append(np.stack(t["observation.state"].values).astype(np.float32)); I.append(np.stack([t["episode_index"].values, t["frame_index"].values], 1))
P = np.concatenate(P); I = np.concatenate(I)
np.save("/root/fk/proprio.npy", P); np.save("/root/fk/index.npy", I.astype(np.int64)); print("DUMPED", P.shape, I.shape)

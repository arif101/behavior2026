"""Hand->rail distance at the moment the RIGHT gripper command closes, in the training mix (human demos = episodes with the
longest lengths / source map). Uses action[22] (right gripper cmd) and target_points_v2[3:6] (rail/button - right EE, base frame)."""
import os, glob, numpy as np, pyarrow.parquet as pq
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
from huggingface_hub import snapshot_download
tok = open("/root/.hf_token").read().strip()
root = snapshot_download("arif101/b26-run3-mixes", repo_type="dataset", token=tok, local_dir="/root/mixes", allow_patterns=["mix_full/data/**", "mix_full/meta/**"])
files = sorted(glob.glob("/root/mixes/mix_full/data/**/*.parquet", recursive=True)); print("parquet files", len(files), flush=True)
A=[];T=[];E=[];S=[]
for f in files:
    t = pq.read_table(f, columns=["action","target_points_v2","episode_index","stage_v2"]).to_pandas()
    A.append(np.stack(t["action"].values)); T.append(np.stack(t["target_points_v2"].values)); E.append(t["episode_index"].values); S.append(t["stage_v2"].values)
A=np.concatenate(A); T=np.concatenate(T); E=np.concatenate(E); S=np.concatenate(S)
gR=A[:,22]; dR=np.linalg.norm(T[:,3:6],axis=1)   # right EE -> stage target (rail before lift)
print("grip cmd R quantiles", np.round(np.percentile(gR,[1,10,50,90,99]),2), "frames", len(gR))
mid=(np.percentile(gR,99)+np.percentile(gR,1))/2
close_d=[]; close_stage=[]; n_eps=0
for e in np.unique(E):
    m=E==e; g=gR[m]; d=dR[m]; s=S[m]
    c=g<mid
    if not c.any() or c[0]: continue
    i=int(np.argmax(c)); close_d.append(d[i]); close_stage.append(s[i]); n_eps+=1
close_d=np.array(close_d)
print(f"episodes with a right-gripper close: {n_eps}; hand->rail distance at first close: p10/50/90 = {np.round(np.percentile(close_d,[10,50,90]),3)} m; frac < 0.06 m: {(close_d<0.06).mean():.2f}; frac > 0.15 m: {(close_d>0.15).mean():.2f}")
print("stage at first close:", np.bincount(np.array(close_stage).astype(int), minlength=4).tolist())

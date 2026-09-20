"""Is the close triggered by 'hand stopped' (proprio/velocity shortcut) rather than 'rail between fingers'?
Policy: approach speed (d dist/dt) in the 30 steps before the close cmd vs during the reach. Demos: same at the human close."""
import json, glob, numpy as np, pyarrow.parquet as pq
print("== POLICY (full ckpt, instrumented) ==")
for d in sorted(glob.glob("/root/run3_eval/full_grip/run_*"), key=lambda x:int(x.split("_")[-1])):
    try: A=[json.loads(l) for l in open(d+"/action_log.jsonl")]; st=json.load(open(d+"/affordance_wrapper_stats.json"))
    except Exception: continue
    if "grip_R" not in A[0]: continue
    gR=np.array([a["grip_R"] for a in A]); dR=np.array(st.get("dist_R_true_full") or []); n=min(len(gR),len(dR)); gR,dR=gR[:n],dR[:n]
    if n<100: continue
    v=-np.gradient(dR)*30  # m/s toward the rail (positive = approaching)
    mid=(gR.max()+gR.min())/2; closed=gR<mid
    if not closed.any(): print(d.split("/")[-1], "never closed"); continue
    c=int(np.argmax(closed)); reach=slice(max(0,c-300),max(0,c-30)); pre=slice(max(0,c-30),c)
    print(f"{d.split('/')[-1]}: close@{c} dist={dR[c]:.3f} | approach speed 300..30 steps before: {v[reach].mean():+.3f} m/s | last 30 steps before close: {v[pre].mean():+.3f} m/s | min dist before close {dR[:c].min():.3f}")
print("== DEMOS (mix_full) ==")
files=sorted(glob.glob("/root/mixes/mix_full/data/**/*.parquet", recursive=True)); A=[];T=[];E=[]
for f in files:
    t=pq.read_table(f, columns=["action","target_points_v2","episode_index"]).to_pandas(); A.append(np.stack(t["action"].values)); T.append(np.stack(t["target_points_v2"].values)); E.append(t["episode_index"].values)
A=np.concatenate(A); T=np.concatenate(T); E=np.concatenate(E); gR=A[:,22]; dR=np.linalg.norm(T[:,3:6],axis=1)
mid=0.0; sp_pre=[]; sp_reach=[]; stalls_no_close=0; eps=0
for e in np.unique(E):
    m=np.where(E==e)[0]; g=gR[m]; d=dR[m]; c=g<mid
    if not c.any() or c[0] or len(m)<400: continue
    i=int(np.argmax(c)); v=-np.gradient(d)*30
    sp_pre.append(v[max(0,i-30):i].mean()); sp_reach.append(v[max(0,i-300):max(0,i-30)].mean()); eps+=1
    # negatives: windows where the hand is nearly stationary (|v|<0.02 m/s over 30 steps) with 0.15<dist<0.35 and gripper open, before the close
    for s in range(0, i-30, 30):
        w=slice(s,s+30)
        if abs(v[w]).mean()<0.02 and 0.15<d[w].mean()<0.35 and (g[w]>mid).all(): stalls_no_close+=1; break
print(f"demos n={eps}: approach speed 300..30 before close {np.mean(sp_reach):+.3f} m/s; last 30 before close {np.mean(sp_pre):+.3f} m/s (p10/50/90 {np.round(np.percentile(sp_pre,[10,50,90]),3)})")
print(f"demos with a STATIONARY OPEN-GRIPPER window at 0.15-0.35 m before the close (the missing negative): {stalls_no_close}/{eps}")

import json, glob, numpy as np, sys
for d in sorted(glob.glob("/root/run3_eval/full_grip/run_*"), key=lambda x:int(x.split("_")[-1])):
    try:
        A=[json.loads(l) for l in open(d+"/action_log.jsonl")]; st=json.load(open(d+"/affordance_wrapper_stats.json"))
    except Exception as e: print(d, "skip", e); continue
    if "grip_R" not in A[0]: print(d, "no grip channel"); continue
    gR=np.array([a["grip_R"] for a in A]); gL=np.array([a["grip_L"] for a in A])
    dR=np.array(st.get("dist_R_true_full") or []); dL=np.array(st.get("dist_L_true_full") or []); qR=np.array(st.get("grip_qR_series") or []); qL=np.array(st.get("grip_qL_series") or [])
    n=min(len(gR), len(dR), len(qR)); gR,dR,qR=gR[:n],dR[:n],qR[:n]; gL,dL,qL=gL[:n],dL[:n],qL[:n]
    run=d.split("_")[-1]
    print(f"run {run}: n={n} grasp={st.get('grasp')} minR_true={dR.min():.3f}@{int(np.argmin(dR))} minL_true={dL.min():.3f}@{int(np.argmin(dL))}")
    print(f"  cmd grip_R: range [{gR.min():.2f},{gR.max():.2f}] p10/50/90={np.round(np.percentile(gR,[10,50,90]),2)} | qpos R range [{qR.min():.3f},{qR.max():.3f}]")
    print(f"  cmd grip_L: range [{gL.min():.2f},{gL.max():.2f}] p10/50/90={np.round(np.percentile(gL,[10,50,90]),2)} | qpos L range [{qL.min():.3f},{qL.max():.3f}]")
    # timeline every 200 steps: distR, cmdR, qR
    print("  step: distR cmdR qR | distL cmdL qL")
    for s in range(0,n,200):
        w=slice(s,min(n,s+200)); print(f"  {s:4d}: {dR[w].mean():.2f} {gR[w].mean():+.2f} {qR[w].mean():.3f} | {dL[w].mean():.2f} {gL[w].mean():+.2f} {qL[w].mean():.3f}")
    # close events: cmd below midpoint
    mid=(gR.max()+gR.min())/2
    closed=gR<mid
    if closed.any():
        first=int(np.argmax(closed)); print(f"  first R close cmd at step {first} (distR_true then {dR[first]:.3f} m); frac steps closed {closed.mean():.2f}; distR at closest {dR.min():.3f}; cmd at closest {gR[int(np.argmin(dR))]:+.2f}")
    else: print("  R never commanded closed")

import glob, numpy as np, pyarrow.parquet as pq
files=sorted(glob.glob("/root/mixes/mix_full/data/**/*.parquet", recursive=True)); A=[];T=[];E=[];S=[]
for f in files:
    t=pq.read_table(f, columns=["action","target_points_v2","episode_index","stage_v2"]).to_pandas(); A.append(np.stack(t["action"].values)); T.append(np.stack(t["target_points_v2"].values)); E.append(t["episode_index"].values); S.append(t["stage_v2"].values)
A=np.concatenate(A); T=np.concatenate(T); E=np.concatenate(E); S=np.concatenate(S)
gR=A[:,22]; rail=T[:,3:6]                       # rail - EE_R (base frame) while stage<=1
button=rail-np.array([0,0,0.141])               # button - EE_R
dRail=np.linalg.norm(rail,axis=1); dBut=np.linalg.norm(button,axis=1)
cl_rail=[];cl_but=[];min_but=[]
for e in np.unique(E):
    m=np.where(E==e)[0]; g=gR[m]; c=g<0
    if not c.any() or c[0] or len(m)<400: continue
    i=int(np.argmax(c)); 
    if S[m][i]>1: continue   # only closes before lift (rail-adjusted target valid)
    cl_rail.append(dRail[m][i]); cl_but.append(dBut[m][i]); min_but.append(dBut[m][:i+1].min())
cl_rail=np.array(cl_rail); cl_but=np.array(cl_but); min_but=np.array(min_but)
print(f"n={len(cl_but)} demo closes (pre-lift). EE->RAIL at close p10/50/90 {np.round(np.percentile(cl_rail,[10,50,90]),3)} | EE->BUTTON at close p10/50/90 {np.round(np.percentile(cl_but,[10,50,90]),3)} | min EE->BUTTON before close p10/50/90 {np.round(np.percentile(min_but,[10,50,90]),3)}")
# vertical component: is the EE above the button at close?
z=[]; 
for e in np.unique(E):
    m=np.where(E==e)[0]; g=gR[m]; c=g<0
    if not c.any() or c[0] or len(m)<400: continue
    i=int(np.argmax(c)); z.append(-button[m][i][2])   # EE_z - button_z
print("EE height above button at close p10/50/90:", np.round(np.percentile(z,[10,50,90]),3))

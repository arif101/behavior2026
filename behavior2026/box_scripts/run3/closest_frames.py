import json, glob, numpy as np, subprocess, os
os.makedirs("/root/film_frames/closest", exist_ok=True)
picks = [("full_oracle",7),("full_ptoff",4),("full_ptoff",7),("full_ptoff",8),("full_ptoff",10),("a4_oracle",2),("a4_oracle",5),("full_hist_off",7)]
for arm, run in picks:
    d=f"/root/run3_eval/{arm}/run_{run}"; st=json.load(open(d+"/affordance_wrapper_stats.json"))
    ser=st.get("dist_R_true_series") or []; n=st["n_steps"]
    if not ser: print(arm, run, "no series"); continue
    i=int(np.argmin(ser)); step=int(i*10)  # series is [::10] of per-step
    v=glob.glob(d+"/videos/*.mp4")[0]
    frames=[max(0,step-150), max(0,step-60), step, min(n-1,step+60), min(n-1,step+200)]
    sel="+".join(f"eq(n\\,{f})" for f in frames)
    out=f"/root/film_frames/closest/{arm}_r{run}_s{step}.jpg"
    subprocess.run(["ffmpeg","-v","error","-y","-i",v,"-vf",f"select={sel},crop=224:448:0:0,scale=224:-1,tile=5x1","-vsync","vfr",out])
    print(arm, run, "closest step", step, "distR", ser[i], "frames", frames, "->", out)

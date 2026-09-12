#!/usr/bin/env python
"""Post-grasp failure quantification: does grasping lead toward the button press?
Re-run as rollouts accumulate: python postgrasp_analysis.py a0 a4 a2"""
import glob,json,sys,numpy as np
arms = sys.argv[1:] or ['a0','a4','a2']
for arm in arms:
    fs=glob.glob(f'/root/eval_{arm}_n25/json/*.json')
    if not fs: continue
    r=[json.load(open(f)) for f in fs]
    g=[x for x in r if x['grasp_completed']]; ng=[x for x in r if not x['grasp_completed']]
    def med(xs,k): v=[x[k] for x in xs]; return (np.median(v),min(v)) if v else (None,None)
    # KEY: does the EE ever reach the button (<~5cm) in ANY rollout? (button press needs contact)
    eb_all=[x['ee_button_min_m'] for x in r]
    near_button=sum(1 for e in eb_all if e<0.05)
    print(f"== {arm}: {len(r)} rollouts, grasp {len(g)}/{len(r)}, success {sum(x['success'] for x in r)}/{len(r)}")
    if g:  print(f"   grasp    : fingertip med {med(g,'fingertip_min_m')[0]:.3f}  ee_button med {med(g,'ee_button_min_m')[0]:.3f} (best {med(g,'ee_button_min_m')[1]:.3f})")
    if ng: print(f"   no-grasp : fingertip med {med(ng,'fingertip_min_m')[0]:.3f}  ee_button med {med(ng,'ee_button_min_m')[0]:.3f}")
    print(f"   rollouts with EE within 5cm of BUTTON (press contact): {near_button}/{len(r)}   min ee_button over ALL: {min(eb_all):.3f} m")

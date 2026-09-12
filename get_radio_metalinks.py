"""Enumerate the radio's metalinks (grasp? handle? toggle?) + their offsets in the radio-root frame.
Settles the grasp-stage target: a real grasp metalink vs the radio centroid. No rollout."""
import numpy as np, json, os
from omegaconf import OmegaConf
from omnigibson.macros import gm
from omnigibson.eval.evaluator import Evaluator, resolve_instance_ids
gm.HEADLESS = True
inst = resolve_instance_ids("turning_on_radio",[0],mode="public_test")[0]
rc = OmegaConf.load("/root/bw/BEHAVIOR-1K/OmniGibson/omnigibson/eval/r1pro.yaml")
cfg = OmegaConf.create({"env_wrapper":{"_target_":"omnigibson.envs.EnvironmentWrapper"},
  "policy_name":"local","model":{"_target_":"omnigibson.eval.policies.LocalPolicy","action_dim":None},
  "headless":True,"partial_scene_load":True,"max_steps":1,"write_video":False,
  "mode":"public_test","seed":0,"task":{"name":"turning_on_radio"},"robot":rc})
def q2R(q):
    x,y,z,w=q; return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
with Evaluator(cfg) as ev:
    ev.reset(); ev.load_task_instance(int(inst))
    radio = next(o for o in ev.env.scene.objects if "radio" in o.name.lower())
    rp,rq = radio.get_position_orientation(); rp=np.asarray(rp,float).reshape(3); rq=np.asarray(rq,float).reshape(4); Rt=q2R(rq).T
    out={"radio_aabb_extent": np.asarray(radio.aabb_extent,float).tolist(), "metalinks":{}}
    for lname,link in radio.links.items():
        if "meta" in lname.lower():
            try:
                lp=np.asarray(link.get_position_orientation()[0],float).reshape(3)
                out["metalinks"][lname]={"offset_root_frame": (Rt@(lp-rp)).round(4).tolist()}
            except Exception as e: out["metalinks"][lname]={"err":str(e)[:60]}
    out["has_grasp_metalink"]= any(any(k in n.lower() for k in ("grasp","handle","graspable")) for n in out["metalinks"])
    json.dump(out,open("/root/radio_metalinks.json","w"),indent=2)
    print("METALINKS", json.dumps(out))
os._exit(0)

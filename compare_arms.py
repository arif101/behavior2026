#!/usr/bin/env python
"""Seed-paired comparison of two eval arms (rollout_id matched)."""
import sys, json, glob, os
import numpy as np
def load(d):
    rows={}
    for f in glob.glob(os.path.join(d,"json","*.json")):
        r=json.load(open(f)); rows[r["rollout_id"]]=r
    return rows
A=load(sys.argv[1]); B=load(sys.argv[2])
na,nb=sys.argv[3],sys.argv[4]
ids=sorted(set(A)&set(B))
ga=sum(A[i]["grasp_completed"] for i in ids); gb=sum(B[i]["grasp_completed"] for i in ids)
sa=sum(A[i]["success"] for i in ids); sb=sum(B[i]["success"] for i in ids)
# McNemar paired: b = B grasp & not A ; c = A grasp & not B
b=sum(1 for i in ids if B[i]["grasp_completed"] and not A[i]["grasp_completed"])
c=sum(1 for i in ids if A[i]["grasp_completed"] and not B[i]["grasp_completed"])
fa=[A[i]["fingertip_min_m"] for i in ids]; fb=[B[i]["fingertip_min_m"] for i in ids]
closer=sum(1 for i in ids if B[i]["fingertip_min_m"] < A[i]["fingertip_min_m"])
print(f"paired rollouts: n={len(ids)}")
print(f"GRASP  {na}={ga}/{len(ids)}  {nb}={gb}/{len(ids)}   delta={gb-ga}")
print(f"  McNemar discordant: {nb}-only={b}  {na}-only={c}")
print(f"SUCCESS {na}={sa}/{len(ids)}  {nb}={sb}/{len(ids)}")
print(f"fingertip_min median  {na}={np.median(fa):.3f}  {nb}={np.median(fb):.3f}")
print(f"fingertip_min best    {na}={min(fa):.3f}  {nb}={min(fb):.3f}")
print(f"{nb} reached closer than {na} in {closer}/{len(ids)} paired rollouts")
# exact binomial sign p (two-sided) on fingertip improvement
from math import comb
k=closer; n=len(ids)
p=sum(comb(n,j) for j in range(0,n+1) if abs(j-n/2)>=abs(k-n/2))/2**n
print(f"sign-test p (fingertip closer, two-sided) = {p:.3f}")

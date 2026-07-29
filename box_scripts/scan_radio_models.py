"""Which radio model(s) appear across the 200 demo episodes / 10 instances?

Decides whether the togglebutton offset is ONE constant (all instances share a model+scale) or
per-instance. Reads the rescue pose JSONs' frame-0 object registries.
"""

import collections
import glob
import json

names = collections.Counter()          # radio registry names
insts = collections.Counter()          # episodes per task instance
per_inst = collections.defaultdict(set)  # instance -> radio names seen
all_objs = collections.Counter()       # every tracked object name

for f in glob.glob("/root/poses_x/turning_on_radio/*.json"):
    d = json.load(open(f))
    objs = d["frames"][0]["objects"].keys()
    rads = [k for k in objs if "radio" in k]
    for r in rads:
        names[r] += 1
    for o in objs:
        all_objs[o] += 1
    insts[d["instance"]] += 1
    per_inst[d["instance"]].update(rads)

print("radio names:", dict(names))
print("episodes per instance:", dict(sorted(insts.items())))
for i in sorted(per_inst):
    print(f"  instance {i}: {sorted(per_inst[i])}")
print("all tracked objects:", dict(all_objs))

"""Do the rescue pose JSONs carry object ORIENTATION? Decides metalink label fast-path.

If frames have radio pos+quat: metalink_world(t) = radio_pose(t) . fixed_offset — labels for all
200 episodes are pure composition on data we already have (minutes). If position-only: full
HDF5PlaybackWrapper replays (~12h fleet time).
"""

import glob
import json
import subprocess

subprocess.run(["mkdir", "-p", "/root/poses_x"], check=True)
subprocess.run(["tar", "xzf", "/root/rescue_dl/rescue/poses_radio.tgz", "-C", "/root/poses_x"], check=True)

fs = sorted(glob.glob("/root/poses_x/**/*.json", recursive=True))
print(f"{len(fs)} JSONs, first: {fs[0] if fs else 'NONE'}")
d = json.load(open(fs[0]))
print("top keys:", list(d.keys())[:8])
frames = d.get("frames", None)
fr = frames[0] if frames else d[list(d.keys())[0]]
print("frame type:", type(fr).__name__)
if isinstance(fr, dict):
    for k, v in fr.items():
        if isinstance(v, dict):
            for k2, v2 in list(v.items())[:4]:
                desc = f"list[{len(v2)}] {v2[:4]}" if isinstance(v2, list) else repr(v2)[:60]
                print(f"  {k}.{k2}: {desc}")
        elif isinstance(v, list):
            print(f"  {k}: list[{len(v)}] = {v[:8]}")
        else:
            print(f"  {k}: {v}")
if frames:
    print(f"n_frames: {len(frames)}")

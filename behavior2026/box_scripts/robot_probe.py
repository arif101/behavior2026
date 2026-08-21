import omnigibson.robots as R
names = [n for n in dir(R) if not n.startswith("_")]
print("robot classes:", names[:20])
cls = None
for n in names:
    if "r1" in n.lower():
        cls = getattr(R, n); print("found:", n); break
if cls is not None:
    cands = [m for m in dir(cls) if any(k in m.lower() for k in ("eef", "proprio"))]
    print("eef/proprio members:", cands[:24])

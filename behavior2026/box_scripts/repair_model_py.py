"""One-time repair: reverse the corrupted map_tokens insertions in the fork's model.py.

The first patch attempt wrote raw-string escapes literally (backslash-quote) into three
insertions before the compile check ran. The fork has no git; the insertions are known
verbatim, so removal restores the original exactly. py_compile proves it.
"""

import py_compile

P = "/root/openpi_fork/src/openpi/models/model.py"
s = open(P).read()

bad = [
    "\n    # FOVEATED MEMORY: map soft-token features (K, D) from FoveatedMap.query().\n"
    '    map_tokens: at.Float[ArrayT, \\"*b k d\\"] | None = None\n',
    '            map_tokens=data.get(\\"map_tokens\\"),\n',
    "        map_tokens=observation.map_tokens,\n",
]
for b in bad:
    if b in s:
        s = s.replace(b, "", 1)
        print(f"removed: {b.strip().splitlines()[-1][:60]!r}")
    else:
        print(f"not present (ok): {b.strip().splitlines()[-1][:60]!r}")

assert "map_tokens" not in s, "leftover map_tokens content after repair"
open(P, "w").write(s)
py_compile.compile(P, doraise=True)
print("model.py restored and compiles")

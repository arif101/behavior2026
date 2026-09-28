"""Build the ODART twin map (sim box): for every object-perturbed LeRobot root, which (demo, tag) each episode is, plus the
demo id of every b1k_radio_factory episode. Both are recoverable from the conversion order: the converter takes
sorted(glob("rac_*_200.npz")) of the round's renders, i.e. episodes are in string-sorted order of the encoded name
rac_<demo><code>_200 (codes ol5 11 .. omix2 18); the round-k root holds the clips logged as OBS_SAVED in odart_round{k-1}.out
(root r1 = the pre-loop test clip d20 ol5). Every assignment is verified against the episode length (= steps).
The factory root was built from rac_<demo>_200.npz over the 38 factory_clips demos -> string-sorted demo order.
Writes /root/odart_episode_map.json: {"odart": {root: [{"ep","demo","tag","length"}]}, "factory": {ep: demo}}.
"""
import glob, json, re, collections, pyarrow.parquet as pq
CODES = {"ol5": 11, "olm5": 12, "od5": 13, "odm5": 14, "oy15": 15, "oym15": 16, "omix1": 17, "omix2": 18}
saved = collections.defaultdict(list)                       # round k -> [(demo, tag, steps)]
for f in glob.glob("/root/odart_round*.out"):
    k = int(re.search(r"odart_round(\d+)\.out", f).group(1))
    for line in open(f):
        m = re.match(r"^d(\d+) (\w+) \d\d:\d\d OBS_SAVED rac_(\d+)_(\w+)_200\.npz \((\d+) steps\)", line.strip())
        if m: saved[k].append((int(m.group(1)), m.group(2), int(m.group(5))))
saved[0] = [(20, "ol5", 449)]                                # the pre-loop test clip, converted by round 1
out = {"odart": {}, "factory": {}}; bad = 0
for root in sorted(glob.glob("/root/b1k_radio_odart_r*"), key=lambda p: int(re.search(r"_r(\d+)_", p).group(1))):
    k = int(re.search(r"_r(\d+)_", root).group(1)); clips = sorted(saved.get(k - 1, []), key=lambda c: f"rac_{c[0]}{CODES[c[1]]:02d}_200.npz")
    e = pq.read_table(sorted(glob.glob(f"{root}/meta/episodes/**/*.parquet", recursive=True))[0]).to_pandas()
    lens = e.sort_values("episode_index")["length"].tolist(); name = root.split("/")[-1]; out["odart"][name] = []
    if len(lens) != len(clips): print(f"{name}: {len(lens)} episodes but {len(clips)} clips logged for round {k-1}"); bad += 1; continue
    for ep, ((d, tag, steps), L) in enumerate(zip(clips, lens)):
        if steps != L: print(f"{name} ep{ep}: length {L} != logged steps {steps} for d{d} {tag}"); bad += 1
        out["odart"][name].append({"ep": ep, "demo": d, "tag": tag, "length": L})
    print(f"{name}: {len(lens)} episodes mapped")
demos = sorted({int(re.search(r"d(\d+)_meta", f).group(1)) for f in glob.glob("/root/factory_clips/d*_meta.json")})
order = sorted(demos, key=lambda d: f"rac_{d}_200.npz")
fe = pq.read_table(sorted(glob.glob("/root/manufactured/b1k_radio_factory/meta/episodes/**/*.parquet", recursive=True))[0]).to_pandas()
assert len(order) == len(fe), (len(order), len(fe))
out["factory"] = {int(ep): d for ep, d in enumerate(order)}
json.dump(out, open("/root/odart_episode_map.json", "w"), indent=1)
print("factory demo order:", order[:8], "...", "| total odart eps:", sum(len(v) for v in out["odart"].values()), "| mismatches:", bad)
print("ODART_MAP_OK" if bad == 0 else "ODART_MAP_MISMATCH")

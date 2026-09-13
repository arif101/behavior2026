"""Reconstruct /root/factory_clips/d{DDD}_grasp_transport.npz + d{DDD}_meta.json from the HF
raw_segments renders (obs_render_factory2 output: actions = executed command stream at control
steps, meta = the clip meta json). The old sim box that held the originals is gone (2026-09-12).
Also writes /root/factory_clips/REBUILT_FROM_RAW_SEGMENTS.md documenting provenance.
"""
import glob, json, os, pathlib
import numpy as np

SRC = "/root/manufactured/raw_segments"
OUT = pathlib.Path("/root/factory_clips"); OUT.mkdir(exist_ok=True)
rows = []
for f in sorted(glob.glob(f"{SRC}/rac_*_0.npz")):
    demo = int(os.path.basename(f).split("_")[1])
    z = np.load(f, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    acts = np.asarray(z["actions"], np.float32)
    np.savez_compressed(OUT / f"d{demo:03d}_grasp_transport.npz", cmds=acts, meta=json.dumps(meta))
    json.dump(meta, open(OUT / f"d{demo:03d}_meta.json", "w"), indent=1)
    rows.append((demo, acts.shape, {k: meta.get(k) for k in ("t0", "closure", "weld_k", "radio_z_end", "demo", "success") if k in meta}))
    print(f"d{demo:03d}: cmds {acts.shape} meta keys {sorted(meta)[:14]}", flush=True)
(OUT / "REBUILT_FROM_RAW_SEGMENTS.md").write_text(
    "Rebuilt 2026-09-12 from arif101/b26-radio-manufactured raw_segments/rac_<demo>_0.npz (obs_render_factory2 output);\n"
    "cmds = the render's `actions` (executed command stream at control-step boundaries), meta = the render's `meta` json.\n"
    f"{len(rows)} demos: {[r[0] for r in rows]}\n")
print("FACTORY_CLIPS_REBUILT", len(rows), flush=True)
print("example meta:", rows[0][2], flush=True)

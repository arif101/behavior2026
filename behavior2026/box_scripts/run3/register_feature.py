"""Register a scalar parquet column in a LeRobot-v3 root's meta/info.json features
(idempotent). A column present in the data parquets but absent from `features` makes
LeRobotDataset raise DatasetGenerationError, which surfaces as a bogus HF 401.
  python register_feature.py --root /root/b1k_radio_map --name sample_weight --dtype float32
"""
import argparse
import json
import pathlib

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--dtype", default="float32")
ap.add_argument("--shape", type=int, default=1)
a = ap.parse_args()
ip = pathlib.Path(a.root) / "meta" / "info.json"
info = json.loads(ip.read_text())
if a.name in info["features"]:
    print(f"{a.root}: {a.name} already registered: {info['features'][a.name]}")
else:
    info["features"][a.name] = {"dtype": a.dtype, "shape": [a.shape], "names": None}
    ip.write_text(json.dumps(info, indent=4))
    print(f"{a.root}: registered {a.name} {a.dtype} [{a.shape}]")

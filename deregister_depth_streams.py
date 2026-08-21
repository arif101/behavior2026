"""De-register the depth_linear video streams from an assembled mix's meta/info.json.

Why (2026-08-08, Run-2 launch, re-created 2026-08-09 after the box died with the
uncommitted original): converter-written DEPTH videos for corrective/rac carry interior
pts jitter (±2 frames, quantize_depth VideoFrame path). Training never reads depth
pixels — gt_depth_ds labels are precomputed columns in the data parquets — so the fix
is to remove the three depth_linear features from info.json. The loader then derives
video_keys without them and never decodes (or demands) those files. RGB streams are
untouched; the defective depth mp4s stay on disk as dead weight.

Run AFTER assemble_run2_mix.py, BEFORE compute_norm_stats.
"""

import argparse
import json
import pathlib

DEPTH_PREFIX = "observation.depth_linear."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/b1k_radio_run2mix")
    a = ap.parse_args()
    info_fp = pathlib.Path(a.root) / "meta" / "info.json"
    info = json.loads(info_fp.read_text())
    removed = [k for k in info["features"] if k.startswith(DEPTH_PREFIX)]
    for k in removed:
        del info["features"][k]
    info_fp.write_text(json.dumps(info, indent=4))
    remaining_video = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    print(f"removed {len(removed)} depth features: {removed}")
    print(f"remaining video features: {remaining_video}")
    assert not any(k.startswith(DEPTH_PREFIX) for k in info["features"])
    assert len(remaining_video) == 3, f"expected 3 RGB streams, got {remaining_video}"
    print("DEPTH_DEREGISTERED_OK")


if __name__ == "__main__":
    main()

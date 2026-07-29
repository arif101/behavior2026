"""Dump every Nth observation the policy server receives, for the latent point-cloud probe.

The probe (probes/latent_pointcloud.py, written Jul 26, never yet run) needs RGB + depth +
camera poses from REAL rollout frames. The serving path already receives all of it on every
call — head 720x720 rgb + depth_linear, both wrists at 480, cam_rel_poses (camera extrinsics
relative to the BASE, which is exactly the frame we want), proprio, and (post-passthrough)
target_points. So the cheapest capture is server-side: save the raw received dict every Nth
call during an oracle-conditioned rollout that actually drives toward the radio.

Writes /root/pcl_frames/frame_%04d.npz, capped at MAX_FRAMES.
"""

import argparse
import pathlib

INSERT_AFTER = "                obs = deepcopy(result)\n"

CAPTURE = '''
                # --- frame capture for the latent point-cloud probe (patch_frame_capture.py)
                try:
                    import os as _os

                    import numpy as _np

                    globals()["_pcl_n"] = globals().get("_pcl_n", 0) + 1
                    _n = globals()["_pcl_n"]
                    if _n % 20 == 0:
                        _os.makedirs("/root/pcl_frames", exist_ok=True)
                        _idx = _n // 20
                        if _idx <= 40:
                            _out = {}
                            for _k, _v in obs.items():
                                _a = _np.asarray(_v)
                                if _a.dtype != object:
                                    _out[str(_k)] = _a
                            _np.savez_compressed(f"/root/pcl_frames/frame_{_idx:04d}.npz", **_out)
                except Exception:
                    pass
                # ---------------------------------------------------------------------------
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="/root/openpi_fork/src/openpi/serving/websocket_b1k_server.py")
    a = ap.parse_args()
    p = pathlib.Path(a.server)
    s = p.read_text()
    if "patch_frame_capture.py" in s:
        raise SystemExit("already patched")
    if INSERT_AFTER not in s:
        raise SystemExit(f"anchor not found in {p}")
    p.write_text(s.replace(INSERT_AFTER, INSERT_AFTER + CAPTURE, 1))
    print(f"patched {p} -> /root/pcl_frames/frame_XXXX.npz every 20th call, max 40")


if __name__ == "__main__":
    main()

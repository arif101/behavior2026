"""Log every emitted action to JSONL so we can decompose base translation vs rotation.

WHY. The first honest closed-loop rollout (ckpt 49999, turning_on_radio inst 301, full challenge
config) scored q=0.0 and, per the video, the robot ROTATED to bring the radio into view and then
never approached it: total base path 0.76 m vs the human's 5.70 m, while the arms swept 3x the
human's path. The radio was clearly visible and unoccluded for most of the episode, so this is
neither an occlusion nor a memory failure -- the target was acquired and the approach was not
executed.

THE HYPOTHESIS THIS TESTS. The base is 3 DoF, action[0:3], and our odometry calibration
established these are BODY-FRAME [vx, vy, wz]. "Rotate in place but never translate" is exactly
what you see if the policy emits yaw (wz) while translation (vx, vy) stays near zero. If that is
what is happening, the bottleneck is base action generation -- and neither grounding, nor spatial
memory, nor camera resolution touches it.

The server is the right place to log: it sees the final action actually returned to the eval
client, after receding-horizon selection and any post-processing, which is the thing the robot
executes. Logging inside the policy would miss that.

Writes /root/action_log.jsonl, one record per inference call. Cheap (a few floats per call) and
does not alter the action.
"""

import argparse
import pathlib

INSERT_AFTER = "                action = self._policy.act(obs)\n"

LOGGING = '''
                # --- action logging (patch_action_logger.py) -------------------------------
                # Decompose the emitted action so we can tell rotate-in-place from travel.
                try:
                    import json as _json

                    import numpy as _np

                    _a = action.cpu().numpy() if hasattr(action, "cpu") else _np.asarray(action)
                    _flat = _np.asarray(_a).reshape(-1, _np.asarray(_a).shape[-1])[0]
                    with open("/root/action_log.jsonl", "a") as _fh:
                        _fh.write(
                            _json.dumps(
                                {
                                    # b1k.py: StateActionConfig(name="base", indices=range(3))
                                    # body-frame [vx, vy, wz] per our odometry calibration
                                    "base_vx": float(_flat[0]),
                                    "base_vy": float(_flat[1]),
                                    "base_wz": float(_flat[2]),
                                    "base_trans_mag": float(_np.linalg.norm(_flat[0:2])),
                                    "base_yaw_mag": float(abs(_flat[2])),
                                    "arm_mag": float(_np.linalg.norm(_flat[3:])),
                                    "action_dim": int(_flat.shape[0]),
                                }
                            )
                            + "\\n"
                        )
                except Exception:  # never let logging break the eval
                    pass
                # --------------------------------------------------------------------------
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="/root/openpi_fork/src/openpi/serving/websocket_b1k_server.py")
    a = ap.parse_args()

    p = pathlib.Path(a.server)
    src = p.read_text()
    if "patch_action_logger.py" in src:
        raise SystemExit("already patched")
    if INSERT_AFTER not in src:
        raise SystemExit(f"anchor not found in {p}; server layout changed")

    p.write_text(src.replace(INSERT_AFTER, INSERT_AFTER + LOGGING, 1))
    print(f"patched {p} -> logs to /root/action_log.jsonl")


if __name__ == "__main__":
    main()

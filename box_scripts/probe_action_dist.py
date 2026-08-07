"""ACTION-DISTRIBUTION PROBE: at a restored near-commit state, what fraction of the policy's
sampled action chunks COMMIT (trunk-led descent / gripper close), and how diverse are samples?

Separates the two initiation-wall mechanisms:
  - commit mass ~0 AND low diversity  -> mode absent / collapsed => data (splice clips) + RL path
  - commit mass >~5% with diversity   -> mode exists, chunk-boundary resampling kills it
                                         => serve-time best-of-N / chunk-persistence fix

Method: restore demo 10 to f1100 (25 steps pre-lift), then N times: open a FRESH websocket
connection (mirrors the collector's per-attempt reset semantics) and call act() CHUNK times on
the SAME frozen obs — unrolls one independently-sampled 32-step chunk without stepping the env.

Run inside the behavior env with the policy server up:
  OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 python probe_action_dist.py
"""

import os

import numpy as np

os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")

DEMO = 10
T_START = 1100
N_SAMPLES = 25
CHUNK = 32

from reverse_curriculum_collect import EVAL_PROPRIO_KEYS, restore_to_frame  # noqa: E402


def main():
    import omnigibson as og
    from omnigibson.macros import gm
    gm.HEADLESS = True
    gm.ENABLE_TRANSITION_RULES = False
    from omnigibson.envs.hdf5_data_wrapper import HDF5PlaybackWrapper
    from omnigibson.eval.utils.network_utils import WebsocketClientPolicy

    h5 = f"/root/rawdemos/task-0000/episode_{DEMO:08d}.hdf5"
    wrapper = HDF5PlaybackWrapper.create_from_hdf5(
        input_path=h5, output_path="/root/rc_probe2_tmp.hdf5",
        robot_obs_modalities=("rgb", "depth_linear", "proprio"),
        robot_proprio_keys=EVAL_PROPRIO_KEYS,
    )
    env = wrapper.env
    restore_to_frame(wrapper, 0, T_START)
    obs = env.get_obs()[0]

    def _p(v):
        return np.asarray(v.cpu() if hasattr(v, "cpu") else v, dtype=float)

    pp = _p(obs["robot_r1"]["proprio"] if "robot_r1" in obs else obs["robot_r1::proprio"]).reshape(-1)
    trunk_now = pp[53:57]
    print(f"PROBE2 restored f{T_START}; trunk_now={np.round(trunk_now, 3).tolist()}", flush=True)

    chunks = []
    for i in range(N_SAMPLES):
        policy = WebsocketClientPolicy(host="127.0.0.1", port=8000)
        acts = [np.asarray(policy.act(obs)).reshape(-1) for _ in range(CHUNK)]
        chunks.append(np.stack(acts))
        if (i + 1) % 5 == 0:
            print(f"PROBE2 sampled {i + 1}/{N_SAMPLES}", flush=True)
    C = np.stack(chunks)  # (N, CHUNK, 23): base 0:3 trunk 3:7 armL 7:14 armR 14:21 grip 21,22
    np.save("/root/probe_action_dist_chunks.npy", C)

    # commit signatures per chunk
    trunk_cmd = C[:, :, 3:7]
    trunk_drop = (trunk_cmd[:, :, 0] - trunk_now[0]).min(axis=1)  # most-negative dim0 excursion
    grip_l = C[:, :, 21]
    grip_close = grip_l.min(axis=1)
    div = C.std(axis=0).mean()  # cross-sample diversity of the whole chunk tensor
    within = C.std(axis=1).mean()

    print(f"PROBE2 trunk_drop per-chunk min-excursion: p10={np.percentile(trunk_drop, 10):.3f} "
          f"p50={np.percentile(trunk_drop, 50):.3f} p90={np.percentile(trunk_drop, 90):.3f}", flush=True)
    for thr in (0.10, 0.20, 0.35):
        frac = float((trunk_drop < -thr).mean())
        print(f"PROBE2 commit_mass(trunk drop > {thr:.2f} rad) = {frac:.2f}", flush=True)
    print(f"PROBE2 gripL cmd min: p10={np.percentile(grip_close, 10):.3f} "
          f"p50={np.percentile(grip_close, 50):.3f}", flush=True)
    print(f"PROBE2 diversity: cross-sample std={div:.4f} within-chunk std={within:.4f}", flush=True)
    print("PROBE2_DONE", flush=True)
    og.clear()


if __name__ == "__main__":
    main()

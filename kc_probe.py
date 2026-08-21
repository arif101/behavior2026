"""KEY-CAUSALITY PROBE (baseline): does the sampled trajectory TRACK the affordance point?

At a frozen restored state, sample K chunks per condition: TRUE point vs +/-3cm
displacements (base frame). A lookup ignores the key (delta ~ within-condition noise);
a keyed primitive tracks it (delta >> noise, sign flips with displacement direction).
Run at TWO bands: approach (positive control -- conditioning historically steered
approach) and near-commit (the question). Baseline prediction after the liveness DEAD
verdict: zero tracking at BOTH bands.

Frame-convention note: points are computed like skill_env_wrapper (metalink offset in the
radio root frame -> world -> base frame) and displaced in base frame. If the dataset's
label convention differs, the probe still measures tracking-of-a-3cm-key-shift; verify
the convention against add_target_points.py before the post-retrain run.

Needs: policy server on :8000 (point passthrough consumed via obs["target_points"]).
Run: OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 \
  OMNIGIBSON_HEADLESS=1 python -u kc_probe.py
"""

import json
import os

import numpy as np

os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")

DEMO = 10
BANDS = {"approach": 700, "near_commit": 1100}
K = 10           # chunks per condition
CHUNK = 32
DISP = 0.03      # metres
CONDS = {"true": np.zeros(3), "px": np.array([DISP, 0, 0]), "nx": np.array([-DISP, 0, 0]),
         "py": np.array([0, DISP, 0]), "ny": np.array([0, -DISP, 0])}

from reverse_curriculum_collect import EVAL_PROPRIO_KEYS, restore_to_frame  # noqa: E402
from skill_env_wrapper import q2r, _np  # noqa: E402


def main():
    import omnigibson as og  # noqa: F401
    from omnigibson.macros import gm
    gm.HEADLESS = True
    gm.ENABLE_TRANSITION_RULES = False
    from omnigibson.envs.hdf5_data_wrapper import HDF5PlaybackWrapper
    from omnigibson.eval.utils.network_utils import WebsocketClientPolicy

    h5 = f"/root/rawdemos/task-0000/episode_{DEMO:08d}.hdf5"
    wrapper = HDF5PlaybackWrapper.create_from_hdf5(
        input_path=h5, output_path="/root/kc_probe_tmp.hdf5",
        robot_obs_modalities=("rgb", "depth_linear", "proprio"),
        robot_proprio_keys=EVAL_PROPRIO_KEYS)
    env = wrapper.env
    robot = wrapper.scene.robots[0]
    radio = [o for o in wrapper.scene.objects if "radio" in o.name.lower()][0]
    p_off = np.asarray(json.load(open("/root/behavior2026/labels/metalink_offset.json"))
                       ["offset_pos_root_frame"])

    out = {"demo": DEMO, "K": K, "disp_m": DISP, "bands": {}}
    for band, frame in BANDS.items():
        restore_to_frame(wrapper, 0, frame)
        obs = env.get_obs()[0]
        tp, tq = radio.get_position_orientation()
        pt_world = _np(tp) + q2r(_np(tq)) @ p_off
        bp, bq = robot.get_position_orientation()
        pt_base = q2r(_np(bq)).T @ (pt_world - _np(bp))
        print(f"KC band={band} f={frame} pt_base={np.round(pt_base, 3).tolist()}", flush=True)

        band_res = {}
        for cname, d in CONDS.items():
            pt = (pt_base + d).astype(np.float32)
            chunks = []
            for _ in range(K):
                o = dict(obs)
                o["target_points"] = np.stack([pt, pt]).astype(np.float32)
                o["target_points_mask"] = np.array([True, True], dtype=bool)
                pol = WebsocketClientPolicy(host="127.0.0.1", port=8000)
                acts = [np.asarray(pol.act(o)).reshape(-1) for _ in range(CHUNK)]
                chunks.append(np.stack(acts))
            C = np.stack(chunks)                       # (K, 32, 23)
            band_res[cname] = {"mean": C.mean(axis=0), "std_within": float(C.std(axis=0).mean())}
            print(f"KC   {band}/{cname}: within-cond std={band_res[cname]['std_within']:.4f}",
                  flush=True)

        true_mean = band_res["true"]["mean"]
        noise = band_res["true"]["std_within"]
        rep = {}
        for cname in ("px", "nx", "py", "ny"):
            delta = band_res[cname]["mean"] - true_mean
            rep[cname] = {"delta_norm": float(np.abs(delta).mean()),
                          "snr": float(np.abs(delta).mean() / max(noise, 1e-9))}
        # directional consistency: +x and -x deltas should anti-correlate if tracking
        dpx = (band_res["px"]["mean"] - true_mean).ravel()
        dnx = (band_res["nx"]["mean"] - true_mean).ravel()
        dpy = (band_res["py"]["mean"] - true_mean).ravel()
        dny = (band_res["ny"]["mean"] - true_mean).ravel()
        rep["anticorr_x"] = float(np.corrcoef(dpx, dnx)[0, 1])
        rep["anticorr_y"] = float(np.corrcoef(dpy, dny)[0, 1])
        out["bands"][band] = {"frame": frame, "noise_floor": noise, "report": rep}
        print(f"KC {band} REPORT: " + json.dumps(rep), flush=True)

    for b, r in out["bands"].items():
        snrs = [r["report"][c]["snr"] for c in ("px", "nx", "py", "ny")]
        tracking = (max(snrs) > 3.0 and min(r["report"]["anticorr_x"],
                                            r["report"]["anticorr_y"]) < -0.5)
        r["verdict"] = "TRACKS KEY" if tracking else "IGNORES KEY"
        print(f"KC VERDICT {b}: {r['verdict']} (snr={[round(s, 2) for s in snrs]})", flush=True)
    json.dump({k: v for k, v in out.items() if k != "bands"} |
              {"bands": {b: {"frame": r["frame"], "noise_floor": r["noise_floor"],
                             "report": r["report"], "verdict": r["verdict"]}
                         for b, r in out["bands"].items()}},
              open("/root/kc_probe_baseline.json", "w"), indent=1)
    print("KC_PROBE_DONE -> /root/kc_probe_baseline.json", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()

"""STAGE-HEAD HELD-OUT VALIDATION — closes the Run-2 PENDING secondary; evidence for the
gate-pin decision (Arif's 2026-08-12 open question).

Data: /root/b1k_radio_map episodes 180-199 (held out by the run-1 convention; videos are
on-box), stride-10 frames. Forward: the policy's own input transform -> Observation ->
model.compute_loss run EAGERLY with the demo's action chunk; the stage head's pooled
suffix feature is captured via an instance wrapper and logits recomputed with the head's
own two Linears. CAVEAT (recorded): compute_loss samples the flow-matching time
internally, so this measures accuracy under the head's TRAINING input distribution; the
serving-faithful readout (final denoise step of sample_actions) lands with d7-d8 serving
integration and should be spot-checked then.

Report: 4x4 confusion, per-class recall, and the gate slices —
  A: dmin <= 0.12 m (labeled MANIPULATE by the 12 cm rule = the eval-time hover band):
     P(pred=2), P(pred in {1,2})
  B: 0.07 < dmin <= 0.20 m approach shell: pred distribution
Run (openpi env, GPU): cd /root/openpi_fork && python -u /root/stage_head_valid.py
"""

import glob
import json

import av
import numpy as np
import pyarrow.parquet as pq

import jax
import jax.numpy as jnp

import openpi.training.config as _config
from openpi.models import model as _model_mod
from openpi.policies import policy_config as _policy_config

ROOT = "/root/b1k_radio_map"
EPISODES = list(range(180, 200))
STRIDE = 10
BATCH = 8
CAMS = {  # image_i -> video key (camera 0 = base_0_rgb = head per pi0.py comment)
    0: "observation.rgb.zed_link_camera_0",
    1: "observation.rgb.left_realsense_link_camera_0",
    2: "observation.rgb.right_realsense_link_camera_0",
}

print("loading policy...", flush=True)
cfg = _config.get_config("pi05_radio_run2")
policy = _policy_config.create_trained_policy(cfg, "/root/ckpt",
                                              default_prompt="turning_on_radio")
model = policy._model
H, AD = model.action_horizon, model.action_dim

_ns = json.load(open("/root/ckpt/assets/b1k_radio/norm_stats.json"))
_ns = _ns.get("norm_stats", _ns)["actions"]
Q01 = np.asarray(_ns["q01"], np.float32)
Q99 = np.asarray(_ns["q99"], np.float32)


def norm_pad_chunk(rows):
    """Quantile-normalize (pi05 convention: 2(x-q01)/(q99-q01)-1) the 23 real dims and
    zero-pad to the model's action_dim (padded dims are zeros in training too)."""
    z = 2.0 * (rows - Q01[None, : rows.shape[1]]) / \
        (Q99[None, : rows.shape[1]] - Q01[None, : rows.shape[1]] + 1e-8) - 1.0
    out = np.zeros((rows.shape[0], AD), np.float32)
    out[:, : rows.shape[1]] = z
    return out

orig_in = model.stage_head_in
cap = {}


class _Cap:
    def __call__(self, x):
        cap["feat"] = x
        return orig_in(x)


object.__setattr__(model, "stage_head_in", _Cap())


def head_logits():
    return np.asarray(
        model.stage_head_out(jax.nn.gelu(orig_in(cap["feat"])))[:, :4])


em_fp = sorted(glob.glob(f"{ROOT}/meta/episodes/**/*.parquet", recursive=True))[0]
em = pq.read_table(em_fp).to_pydict()
by_ep = {int(e): i for i, e in enumerate(em["episode_index"])}

rng = jax.random.key(17)
labels_all, preds_all, dmin_all = [], [], []

for ep in EPISODES:
    if ep not in by_ep:
        print(f"ep{ep}: not in meta -- skipped", flush=True)
        continue
    i = by_ep[ep]
    dfp = (f"{ROOT}/data/chunk-{em['data/chunk_index'][i]:03d}/"
           f"file-{em['data/file_index'][i]:03d}.parquet")
    t = pq.read_table(dfp, columns=["episode_index", "observation.state", "stage",
                                    "target_points", "action"])
    m = t["episode_index"].to_numpy() == ep
    st = np.stack(t["observation.state"].to_numpy()[m]).astype(np.float32)
    stage = t["stage"].to_numpy()[m].astype(int)
    tp = np.stack(t["target_points"].to_numpy()[m]).astype(np.float32)
    act = np.stack(t["action"].to_numpy()[m]).astype(np.float32)
    n = len(st)

    frames = {}
    ok = True
    for k, vkey in CAMS.items():
        c = em[f"videos/{vkey}/chunk_index"][i]
        f = em[f"videos/{vkey}/file_index"][i]
        t0 = float(em[f"videos/{vkey}/from_timestamp"][i])
        t1 = float(em[f"videos/{vkey}/to_timestamp"][i])
        path = f"{ROOT}/videos/{vkey}/chunk-{c:03d}/file-{f:03d}.mp4"
        try:
            container = av.open(path)
            stream = container.streams.video[0]
            container.seek(int(t0 / stream.time_base), stream=stream)
            fr = []
            for frame in container.decode(stream):
                ts = float(frame.pts * stream.time_base)
                if ts < t0 - 1e-3:
                    continue
                if ts >= t1 - 1e-3 or len(fr) >= n:
                    break
                fr.append(frame.to_ndarray(format="rgb24"))
            container.close()
            frames[k] = fr
        except Exception as e:  # noqa: BLE001
            print(f"ep{ep} cam{k}: decode failed ({e}) -- episode skipped", flush=True)
            ok = False
            break
    if not ok:
        continue
    n_use = min(n, *(len(frames[k]) for k in CAMS))
    idx = list(range(0, n_use, STRIDE))
    print(f"ep{ep}: {n} rows, {n_use} aligned frames, {len(idx)} sampled", flush=True)

    for b0 in range(0, len(idx), BATCH):
        sel = idx[b0:b0 + BATCH]
        inputs = []
        for s in sel:
            obs = {f"observation/image_{k}": frames[k][s] for k in CAMS}
            obs.update({"observation/state": st[s], "prompt": "turning_on_radio",
                        "target_points": tp[s].reshape(2, 3),
                        "target_points_mask": np.array([True, True]),
                        "map_tokens_full": np.zeros(576, np.float32),
                        "map_tokens_blind": np.zeros(576, np.float32)})
            inputs.append(policy._input_transform(obs))
        batch = jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]),
                             *inputs)
        chunks = []
        for s in sel:
            rows = act[s:s + H]
            if len(rows) < H:
                rows = np.concatenate([rows, np.repeat(rows[-1:], H - len(rows), 0)])
            chunks.append(norm_pad_chunk(rows))
        actions = jnp.asarray(np.stack(chunks))
        rng, sub = jax.random.split(rng)
        obs_b = _model_mod.Observation.from_dict(batch)
        model.compute_loss(sub, obs_b, actions)  # eager; fills cap["feat"]
        pred = head_logits().argmax(-1)
        preds_all.extend(pred.tolist())
        labels_all.extend(stage[sel].tolist())
        d = np.minimum(np.linalg.norm(tp[sel, :3], axis=1),
                       np.linalg.norm(tp[sel, 3:], axis=1))
        dmin_all.extend(d.tolist())

labels = np.array(labels_all)
preds = np.array(preds_all)
dmin = np.array(dmin_all)
conf = np.zeros((4, 4), int)
for l, p in zip(labels, preds):
    conf[l, p] += 1
print("\n==== STAGE HEAD VALIDATION (held-out eps 180-199) ====")
print(f"frames scored: {len(labels)}")
print("confusion (rows=true 0..3, cols=pred):")
print(conf)
acc = (labels == preds).mean()
print(f"overall accuracy: {acc:.3f}")
for c in range(4):
    n_c = (labels == c).sum()
    if n_c:
        print(f"  recall class {c}: {(preds[labels == c] == c).mean():.3f} (n={n_c})")
A = dmin <= 0.12
B = (dmin > 0.07) & (dmin <= 0.20)
if A.any():
    print(f"SLICE A near-field dmin<=0.12 (n={A.sum()}): "
          f"P(pred=2)={np.mean(preds[A] == 2):.3f}  "
          f"P(pred in 1,2)={np.mean(np.isin(preds[A], [1, 2])):.3f}")
if B.any():
    print(f"SLICE B shell 0.07<dmin<=0.20 (n={B.sum()}): "
          f"pred dist={np.bincount(preds[B], minlength=4).tolist()}")
json.dump({"conf": conf.tolist(), "acc": float(acc),
           "sliceA_p2": float(np.mean(preds[A] == 2)) if A.any() else None,
           "sliceA_p12": float(np.mean(np.isin(preds[A], [1, 2]))) if A.any() else None,
           "sliceB_dist": np.bincount(preds[B], minlength=4).tolist() if B.any() else None,
           "n": int(len(labels))},
          open("/root/stage_head_valid.json", "w"), indent=2)
print("STAGE_HEAD_VALID_DONE")

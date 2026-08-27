"""Frozen-backbone frame features for the task-62 stage head (STAGE_HEAD_SPEC_v2 §4, task-62 only).
Per usable demo, every STRIDE-th frame (6 Hz): DINOv2 ViT-B/14 CLS + mean-patch (2x768) for each of the
3 cameras (head zed 720p, left/right wrist 480p, all area-scaled to 224), proprio (observation.state,
61-D) at the same frames, and the per-frame targets from the release (family 0-7 from the annotation,
q + post-slice from reward spikes, held-object per arm from the state's AG block — via relabel_offline).
Output: {out}/ep{raw}.npz  feats f16 [N,3,1536], proprio f32 [N,61], frame [N], fam [N], q [N], agL/agR [N],
        seg_start/seg_end [N] (annotated segment bounds of each frame, for within-skill progress + soft labels)
Run in the openpi env (torch 2.7 + hub DINOv2):  python task62/stage_head/extract_feats.py [--eps 620040,...]
"""
import os, sys, json, glob, argparse, subprocess, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch, pandas as pd

STRIDE, IMG = 5, 224
VID = "/root/t62_lerobot_src/videos"; DATA = "/root/t62_lerobot_src/data/chunk-062"
ANN = "/root/t62_annotations/annotations/task-0062"; LAB = "/root/step0/relabel_offline"
TABLE = "/tmp/claude-0/-root/31447591-5629-4337-8d31-0590aa61cab8/scratchpad/t62_episode_table.json"
CAMS = ["observation.rgb.zed_link_camera_0", "observation.rgb.left_realsense_link_camera_0", "observation.rgb.right_realsense_link_camera_0"]
FAM_OF = {"move to": 0, "open door": 1, "pick up from": 2, "place on": 3, "chop": 4, "place in": 5, "close door": 6, "push to": 7}
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)

def decode(cam, v, n_frames):
    """ffmpeg: seek to the episode, keep every STRIDE-th frame, area-scale to IMG -> uint8 [N,IMG,IMG,3]."""
    path = f"{VID}/{cam}/chunk-{v['chunk']:03d}/file-{v['file']:03d}.mp4"; dur = n_frames / 30.0
    cmd = ["ffmpeg", "-loglevel", "error", "-ss", f"{v['t0']:.3f}", "-i", path, "-t", f"{dur:.3f}",
           "-vf", f"select=not(mod(n\\,{STRIDE})),scale={IMG}:{IMG}:flags=area", "-vsync", "vfr",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    n = len(raw) // (IMG * IMG * 3); return np.frombuffer(raw[: n * IMG * IMG * 3], np.uint8).reshape(n, IMG, IMG, 3)

def targets(raw_id, T, frames):
    ann = json.load(open(f"{ANN}/episode_{raw_id:08d}.json"))["skill_annotation"]
    fam = np.full(T, 7, np.int64); s0 = np.zeros(T, np.int64); s1 = np.full(T, T, np.int64)
    for s in ann:
        a, b = s["frame_duration"]; fam[a:b] = FAM_OF.get(s["skill_description"][0], 7); s0[a:b] = a; s1[a:b] = b
    lab = np.load(f"{LAB}/ep{raw_id}.npz"); n = min(T, len(lab["q"]))
    q = np.zeros(T, np.float32); q[:n] = lab["q"][:n]; q[n:] = q[n - 1]
    agL = np.zeros(T, np.int64); agR = np.zeros(T, np.int64); agL[:n] = lab["ag_L"][:n]; agR[:n] = lab["ag_R"][:n]
    return dict(fam=fam[frames], q=q[frames], agL=agL[frames], agR=agR[frames], seg_start=s0[frames], seg_end=s1[frames])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--eps", default=""); ap.add_argument("--out", default="/root/step0/stage_feats")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    table = json.load(open(TABLE)); summ = json.load(open(f"{LAB}/summary_offline.json"))
    eps = [int(x) for x in a.eps.split(",")] if a.eps else sorted(int(k) for k in table if summ[k]["usable"])
    eps = [e for e in eps if not os.path.exists(f"{a.out}/ep{e}.npz")]
    print(f"{len(eps)} episodes to extract", flush=True)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").eval().cuda().half()
    mean = torch.tensor(MEAN).cuda().view(1, 3, 1, 1).half(); std = torch.tensor(STD).cuda().view(1, 3, 1, 1).half()
    prop = {}   # episode_index -> [T,61] observation.state
    for p in sorted(glob.glob(f"{DATA}/*.parquet")):
        df = pd.read_parquet(p, columns=["episode_index", "frame_index", "observation.state"])
        for e, g in df.groupby("episode_index"): prop[int(e)] = np.stack(g.sort_values("frame_index")["observation.state"].values).astype(np.float32)
    pool = ThreadPoolExecutor(3)
    def embed(imgs):
        out = []
        for i in range(0, len(imgs), 128):
            x = torch.from_numpy(imgs[i:i + 128]).cuda().permute(0, 3, 1, 2).half().div_(255).sub_(mean).div_(std)
            with torch.no_grad(): o = model.forward_features(x)
            out.append(torch.cat([o["x_norm_clstoken"], o["x_norm_patchtokens"].mean(1)], -1).cpu().numpy().astype(np.float16))
        return np.concatenate(out)
    for k, ep in enumerate(eps):
        t0 = time.time(); r = table[str(ep)]; T = r["T_lerobot"]; frames = np.arange(0, T, STRIDE)
        vids = list(pool.map(lambda c: decode(c, r["videos"][c], T), CAMS))
        n = min(len(frames), *[len(v) for v in vids])
        if n < len(frames) - 2: print(f"  ep{ep}: WARNING decoded {[len(v) for v in vids]} vs expected {len(frames)}", flush=True)
        frames = frames[:n]; feats = np.stack([embed(v[:n]) for v in vids], 1)          # [N,3,1536]
        pr = prop[r["episode_index"]]; pr = pr[np.minimum(frames, len(pr) - 1)]
        np.savez(f"{a.out}/ep{ep}.npz", feats=feats, proprio=pr, frame=frames, **targets(ep, T, frames))
        print(f"[{k+1}/{len(eps)}] ep{ep} N={n} {time.time()-t0:.1f}s", flush=True)
    print("EXTRACT_DONE", flush=True)

if __name__ == "__main__": main()

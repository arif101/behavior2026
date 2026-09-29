"""Sim box: `dl_readout_mix.py --mix` pulls HF b26-run3-mixes/mix_readout -> /root/b1k_radio_mix_readout;
`dl_readout_mix.py --step 2500` pulls b26-run3-params/4dall/ckpt_2500/params -> /root/run3_dl/4dall_2500/4dall/ckpt_2500/params."""
import os, sys, argparse; os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
from huggingface_hub import snapshot_download
ap = argparse.ArgumentParser(); ap.add_argument("--mix", action="store_true"); ap.add_argument("--step", default=None); ap.add_argument("--arm", default="4dall"); a = ap.parse_args()
tok = open("/root/.hf_token").read().strip()
if a.mix:
    snapshot_download("arif101/b26-run3-mixes", repo_type="dataset", token=tok, local_dir="/root/mix_readout_dl", allow_patterns=["mix_readout/**"])
    os.makedirs("/root/b1k_radio_mix_readout", exist_ok=True)
    rc = os.system("cp -a /root/mix_readout_dl/mix_readout/. /root/b1k_radio_mix_readout/ && ls /root/b1k_radio_mix_readout /root/b1k_radio_mix_readout/meta")
    print("READOUT_MIX_DL_OK" if rc == 0 else "READOUT_MIX_DL_FAILED", flush=True)
if a.step:
    d = f"/root/run3_dl/{a.arm}_{a.step}"
    snapshot_download("arif101/b26-run3-params", token=tok, local_dir=d, allow_patterns=[f"{a.arm}/ckpt_{a.step}/params/**"])
    p = f"{d}/{a.arm}/ckpt_{a.step}/params"; print(("CKPT_DL_OK " if os.path.isdir(p) else "CKPT_DL_MISSING ") + p, flush=True)

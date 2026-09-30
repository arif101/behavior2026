"""Sim box: pull the FINAL 4dall params (HF b26-run3-params/4dall/params, written by the driver's finalize step, not a
ckpt_<step> folder) to /root/run3_dl/4dall_final/4dall/params and expose them at the readout rule's path via a symlink."""
import os; os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
from huggingface_hub import snapshot_download
tok = open("/root/.hf_token").read().strip()
d = "/root/run3_dl/4dall_final"
snapshot_download("arif101/b26-run3-params", token=tok, local_dir=d, allow_patterns=["4dall/params/**"])
p = f"{d}/4dall/params"
os.makedirs("/root/run3_dl/4dall_14999/4dall/ckpt_14999", exist_ok=True)
link = "/root/run3_dl/4dall_14999/4dall/ckpt_14999/params"
if os.path.islink(link): os.unlink(link)
if not os.path.exists(link): os.symlink(p, link)
ok = os.path.isdir(p) and len(os.listdir(p)) > 0
print(("CKPT_DL_OK " if ok else "CKPT_DL_MISSING ") + p + f" files={len(os.listdir(p)) if os.path.isdir(p) else 0}", flush=True)

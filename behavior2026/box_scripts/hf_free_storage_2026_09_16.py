"""Free HF storage (approved by the user 2026-09-16 after the 403 'set up automatic credit recharge' block).
Deletes, in this order, and prints sizes:
  1. arif101/behavior2026-artifacts: ckpts/phaseA/ (100 GB), sweep_100/sweep_out/ (65 GB), the three LoRA ckpts (28 GB)
  2. arif101/behavior2026-artifacts: ckpts/g3/{lang,lang_point,taskid}/{10000,20000}/ (75 GB; final 29999 params kept)
  3. arif101/b26-run3-params: a0..a4/ckpt_12500/ (62 GB; superseded by <arm>/params)
Token from /root/.hf_token. Run on the trainer box:
  /root/openpi_fork/.venv/bin/python /root/run3/hf_free_storage_2026_09_16.py [--dry-run]
"""
import sys
from huggingface_hub import HfApi, CommitOperationDelete
DRY = "--dry-run" in sys.argv
tok = open("/root/.hf_token").read().strip(); api = HfApi(token=tok)

def delete_prefixes(repo, kind, prefixes, msg):
    info = api.repo_info(repo, repo_type=kind, files_metadata=True, token=tok)
    files = [s for s in info.siblings if any(s.rfilename.startswith(p) for p in prefixes)]
    size = sum((s.size or (s.lfs.size if s.lfs else 0) or 0) for s in files)
    print(f"{'DRY ' if DRY else ''}{repo}: {len(files)} files, {size/1e9:.1f} GB under {prefixes}", flush=True)
    if DRY or not files:
        return
    ops = [CommitOperationDelete(path_in_repo=s.rfilename) for s in files]
    for i in range(0, len(ops), 500):
        api.create_commit(repo_id=repo, repo_type=kind, operations=ops[i:i + 500], commit_message=f"{msg} (batch {i // 500 + 1})", token=tok)
    print("   deleted", flush=True)

delete_prefixes("arif101/behavior2026-artifacts", "model",
                ["ckpts/phaseA/", "sweep_100/sweep_out/", "ckpts/g2_radio_lora_5000/", "ckpts/wood_lora_10000/", "ckpts/trash_lora_29999/"],
                "free storage: July Phase-A ckpts, sweep_100 outputs, LoRA ckpts (superseded; no current arm reads them)")
# item 2 (adjusted 2026-09-16): the G3 arms carry no train_state on HF, only params at steps 10000/20000/29999 -> keep the
# final 29999 params per arm, drop the two intermediates (6 x 12.4 GB = 75 GB).
delete_prefixes("arif101/behavior2026-artifacts", "model",
                [f"ckpts/g3/{a}/{st}/" for a in ("lang", "lang_point", "taskid") for st in ("10000", "20000")],
                "free storage: G3 intermediate params (final 29999 kept per arm)")
delete_prefixes("arif101/b26-run3-params", "model", [f"a{i}/ckpt_12500/" for i in range(5)],
                "free storage: step-12500 intermediates (finals in <arm>/params supersede them)")
for repo in ("arif101/behavior2026-artifacts", "arif101/b26-run3-params"):
    info = api.repo_info(repo, repo_type="model", files_metadata=True, token=tok)
    print(f"{repo} now {sum((s.size or (s.lfs.size if s.lfs else 0) or 0) for s in info.siblings)/1e9:.1f} GB")
print("HF_FREE_STORAGE_DONE")

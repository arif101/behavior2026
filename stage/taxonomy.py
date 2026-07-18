"""Official BEHAVIOR-2026 skill taxonomy + literal encoding.

Source of truth: stage/official_taxonomy.json, extracted from the organizers'
per-episode annotations in HF `behavior-1k/2026-challenge-demos` (annotations/
task-XXXX/episode_*.json, `skill_annotation` field). 34 skill ids observed over
3 sampled episodes x 100 tasks; per-task skill sets drive the task-masked
logits (Larchenko trick). Rebuild from a full local dump with build_taxonomy().

Stage class space = [IDLE] + official skill ids (sorted). IDLE is ours: the
official stream is single-track, but episodes are bimanual-parallel, so at any
frame one arm may have no active skill.

Literal encoding: a goal literal is (predicate, target category, reference
category). Predicates get a small closed vocabulary; categories are hash-
bucketed (256 buckets) so unseen categories at serve time still embed -- the
director supplies real BDDL literals through the same encoder.
"""

import hashlib
import json
import os

_TAX = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "official_taxonomy.json")))

SKILL_IDS = sorted(int(k) for k in _TAX["skills"])          # official ids
SKILL_NAMES = {int(k): v for k, v in _TAX["skills"].items()}
IDLE = 0                                                     # stage class 0
STAGE_OF_SKILL = {sid: i + 1 for i, sid in enumerate(SKILL_IDS)}
SKILL_OF_STAGE = {i + 1: sid for i, sid in enumerate(SKILL_IDS)}
N_STAGES = 1 + len(SKILL_IDS)                                # 35
# task_skills is keyed by the HF dir (task-0000); the repo uses underscored
# task names (turning_on_radio) -- index by both.
TASK_SKILLS = {t: set(v) for t, v in _TAX["task_skills"].items()}
for _t, _name in _TAX["task_names"].items():
    if _t in TASK_SKILLS:
        TASK_SKILLS.setdefault(_name.replace(" ", "_"), TASK_SKILLS[_t])

NAV_SKILL_IDS = {1}                                          # "move to"

PREDICATES = ["ontop", "inside", "nextto", "under", "attached", "draped",
              "open", "closed", "toggled_on", "toggled_off", "filled",
              "covered", "saturated", "folded", "unfolded", "cooked",
              "frozen", "hot", "on_fire", "broken", "grasped", "touching",
              "overlaid", "other"]
PRED_IDX = {p: i for i, p in enumerate(PREDICATES)}
N_PREDICATES = len(PREDICATES)
CAT_BUCKETS = 256


def stage_name(stage_idx):
    if stage_idx == IDLE:
        return "idle"
    return SKILL_NAMES[SKILL_OF_STAGE[stage_idx]]


def task_stage_mask(task):
    """[N_STAGES] bool: IDLE + this task's official skills are live."""
    mask = [False] * N_STAGES
    mask[IDLE] = True
    for sid in TASK_SKILLS.get(task, set(SKILL_IDS)):  # unknown task -> all live
        mask[STAGE_OF_SKILL[sid]] = True
    if task not in TASK_SKILLS:
        mask = [True] * N_STAGES
    return mask


def cat_bucket(name):
    """Stable hash bucket for an object/synset category name ('' -> bucket 0)."""
    if not name:
        return 0
    h = hashlib.md5(name.split(".")[0].encode()).digest()
    return 1 + int.from_bytes(h[:4], "little") % (CAT_BUCKETS - 1)


def encode_literals(literals, l_max):
    """literals: [{'predicate','target','reference'}] -> (pred, tgt, ref, mask)
    int/bool lists of length l_max. Truncates beyond l_max (spec cap)."""
    pred = [0] * l_max
    tgt = [0] * l_max
    ref = [0] * l_max
    mask = [False] * l_max
    for i, lit in enumerate(literals[:l_max]):
        pred[i] = PRED_IDX.get(lit.get("predicate", "other"), PRED_IDX["other"])
        tgt[i] = cat_bucket(lit.get("target", ""))
        ref[i] = cat_bucket(lit.get("reference", ""))
        mask[i] = True
    return pred, tgt, ref, mask


def literals_from_task_targets(tt):
    """Approximate literal set from a task_targets.json entry (v1 supervision;
    the director's real BDDL parse replaces this at serve through the same
    encoder). One literal per goal literal count, cycling target x reference."""
    tgts = tt.get("targets") or [""]
    refs = tt.get("references") or [""]
    n = tt.get("n_goal_literals") or max(len(tgts), 1)
    pred = "inside" if any(("box" in r or "bin" in r or "car" in r or
                            "basket" in r or "bag" in r) for r in refs) else "ontop"
    lits = []
    for i in range(n):
        lits.append({"predicate": pred,
                     "target": tgts[i % len(tgts)],
                     "reference": refs[i % len(refs)] if refs else ""})
    return lits


def build_taxonomy(annot_dir, out_path):
    """Rebuild official_taxonomy.json from a full local annotation dump
    (annotations/task-XXXX/episode_*.json)."""
    skills, task_skills, task_names = {}, {}, {}
    for tdir in sorted(os.listdir(annot_dir)):
        full = os.path.join(annot_dir, tdir)
        if not os.path.isdir(full):
            continue
        for f in sorted(os.listdir(full)):
            if not f.endswith(".json"):
                continue
            a = json.load(open(os.path.join(full, f)))
            task_names[tdir] = a.get("task_name", "?")
            ts = task_skills.setdefault(tdir, set())
            for s in a.get("skill_annotation", []):
                for sid, sd in zip(s["skill_id"], s["skill_description"]):
                    skills.setdefault(int(sid), set()).add(sd)
                    ts.add(int(sid))
    out = {"provenance": f"rebuilt from {annot_dir} (full dump)",
           "skills": {str(k): sorted(skills[k])[0] for k in sorted(skills)},
           "task_skills": {t: sorted(v) for t, v in sorted(task_skills.items())},
           "task_names": dict(sorted(task_names.items()))}
    json.dump(out, open(out_path, "w"), indent=1)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--annot_dir", required=True)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "official_taxonomy.json"))
    args = ap.parse_args()
    out = build_taxonomy(args.annot_dir, args.out)
    print(f"{len(out['skills'])} skills, {len(out['task_skills'])} tasks -> {args.out}")

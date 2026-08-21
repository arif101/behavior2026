#!/usr/bin/env python
"""Build the task -> target/reference category mapping for the 100 BEHAVIOR-2026 challenge tasks.

For each task we parse its BDDL goal and classify object synsets as:
  - TARGET   : subject of a state-change/relocation literal (first arg of spatial
               relations like inside/ontop/nextto/under/touching/attached, or the
               argument of unary state predicates like toggled_on/open/cooked/real,
               or the object arg of substance predicates covered/filled/contains).
  - REFERENCE: second arg of spatial relations (container/surface the target
               relates to).
Synsets are mapped to OmniGibson asset categories via the bddl object taxonomy
(output_hierarchy.json, subtree union for non-leaf synsets) and validated
against the asset library category directories.

Quantifiers (forall / exists / forpairs / forn) are expanded; (or ...) branches
are unioned and flagged with has_options. agent.n.01 / floor.n.01 / walls /
room synsets are ignored. Substance synsets (per the 'substance' ability in
output_hierarchy_properties.json) are recorded in notes, not as targets.

Pure CPU, no sim import. Run on the box with the behavior env python:
  /root/miniconda3/envs/behavior/bin/python extract_task_targets.py

Output: JSON {task: {targets, references, target_synsets, reference_synsets,
n_goal_literals, has_options, has_forall, unmapped, notes}}
"""

import argparse
import json
import os
import re
import sys
from collections import Counter

# ---------------------------------------------------------------- defaults
DEF_TASKS_PARQUET = "/root/2026-challenge-demos/meta/tasks.parquet"
DEF_ACTIVITY_DIR = "/root/BEHAVIOR-1K/bddl3/bddl/activity_definitions"
DEF_HIERARCHY = "/root/BEHAVIOR-1K/bddl3/bddl/generated_data/output_hierarchy.json"
DEF_HIERARCHY_PROPS = "/root/BEHAVIOR-1K/bddl3/bddl/generated_data/output_hierarchy_properties.json"
DEF_ASSETS_OBJECTS = "/root/BEHAVIOR-1K/datasets/behavior-1k-assets/objects"
DEF_OUT = "/root/task_targets.json"

# Binary predicates "(pred X Y)" where X is moved/placed relative to Y.
SPATIAL_BINARY = {
    "inside", "ontop", "nextto", "under", "touching", "attached",
    "overlaid", "draped", "on", "onfloor", "broken_into",
}
# Binary predicates "(pred OBJ SUBSTANCE)": OBJ's state changes, 2nd arg is a
# particle system (covered obj dust / filled pot water / contains bag popcorn).
SUBSTANCE_BINARY = {"covered", "filled", "contains", "saturated", "insource"}
# Unary state-change predicates: the argument is a target.
UNARY_STATE = {
    "open", "closed", "toggled_on", "toggled_off", "cooked", "frozen",
    "on_fire", "real", "future", "folded", "unfolded", "hot", "burnt",
    "broken", "assembled", "attached_to_wall", "hung", "sliced", "grasped",
}
QUANTIFIERS = {"forall", "exists", "forpairs", "forn"}
UNIVERSAL_QUANTIFIERS = {"forall", "forpairs", "forn"}
CONNECTIVES = {"and", "or", "not", "imply"}

IGNORE_SYNSETS = {"agent.n.01", "floor.n.01", "wall.n.01", "ceiling.n.01"}
ROOM_RE = re.compile(r"^(?:.*room|kitchen|bathroom|bedroom|garage|corridor|closet|pantry|garden)\.n\.\d+$")
SYNSET_RE = re.compile(r"^(.+\.n\.\d+)(?:_(\d+))?$")


# ---------------------------------------------------------------- s-expression parsing
def tokenize(text):
    text = re.sub(r";[^\n]*", "", text)  # strip line comments
    return re.findall(r"\(|\)|[^\s()]+", text)


def parse_sexpr(tokens, i=0):
    """Parse token list -> nested lists of strings. Returns (node, next_index)."""
    if tokens[i] != "(":
        return tokens[i], i + 1
    i += 1
    out = []
    while tokens[i] != ")":
        node, i = parse_sexpr(tokens, i)
        out.append(node)
    return out, i + 1


def parse_bddl(path):
    """Return (objects: {instance -> synset}, goal_expr)."""
    tokens = tokenize(open(path).read())
    tree, _ = parse_sexpr(tokens)
    objects, goal = {}, None
    for section in tree:
        if not isinstance(section, list) or not section:
            continue
        if section[0] == ":objects":
            # sequence of "inst1 inst2 ... - synset" groups
            pending = []
            it = iter(section[1:])
            for tok in it:
                if tok == "-":
                    syn = next(it)
                    for inst in pending:
                        objects[inst] = syn
                    pending = []
                else:
                    pending.append(tok)
        elif section[0] == ":goal":
            goal = section[1] if len(section) == 2 else ["and"] + section[1:]
    return objects, goal


# ---------------------------------------------------------------- goal walking
def term_to_synset(term, env, objects):
    """Resolve a goal term (?var / ?instance / instance) to its synset."""
    name = term.lstrip("?")
    if name in env:
        return env[name]
    if name in objects:
        return objects[name]
    m = SYNSET_RE.match(name)
    if m:
        return m.group(1)
    return None


def parse_var_binding(binding):
    """(?var - synset) -> (var, synset)."""
    var = binding[0].lstrip("?")
    assert binding[1] == "-", f"bad binding {binding}"
    return var, binding[2]


class GoalWalker:
    def __init__(self, objects):
        self.objects = objects
        self.targets = []       # synsets (ordered, deduped later)
        self.references = []
        self.substances = []    # 2nd args of covered/filled/contains/...
        self.ignored = []       # floor/agent/etc. hits
        self.ignored_targets = []  # ignored synsets that were in TARGET position
        self.n_literals = 0
        self.has_or = False
        self.quantifiers = set()
        self.unknown_predicates = set()

    def resolve(self, term, env):
        syn = term_to_synset(term, env, self.objects)
        if syn is None:
            self.unknown_predicates.add(f"unresolved-term:{term}")
        return syn

    def _add(self, bucket, syn):
        if syn is None:
            return
        if syn in IGNORE_SYNSETS or ROOM_RE.match(syn):
            self.ignored.append(syn)
            if bucket is self.targets:
                self.ignored_targets.append(syn)
            return
        bucket.append(syn)

    def walk(self, expr, env):
        if not isinstance(expr, list) or not expr:
            return
        head = expr[0]
        if head == "not":
            self.walk(expr[1], env)
        elif head in ("and", "or"):
            if head == "or":
                self.has_or = True
            for sub in expr[1:]:
                self.walk(sub, env)
        elif head == "imply":
            # condition (expr[1]) is not a demanded state change; only walk consequence
            self.walk(expr[2], env)
        elif head in ("forall", "exists"):
            var, syn = parse_var_binding(expr[1])
            self.quantifiers.add(head)
            self.walk(expr[2], {**env, var: syn})
        elif head == "forpairs":
            var1, syn1 = parse_var_binding(expr[1])
            var2, syn2 = parse_var_binding(expr[2])
            self.quantifiers.add(head)
            self.walk(expr[3], {**env, var1: syn1, var2: syn2})
        elif head == "forn":
            # (forn (N) (?var - synset) body)
            var, syn = parse_var_binding(expr[2])
            self.quantifiers.add(head)
            self.walk(expr[3], {**env, var: syn})
        else:
            self.literal(head, expr[1:], env)

    def literal(self, pred, args, env):
        self.n_literals += 1
        if pred in SPATIAL_BINARY and len(args) >= 2:
            self._add(self.targets, self.resolve(args[0], env))
            self._add(self.references, self.resolve(args[1], env))
        elif pred in SUBSTANCE_BINARY and len(args) >= 2:
            self._add(self.targets, self.resolve(args[0], env))
            syn = self.resolve(args[1], env)
            if syn:
                self.substances.append(syn)
        elif pred in UNARY_STATE:
            self._add(self.targets, self.resolve(args[0], env))
        else:
            self.unknown_predicates.add(pred)
            # conservative fallback: treat like a spatial literal
            if args:
                self._add(self.targets, self.resolve(args[0], env))
            if len(args) > 1:
                self._add(self.references, self.resolve(args[1], env))


# ---------------------------------------------------------------- taxonomy
def build_synset_maps(hierarchy_path, props_path):
    """Return (synset -> sorted subtree categories, set of substance synsets)."""
    syn_to_cats = {}

    def walk(node):
        cats = set(node.get("categories") or [])
        for child in node.get("children", []):
            cats |= walk(child)
        name = node["name"]
        syn_to_cats.setdefault(name, set()).update(cats)
        return cats

    walk(json.load(open(hierarchy_path)))

    substance_synsets = set()

    def walk_props(node):
        if "substance" in node.get("abilities", {}):
            substance_synsets.add(node["name"])
        for child in node.get("children", []):
            walk_props(child)

    walk_props(json.load(open(props_path)))
    return syn_to_cats, substance_synsets


def load_task_names(parquet_path):
    try:
        import pandas as pd
        return list(pd.read_parquet(parquet_path).index)
    except ImportError:
        import pyarrow.parquet as pq
        t = pq.read_table(parquet_path)
        # index stored as a column named 'task' or '__index_level_0__'
        for col in ("task", "__index_level_0__"):
            if col in t.column_names:
                return t.column(col).to_pylist()
        raise RuntimeError(f"cannot find task-name column in {t.column_names}")


def dedup(seq):
    return list(dict.fromkeys(seq))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tasks-parquet", default=DEF_TASKS_PARQUET)
    ap.add_argument("--activity-dir", default=DEF_ACTIVITY_DIR)
    ap.add_argument("--hierarchy", default=DEF_HIERARCHY)
    ap.add_argument("--hierarchy-props", default=DEF_HIERARCHY_PROPS)
    ap.add_argument("--assets-objects", default=DEF_ASSETS_OBJECTS)
    ap.add_argument("--out", default=DEF_OUT)
    args = ap.parse_args()

    tasks = load_task_names(args.tasks_parquet)
    syn_to_cats, substance_synsets = build_synset_maps(args.hierarchy, args.hierarchy_props)
    asset_categories = set(os.listdir(args.assets_objects))

    results = {}
    problem_tasks = {}

    for task in tasks:
        bddl = os.path.join(args.activity_dir, task, "problem0.bddl")
        if not os.path.exists(bddl):
            problem_tasks[task] = "missing problem0.bddl"
            results[task] = {"targets": [], "references": [], "target_synsets": [],
                             "reference_synsets": [], "n_goal_literals": 0,
                             "has_options": False, "has_forall": False,
                             "unmapped": [], "notes": "missing problem0.bddl"}
            continue

        objects, goal = parse_bddl(bddl)
        walker = GoalWalker(objects)
        walker.walk(goal, {})

        target_synsets = dedup(walker.targets)
        reference_synsets = dedup(walker.references)

        notes, unmapped = [], []

        def synsets_to_categories(synsets):
            cats = []
            for syn in synsets:
                if syn in substance_synsets:
                    notes.append(f"substance synset (particle system, no asset category): {syn}")
                    continue
                sc = syn_to_cats.get(syn)
                if not sc:
                    unmapped.append(syn)
                    notes.append(f"unmapped synset (no category in taxonomy): {syn}")
                    continue
                bad = sorted(c for c in sc if c not in asset_categories)
                for c in bad:
                    unmapped.append(c)
                    notes.append(f"category not in assets dir: {c} (from {syn})")
                good = sorted(c for c in sc if c in asset_categories)
                if len(good) > 15:
                    notes.append(f"BROAD synset {syn} -> {len(good)} categories; "
                                 "restrict to actual scene instances downstream")
                cats.extend(good)
            return dedup(cats)

        targets = synsets_to_categories(target_synsets)
        references = synsets_to_categories(reference_synsets)

        if walker.substances:
            notes.append("goal substances: " + ", ".join(dedup(walker.substances)))
        if walker.ignored:
            notes.append("ignored synsets in goal: " + ", ".join(dedup(walker.ignored)))
        if walker.quantifiers:
            notes.append("quantifiers: " + ", ".join(sorted(walker.quantifiers)))
        if walker.unknown_predicates:
            notes.append("UNKNOWN predicates (fallback classification): "
                         + ", ".join(sorted(walker.unknown_predicates)))
            problem_tasks[task] = "unknown predicates: " + ", ".join(sorted(walker.unknown_predicates))
        if not targets:
            if walker.ignored_targets and walker.substances:
                notes.append("cleaning task: state-change subject is an ignored synset ("
                             + ", ".join(dedup(walker.ignored_targets))
                             + "); goal = remove substance, no manipulable target category")
                problem_tasks[task] = "no target category (substance-removal on ignored surface)"
            else:
                problem_tasks[task] = problem_tasks.get(task, "") + " NO mapped target categories"
        if unmapped:
            problem_tasks.setdefault(task, "unmapped: " + ", ".join(dedup(unmapped)))

        results[task] = {
            "targets": targets,
            "references": references,
            "target_synsets": target_synsets,
            "reference_synsets": reference_synsets,
            "n_goal_literals": walker.n_literals,
            "has_options": walker.has_or,
            "has_forall": bool(walker.quantifiers & UNIVERSAL_QUANTIFIERS),
            "unmapped": dedup(unmapped),
            "notes": "; ".join(dedup(notes)),
        }

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out} ({len(results)} tasks)")

    # ------------------------------------------------------------ validation
    radio = results.get("turning_on_radio", {})
    ok = radio.get("targets") == ["radio"]
    print(f"\nVALIDATION turning_on_radio targets == ['radio']: {'PASS' if ok else 'FAIL'} -> {radio.get('targets')}")

    n_unmapped = sum(1 for r in results.values() if r["unmapped"])
    n_full = sum(1 for r in results.values() if r["targets"] and not r["unmapped"])
    tgt_dist = Counter(len(r["targets"]) for r in results.values())
    print(f"\nSUMMARY: {len(results)} tasks | fully mapped (>=1 target, 0 unmapped): {n_full} | with unmapped synsets/categories: {n_unmapped}")
    print("targets-per-task distribution: " + ", ".join(f"{k}:{v}" for k, v in sorted(tgt_dist.items())))
    print(f"has_options: {sum(r['has_options'] for r in results.values())} | has_forall: {sum(r['has_forall'] for r in results.values())}")

    if problem_tasks:
        print(f"\nPROBLEM TASKS ({len(problem_tasks)}) needing manual attention:")
        for t, why in sorted(problem_tasks.items()):
            print(f"  {t}: {why.strip()}")
    else:
        print("\nno problem tasks")


if __name__ == "__main__":
    main()

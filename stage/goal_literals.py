"""Real goal literals from BDDL problem definitions (v4).

Replaces taxonomy.literals_from_task_targets's inside/ontop guess with the
task's actual goal: each literal keeps its true predicate (toggled_on, inside,
ontop, covered, ...), its target/reference synsets, the asset categories they
map to (for scene binding + name-prefix matching), and its quantifier context.
The ORDER of the emitted list is the ledger's literal indexing everywhere:
literals.json, the p_sat targets from eval_predicates.py, and serve.

S-expression parsing and goal walking follow probes/extract_task_targets.py
(same tokenizer/walker semantics, validated over the 100 challenge tasks);
this module differs in emitting ordered literal specs rather than
target/reference category buckets.

Box usage (CPU, no sim -- needs the BEHAVIOR-1K clone for bddl3):
  python3 goal_literals.py --tasks turning_on_radio picking_up_trash ... \
      --out /root/task_literals.json

Output: {task: {"literals": [{predicate, target, reference, target_synset,
reference_synset, target_cats, reference_cats, quantifier, or_group}],
"n_goal_literals": int}}  -- capped at L_MAX per task, spec ledger cap.
"""

import argparse
import json
import os
import re

from common import L_MAX
from taxonomy import PRED_IDX

DEF_ACTIVITY_DIR = "/root/BEHAVIOR-1K/bddl3/bddl/activity_definitions"
DEF_HIERARCHY = "/root/BEHAVIOR-1K/bddl3/bddl/generated_data/output_hierarchy.json"

# BDDL predicate name -> taxonomy.PREDICATES vocab entry.
PRED_MAP = {
    "on": "ontop", "onfloor": "ontop", "attached_to_wall": "attached",
    "hung": "attached", "contains": "filled", "insource": "filled",
    "overlaid": "overlaid", "draped": "draped",
}

QUANTIFIERS = {"forall", "exists", "forpairs", "forn"}
IGNORE_SYNSETS = {"agent.n.01", "floor.n.01", "wall.n.01", "ceiling.n.01"}
ROOM_RE = re.compile(r"^(?:.*room|kitchen|bathroom|bedroom|garage|corridor|closet|pantry|garden)\.n\.\d+$")
SYNSET_RE = re.compile(r"^(.+\.n\.\d+)(?:_(\d+))?$")


# ---- s-expression parsing (extract_task_targets.py conventions) ----
def tokenize(text):
    text = re.sub(r";[^\n]*", "", text)
    return re.findall(r"\(|\)|[^\s()]+", text)


def parse_sexpr(tokens, i=0):
    if tokens[i] != "(":
        return tokens[i], i + 1
    i += 1
    out = []
    while tokens[i] != ")":
        node, i = parse_sexpr(tokens, i)
        out.append(node)
    return out, i + 1


def parse_bddl(path):
    tokens = tokenize(open(path).read())
    tree, _ = parse_sexpr(tokens)
    objects, goal = {}, None
    for section in tree:
        if not isinstance(section, list) or not section:
            continue
        if section[0] == ":objects":
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


def term_to_synset(term, env, objects):
    name = term.lstrip("?")
    if name in env:
        return env[name]
    if name in objects:
        return objects[name]
    m = SYNSET_RE.match(name)
    return m.group(1) if m else None


def parse_var_binding(binding):
    var = binding[0].lstrip("?")
    assert binding[1] == "-", f"bad binding {binding}"
    return var, binding[2]


class LiteralWalker:
    """Emit ordered literal specs from a goal expression."""

    def __init__(self, objects):
        self.objects = objects
        self.literals = []
        self._quant = []          # active quantifier stack
        self._or_group = -1       # current (or ...) group id, -1 = none
        self._n_or = 0

    def _ignored(self, syn):
        return syn is None or syn in IGNORE_SYNSETS or ROOM_RE.match(syn)

    def walk(self, expr, env):
        if not isinstance(expr, list) or not expr:
            return
        head = expr[0]
        if head == "not":
            self.walk(expr[1], env)          # negation kept as same literal slot
        elif head == "and":
            for sub in expr[1:]:
                self.walk(sub, env)
        elif head == "or":
            gid, self._or_group = self._or_group, self._n_or
            self._n_or += 1
            for sub in expr[1:]:
                self.walk(sub, env)
            self._or_group = gid
        elif head == "imply":
            self.walk(expr[2], env)
        elif head in ("forall", "exists"):
            var, syn = parse_var_binding(expr[1])
            self._quant.append(head)
            self.walk(expr[2], {**env, var: syn})
            self._quant.pop()
        elif head == "forpairs":
            var1, syn1 = parse_var_binding(expr[1])
            var2, syn2 = parse_var_binding(expr[2])
            self._quant.append(head)
            self.walk(expr[3], {**env, var1: syn1, var2: syn2})
            self._quant.pop()
        elif head == "forn":
            var, syn = parse_var_binding(expr[2])
            self._quant.append(("forn", int(expr[1][0])))
            self.walk(expr[3], {**env, var: syn})
            self._quant.pop()
        else:
            self.literal(head, expr[1:], env)

    def literal(self, pred, args, env):
        tgt = term_to_synset(args[0], env, self.objects) if args else None
        ref = term_to_synset(args[1], env, self.objects) if len(args) > 1 else None
        if self._ignored(tgt):
            return
        q = self._quant[-1] if self._quant else ""
        n = 0
        if isinstance(q, tuple):
            q, n = q
        self.literals.append(dict(
            predicate=pred,
            target_synset=tgt,
            reference_synset=None if self._ignored(ref) else ref,
            quantifier=q, forn_n=n, or_group=self._or_group))


def build_syn_to_cats(hierarchy_path):
    """synset -> sorted subtree asset categories (extract_task_targets walk)."""
    syn_to_cats = {}

    def walk(node):
        cats = set(node.get("categories") or [])
        for child in node.get("children", []):
            cats |= walk(child)
        syn_to_cats.setdefault(node["name"], set()).update(cats)
        return cats

    walk(json.load(open(hierarchy_path)))
    return {k: sorted(v) for k, v in syn_to_cats.items()}


def literal_specs(task, activity_dir, syn_to_cats):
    """Ordered, L_MAX-capped literal list for one task."""
    bddl = os.path.join(activity_dir, task, "problem0.bddl")
    objects, goal = parse_bddl(bddl)
    w = LiteralWalker(objects)
    w.walk(goal, {})

    out = []
    for lit in w.literals[:L_MAX]:
        pred = PRED_MAP.get(lit["predicate"], lit["predicate"])
        if pred not in PRED_IDX:
            pred = "other"
        tcats = syn_to_cats.get(lit["target_synset"]) or []
        rcats = (syn_to_cats.get(lit["reference_synset"]) or []) if lit["reference_synset"] else []
        stem = lambda s: s.split(".")[0] if s else ""
        out.append(dict(
            predicate=pred,
            # representative category name feeds cat_bucket + name-prefix match
            target=tcats[0] if tcats else stem(lit["target_synset"]),
            reference=rcats[0] if rcats else stem(lit["reference_synset"]),
            target_synset=lit["target_synset"],
            reference_synset=lit["reference_synset"],
            target_cats=tcats or [stem(lit["target_synset"])],
            reference_cats=rcats or ([stem(lit["reference_synset"])]
                                     if lit["reference_synset"] else []),
            quantifier=lit["quantifier"], forn_n=lit["forn_n"],
            or_group=lit["or_group"]))
    return dict(literals=out, n_goal_literals=len(w.literals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--activity_dir", default=DEF_ACTIVITY_DIR)
    ap.add_argument("--hierarchy", default=DEF_HIERARCHY)
    ap.add_argument("--out", default="/root/task_literals.json")
    args = ap.parse_args()

    syn_to_cats = build_syn_to_cats(args.hierarchy)
    out = {}
    for task in args.tasks:
        out[task] = literal_specs(task, args.activity_dir, syn_to_cats)
        lits = out[task]["literals"]
        preds = [l["predicate"] for l in lits]
        print(f"{task}: {len(lits)} literals ({out[task]['n_goal_literals']} in goal) "
              f"preds={sorted(set(preds))}")
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

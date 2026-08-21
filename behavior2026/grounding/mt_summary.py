"""Combine v0.5 results JSONs into the final v2-vs-v3 comparison tables."""

import json
import sys

OUT = "/root/grounding_v05"


def load(name):
    try:
        return json.load(open(f"{OUT}/{name}"))
    except FileNotFoundError:
        return None


def row(v):
    if not v or not v.get("n"):
        return "     -"
    return "%5d %7.1f %7.1f %6.1f %8.3f" % (v["n"], v["px_median"], v["px_p90"],
                                            v["within_30px_pct"], v["depth_mae_m"])


def main():
    r2 = load("results_v05_dinov2.json")
    r3 = load("results_v05_dinov3.json")
    rc = load("results_v05_radio_control.json")

    for split, key in [("HELD-OUT EPISODES (24 train tasks)", "heldout_ep_per_task"),
                       ("HELD-OUT TASKS (4)", "heldout_task_per_task")]:
        print(f"\n==== {split} ====")
        print("%-44s | %-38s | %-38s" % ("task", "DINOv2: n px_med p90 w30% dMAE",
                                         "DINOv3: n px_med p90 w30% dMAE"))
        tasks = sorted((r2 or r3)[key])
        for t in tasks:
            print("%-44s | %38s | %38s" % (
                t, row(r2[key].get(t)) if r2 else "-",
                row(r3[key].get(t)) if r3 else "-"))
        okey = key.replace("_per_task", "_overall")
        print("%-44s | %38s | %38s" % ("OVERALL",
              row(r2[okey]) if r2 else "-", row(r3[okey]) if r3 else "-"))

    if r2:
        print("\nseen-vs-unseen categories on held-out tasks (v2):",
              json.dumps({"seen": r2["heldout_task_seen_cats"],
                          "unseen": r2["heldout_task_unseen_cats"]}, indent=1))
    if r3:
        print("seen-vs-unseen categories on held-out tasks (v3):",
              json.dumps({"seen": r3["heldout_task_seen_cats"],
                          "unseen": r3["heldout_task_unseen_cats"]}, indent=1))
    if rc:
        radio = rc["heldout_ep_per_task"].get("turning_on_radio")
        print("\nradio single-task control (4 eps, heldout ep):", row(radio))
    if r2:
        radio2 = r2["heldout_ep_per_task"].get("turning_on_radio")
        print("radio in multi-task v2 (same heldout ep):      ", row(radio2))
        print("pilot (18 eps, single-task, ungated labels): px_med 4.6 (banked)")

    # worst categories by px_median (v2, heldout episodes), n>=30
    if r2:
        cats = [(v["px_median"], k, v) for k, v in r2["heldout_ep_per_category"].items()
                if v.get("n", 0) >= 30]
        cats.sort(reverse=True)
        print("\nworst categories (v2, heldout eps, n>=30):")
        for px, k, v in cats[:12]:
            print("  %-32s n=%4d px_med=%7.1f w30=%5.1f" % (k, v["n"], px,
                                                            v["within_30px_pct"]))


if __name__ == "__main__":
    main()

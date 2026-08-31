# 🏀 Torso chain sweep — final tally: 4 instances complete, aimed press 3-for-4

**Simple version:** Last night our robot learned to press the radio's button *on purpose* (torso bent, fingertip aimed at the actual button) instead of relying on the radio getting bumped during set-down. Today we road-tested that recipe on four episodes it had never seen. It went 2-for-3 on presses, and the one miss wasn't the press's fault — the radio had been set down on the table's edge and toppled into an unreachable gap. NBA analogy: our kicker had only ever made field goals in his home stadium. Today he went on the road — made them in two stadiums, and the one miss came because the *holder* placed the ball off the tee. One game was also called off before kickoff on a technicality (a two-frame timing offset) that we already solved months ago in a different part of our system.

---

**PhD version:** The round-51 chain (streak-gated verified weld → transport replay → carry-to-reachable → set-down → release → toggle-state check → 11-DOF arm+trunk servo press on the live button metalink) ran unmodified on d100/d190/d260/d310. Results: d100 TOGGLED (it=9, best_btn=4.28cm), d190 WELD_FAILED (chain hardcodes t0=closure+6; factory meta says this demo needs +4 — the streak gate honestly refused certification after one condition flicker), d260 NOT toggled (set-down released the radio perched on the table lip at z=0.469; it toppled into the table–couch gap mid-press; servo cos went hard negative fighting an unreachable target through the glass), d310 TOGGLED (it=141, best_btn=4.06cm). Campaign tally: chain complete on 4 radio instances (d30 incidental ×3 repro, d20/d100/d310 aimed). The three aimed toggles share a ~4.1–4.3cm best_btn contact signature — consistent fingertip-to-button-center geometry, i.e., a mechanism, not luck. Both failures are pre-press: one anchor bookkeeping, one set-down placement. The press itself is 3-for-3 whenever its target was on a support surface. Notably d20's toggle was of a floor-level radio after an unplanned fall — the trunk block extends the press envelope well below any demo's press station.

---

**GLOSSARY** — *Verified weld*: our grasp certification — the eval's assisted-grasp conditions (contact + between-fingers ray + sustained closing) must hold for ~300 consecutive physics steps before we attach the object. *Aimed press vs incidental*: servoing the fingertip to the measured button position vs the button firing as a side effect of set-down contact. *best_btn*: closest logged fingertip-to-button distance before contact. *Anchor (t0)*: which demo frame we restore physics from; off-by-two frames can break grasp-condition timing. *11-DOF servo*: Jacobian control over 7 right-arm joints + 4 trunk joints — the trunk was frozen in all 50 prior press attempts and was the hidden root cause.

---

**RESOURCES** — Film page ("The First Grasp," new section *The aimed press*): https://claude.ai/code/artifact/8a4b5888-bcd6-4f13-a8ca-cf4bd787852c · 2025 winner's playbook we're mining for Run-3: https://github.com/IliaLarchenko/behavior-1k-solution · Time-reversal augmentation precedent for the un-grasp factory: https://arxiv.org/abs/2505.13925

---

**NEXT STEPS** — 1) Chain fixes: read per-demo anchor from factory meta; support-surface raycast + upright/settled gate before press; per-demo film dirs. 2) Re-run d190 (expect 5th instance). 3) Round 52: in-hand press — human playbook (hold + torso down + left-hand press), now viable since both historical blockers (frozen trunk, ghost-button retargeting) are solved. 4) Policy relay: spawn trained π0.5 at post-transport state, test whether it can finish (isolates the data poison to the grasp segment). 5) v2 3-camera obs batch overnight → converter → LeRobot packing → Run-3 A/B.

---

**HUMAN INPUT NEEDED** — None blocking. Optional: office-hours question list is drafted (pull provenance, incidental-toggle scoring, v3.9.2 scope) — say the word if you want it posted anywhere before the session.

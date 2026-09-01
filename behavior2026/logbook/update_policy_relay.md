# 🏈 The policy can finish the play: π0.5 relay 3-for-3, two instances

**Simple version:** Big one. We took our trained robot brain (the one that always freezes right before grabbing the radio) and handed it the robot AFTER our manufactured honest grasp — radio already in hand. It finished the job on its own, three times out of three, on two different room layouts: carried the radio steady, reached over with its free hand, and pressed the power button. NFL analogy: our QB fumbled every snap in practice because the practice tapes had a rigged center exchange — but once you put the ball cleanly in his hands, he throws touchdowns. The tapes taught him everything except the snap. We now know exactly which skill to manufacture (the snap), and we have a factory for it.

---

**PhD version:** Relay protocol: replay a factory clip's certified command stream (verified weld, honest physics) to the post-transport state, then hand control to the radio_run2 π0.5 checkpoint via file-RPC (openpi env, 0.3s warm infer; 16-of-32 chunk execution with re-infer; zero map tokens = trained null convention; no affordance points). Results: d20 toggled at step 99 (run 1) and step 214 (independent stochastic draw), d100 at step 226 — all three with the weld intact (`ag_end=true`), radio altitude held, per-step contact attribution at the toggle instant naming the LEFT fingertips (5.6–6.8cm mean-link-to-metalink-origin = fingertip-on-dome; right hand 17–18cm away at the grip). The policy's chosen strategy is the in-hand press — the human playbook — decisively refuting incidental-toggle explanations. Interpretation: (1) behavioral confirmation that demo poison is confined to the grasp segment — transport and press competence transferred from BC on human data; (2) the grasp-segment corpus is the correct product for Run-3's demos+corpus arm; (3) a second episode factory exists: scripted honest grasp + learned finish = complete, style-consistent task episodes with no press scripting at all. Overnight, the set-down chain's d190/d260 failures were root-caused to in-hand attitude tilt at release (edge-touch tip-over) — moot under the in-hand strategy the policy itself prefers.

---

**GLOSSARY** — *Policy relay*: spawning the trained policy mid-task at a known-good state to isolate which segment of the task it can/cannot perform. *Handoff state*: post-transport — radio welded in hand, robot at the demo's press neighborhood. *Action chunk*: π0.5 emits 32 future actions per inference; we execute 16 then re-infer. *Stochastic draws*: flow-matching sampling injects noise, so each run is an independent trial, not a replay. *Contact attribution*: per-step distance from each hand's finger links to the button metalink; the toggle can only be fired by a finger link inside the button volume.

---

**RESOURCES** — Film page (relay clips being added): https://claude.ai/code/artifact/8a4b5888-bcd6-4f13-a8ca-cf4bd787852c · Repo: github.com/arif101/behavior2026 (rt_relay_test1.py, relay_worker.py, rt_replay_test52.py) · 2025 winner's π0.5 recipe: github.com/IliaLarchenko/behavior-1k-solution

---

**NEXT STEPS** — 1) v2 obs batch to completion (12/38 banked, all bit-exact) → full conversion → LeRobot packing. 2) R52 scripted in-hand press test (the policy just endorsed the strategy; scripted version gives it to demos where the policy fails). 3) Relay-based episode factory: add obs capture to relay runs → full episodes with honest grasps AND learned presses as training data. 4) A100 request → Run-3 A/B (demos vs demos+corpus), bar = 26%. 5) Relay N-trial success-rate measurement across more instances for a real number.

---

**HUMAN INPUT NEEDED** — None blocking. Heads-up: an A100 (or similar) training request is imminent once packing completes — likely within ~2 days.

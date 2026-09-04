# 🎬 The whole chain, no assist — on film

**Simple version:** For the first time, one continuous episode with nothing touched by anything but the robot: the hand starts a foot away from the radio sitting on the table, reaches over (torso and all), lines up its fingers on the handle, closes, the grasp certifies under the same rule the competition's grasp-assist uses, it carries the radio in, moves it where the human demos moved it — and then the trained policy takes the robot and presses the power button itself. NFL analogy: we finally have game film of the full drive, snap to touchdown, with no holding penalty. Until yesterday every clip started with the ball already in the QB's hands.

Watch: https://claude.ai/code/artifact/8a4b5888-bcd6-4f13-a8ca-cf4bd787852c (top section, "The whole chain, no assist").

---

**PhD version:** Approach factory v7 (restore closure−30, pre-pull; 3-probe native-J row + validated angular block + 4-channel trunk probe; stage 10 cm above the certified grasp pose; gentle orient-first at the staging point, ≤0.03 rad/step with a position-hold gain — converged to 0.031 rad in 42 iterations with 5.6 mm drift; descent arriving at 0.005 rad; 1.5 cm compliance push; finger-midpoint ALIGN to the certified finger positions in the radio frame; closure → VERIFIED_WELD k=425 streak 296; carry-in along the pull vector; posture reset; transport replay) produced a 601-step single-pass episode with the causal-honesty certificate (pre-contact radio displacement 1.85 cm — the push's own touch a sensor-step early; total 3.5 cm; first contact at the deliberate push). The π0.5 relay on that clip toggled on draw 2 at policy step 130 with the weld intact → full episode rac_20_301 (746 steps). Measured why v5 clips failed the policy 7/7: in-hand attitude 12.7° off the demo grip (wrist residual 0.384 rad at closure); v7 halves it to 6.4° / 1.9 cm at the button, inside the policy's tolerance on this instance. The learned press is attitude-sensitive — a measured policy weakness that honest approach data across grip attitudes is positioned to fix. Also measured: the rig's assist varies per demo (d30 0.26 m, d20 0.35 m, d10 1.04 m — the human's hand stopped a meter short); arm+trunk reach covers ≲0.55 m, beyond that the honest approach needs base motion first.

---

**GLOSSARY** — *Approach factory*: manufactures the pre-contact segment the human demos never contain (the rig slid the object in). *Orient-first*: converge wrist attitude in open air before descending. *Finger ALIGN*: target the finger positions in the object frame, not the wrist. *Honesty certificate*: object undisturbed until the deliberate touch (pre-contact displacement) + verified weld + bit-exact re-render. *Relay*: hand the trained policy the robot mid-episode and let it finish.

---

**RESOURCES** — Film page: https://claude.ai/code/artifact/8a4b5888-bcd6-4f13-a8ca-cf4bd787852c · Press and Topple (policy press vs scripted set-down): https://claude.ai/code/artifact/7f9e2f5a-ae5b-4b57-9b2b-36d0bc713f19 · Repo: github.com/arif101/behavior2026 (factory_approach_cap_v7.py, relay_episode_factory.py)

---

**NEXT STEPS** — 1) Night run continues on v7 across the reachable demos (reach gate 0.55 m; honesty gate). 2) Convert the approach corpus → b1k_radio_approach (Run-3 source 4, arm A5). 3) v8 (carry servos the welded radio's pose onto the demo's post-pull pose) to close the last 6° for higher relay yield. 4) Base-drive approach variant for the long-pull demos. 5) Pack Run-3 with all four sources + poison-window down-weighting → A100.

**HUMAN INPUT NEEDED** — Training compute timing: the four-source Run-3 mix packs as soon as the approach corpus converts (this weekend).

# 🏭 Episode factory, night one: 58 complete task episodes manufactured

**Simple version:** Overnight the factory ran on autopilot: for each of 37 demos, replay our honest manufactured grasp, then hand the robot to the trained policy three times and let it try to finish the job. Every time it pressed the button, we saved the whole thing as a training episode — an honest grasp followed by a learned press. Result: **58 complete episodes** from 111 attempts. NFL analogy: we ran the same play 111 times from a clean snap. The QB converted 52% overall — but here's the tell: on some fields he went 3-for-3 every time, and on others 0-for-3 every time. It's not a coin flip. Some formations he's simply learned, and some he hasn't — and now we have the list.

---

**PhD version:** Protocol: certified factory clip replay to the post-transport handoff (weld re-established at the recorded step), `og.sim.dump_state` snapshot, then 3 independent policy draws (radio_run2 π0.5, 16-of-32 chunk execution, budget 600 steps) via `load_state` restore (~1s vs ~30min re-replay). Full v2 obs captured per step (3 cams RGB+depth 1080², 61-proprio, base pose, radio pose). Save gate: `toggled_on ∧ weld intact`. Results: 58/111 draws toggled (52%); 23/37 demos ≥1 episode; **bimodal at the instance level** — 16 demos 3/3, 15 demos 0/3, only 6 in between. Zero-yield instances: d50, d70, d80, d160, d170, d210–230, d290, d300, d320, d390–410. This is deployment-distribution data in the PLD sense (base policy states, on-policy tails) with a scripted, causally-honest prefix. The bimodality is diagnostic gold for Run-3: it partitions instances into "handoff state is inside the policy's competence" vs "outside," and the outside set is exactly where segment-clip data (honest grasp only, policy learns the rest from demos) should matter most. Notable: d260 — where the scripted set-down chain toppled the radio in 3 of 3 runs — went 3/3 with the policy pressing in hand.

Infrastructure lesson banked: the box is a 50GB-cgroup container (`free` shows the 503GB host); the LeRobot converter OOM-died silently twice holding all clips' frames. Patched to lazy per-episode frame loading; conversions now sequenced after the factory releases memory.

---

**GLOSSARY** — *Handoff state*: post-transport, radio welded in hand, where scripted control ends and the policy begins. *Draw*: one stochastic policy rollout (flow-matching sampling noise makes each independent). *Snapshot restore*: serialize/deserialize full sim state to reset to the handoff in ~1s. *Bimodal yield*: per-instance success clustering at 0 or 100% rather than spreading around the mean — indicates deterministic solvability per state, not luck. *cgroup cap*: container memory limit invisible to `free`.

---

**RESOURCES** — Film: Press and Topple (policy press vs scripted topple, side by side): https://claude.ai/code/artifact/7f9e2f5a-ae5b-4b57-9b2b-36d0bc713f19 · The First Grasp reel: https://claude.ai/code/artifact/8a4b5888-bcd6-4f13-a8ca-cf4bd787852c · Repo: github.com/arif101/behavior2026 (relay_episode_factory.py, ef_night.sh, poison_windows.json)

---

**NEXT STEPS** — 1) Segment dataset conversion (38 clips → /root/b1k_radio_factory) running now; episode conversion (58 → /root/b1k_radio_episodes) sequenced behind it. 2) Verify both datasets (schema, decode, label sanity). 3) Pack Run-3 mix: 200 human demos with the 2.33% poisoned grasp windows down-weighted + segments + episodes as separate arms. 4) A100 request → Run-3 launch. 5) Read the bimodal table against instance geometry to characterize what separates 3/3 from 0/3.

**HUMAN INPUT NEEDED** — Training compute (A100/H100 access) is now the critical-path dependency: packing completes within ~24h.

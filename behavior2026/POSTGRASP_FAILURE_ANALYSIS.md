# Post-grasp failure quantification (interim: A0 n=25 + A4 first 4)

## Question
Why do A2/A4 grasp but never press? Is the grasp->press gap sequencing, data, or perception?

## Finding 1 (definitive, from code): the policy is NEVER guided to the button
The eval task target is the RADIO OBJECT (task_targets: targets=["radio"], synset radio_receiver.n.01).
The AffordanceMapFullRes wrapper injects target_points = (affordance point on the radio) - EE, for
both hands — i.e. a GRASP affordance on the radio body. There is NO target for the toggle/button.
After grasping, the policy's only spatial guidance still points at the radio it is holding. The button
has no representation in the conditioning signal at all.

## Finding 2 (quantitative): no hand ever approaches the button
ee_button_min = closest either EEF gets to the toggle link, over the whole rollout.
- A0 (control, 25 rollouts): 0/25 within 5 cm of the button; min over ALL 25 = 0.151 m.
- A4 (4 so far, 1 grasp):     0/4  within 5 cm of the button; min over ALL   = 0.154 m.
Even in the grasp rollout, the button floor is ~15 cm. A press needs ~contact (<3 cm). Neither hand
ever drives toward the button — consistent with there being no button target to drive toward.

## Finding 3 (behavioral, video, rollout 3): grasp -> release -> re-approach, no button drive
t1:12 grasp (gripper closes, holds) -> t1:32 released, radio back on table -> t1:44 hovering at the
radio's top without toggling. The grasp does not persist into a press; it decays into re-approach.

## Interpretation — the gap is PERCEPTION/TARGET-DESIGN first, sequencing second, NOT temporal-history
The policy isn't "grasping then forgetting to press." It is never pointed at the button, in the obs or
(apparently) from data, so it never approaches it. That is a cheaper problem than a new architecture.

## Revised next steps (cheap -> expensive)
1. Add a BUTTON/TOGGLE affordance target for the press phase (guide EE to the toggle link, not just the
   radio) — eval wrapper + training affordance labels. Biggest lever, no architecture change.
2. B1 stage-prediction aux: know WHEN to switch grasp-target -> button-target. Light sequencing signal.
3. Data rebalance toward the grasp->press hand-off (press currently ~3.7x vs grasp ~11x).
4. B3 Temporal Forcing (4D/history): RESERVE — the failure is not primarily a history problem, so this
   is the wrong FIRST tool. Escalate only if 1-3 plateau.

Status: interim (A0 done, A4 4/25). Re-run /root/postgrasp_analysis.py as A4/A2 complete.

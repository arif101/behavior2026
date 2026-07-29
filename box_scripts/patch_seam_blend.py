"""Seam blending v1: reserved-overlap cross-fade at replan, in act_receding_horizon.

WHY (measured): winning config (oracle points + h32) converts 1/6. All five archived failures
show the same signature — during the MANIPULATION phase, arm-dim |Δaction| at replan seams is
5-14x interior (pooled 8.99). The one success was visibly jittery at the grasp (user review).
Mode-switching between consecutive flow samples mid-contact is what breaks conversion.

WHAT (winner-shaped, simplified): the 2025 winner (arXiv 2512.06951) predicts a chunk, executes
26-28, RESERVES the tail, and constrains the next chunk's head to match through the learned
action covariance. This v1 keeps their reserved-overlap geometry but uses a plain cross-fade:

    serve with --action-horizon 28 (of a 32-chunk) -> at each replan the old chunk has 4
    unexecuted actions. Blend the new chunk's first 4 actions against them with a linear ramp
    w = (t+1)/5 in [0.2, 0.4, 0.6, 0.8], so the handoff is continuous instead of a jump.

    Grippers are EXCLUDED from blending (binary channel; interpolating open/closed mid-grasp is
    worse than either) — they take the NEW chunk's value, matching the wrapper's existing
    gripper-preservation convention.

The blend activates only when the old chunk actually has unexecuted actions at replan time, so
serving with --action-horizon 32 (no reserve) behaves exactly as before — the A/B is clean.

PRE-REGISTERED READ (5-run rate at h28+blend, oracle points):
    seam ratio < 2 in the manipulation phase  AND  rate >= 3/5  -> inpainting is the rate lever
    seam ratio < 2 but rate <= 1/5            -> seams were not the binding constraint;
                                                 terminal skill is weak -> corrective data / RL
"""

import argparse
import pathlib

ANCHOR = """            # Store actions in buffer
            seq_len = min(target_action.shape[1], self.max_len)
"""

PATCH = """            # --- seam blend v1 (patch_seam_blend.py) --------------------------------
            # If the outgoing chunk still has unexecuted actions (reserved overlap, i.e.
            # action_horizon < max_len), cross-fade the new chunk's head against them so the
            # replan handoff is continuous. Grippers excluded (binary channel).
            try:
                import numpy as _np

                K = 4
                if self.action_buffer is not None:
                    for _bi, _gi in enumerate(indices_needing_inference):
                        _idx = int(self.sequence_indices[_gi, 0])
                        _len = int(self.sequence_lengths[_gi, 0])
                        _left = _len - _idx
                        if 0 < _left:
                            _k = min(K, _left, target_action.shape[1])
                            _old = self.action_buffer[_gi, 0, _idx:_idx + _k].copy()
                            _w = ((_np.arange(_k) + 1.0) / (_k + 1.0))[:, None]
                            _blend = (1.0 - _w) * _old + _w * target_action[_bi, :_k]
                            for _g in self.gripper_indices:
                                _blend[:, _g] = target_action[_bi, :_k, _g]
                            target_action[_bi, :_k] = _blend
            except Exception:
                pass
            # ------------------------------------------------------------------------
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrapper", default="/root/openpi_fork/src/openpi/shared/eval_b1k_wrapper.py")
    a = ap.parse_args()
    p = pathlib.Path(a.wrapper)
    s = p.read_text()
    if "patch_seam_blend.py" in s:
        raise SystemExit("already patched")
    if ANCHOR not in s:
        raise SystemExit(f"anchor not found in {p}")
    p.write_text(s.replace(ANCHOR, PATCH + ANCHOR, 1))
    print(f"patched {p}: reserved-overlap cross-fade active when action_horizon < max_len")


if __name__ == "__main__":
    main()

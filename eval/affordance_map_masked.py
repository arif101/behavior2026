"""Causality-ablation wrapper (RUN2_EVAL_PREREG.md guard, n=5): the A2 stack with the
AFFORDANCE POINT MASKED. Everything else identical to AffordanceMapFullRes — the map runs
live and map_tokens still reach the server via _last_map_tokens; the affordance head still
runs (stats record conf as usual, n_inject counts point computations); but the point is
never delivered: obs keys are scrubbed and the evaluator-passthrough attribute is nulled,
so the server's B1KInputs emits its zeros+invalid-mask null-token sentinel.

Pre-registered expectation: DEGRADATION vs A2. If no degradation, the point conditioning
is decorative — flag loudly."""

from behavior2026_eval.affordance_map_fullres import AffordanceMapFullRes


class AffordanceMapPointsMasked(AffordanceMapFullRes):
    def _inject(self, obs):
        out = super()._inject(obs)
        if isinstance(obs, dict):
            obs.pop("target_points", None)
            obs.pop("target_points_mask", None)
        self._last = None  # evaluator point-passthrough reads this; None => nothing attached
        return out

import logging
import torch as th
from omnigibson.eval.utils.network_utils import WebsocketClientPolicy
from typing import Optional


__all__ = [
    "LocalPolicy",
    "WebsocketPolicy",
]


class LocalPolicy:
    """
    Local policy that directly queries action from policy,
        outputs zero delta action if policy is None.
    """

    def __init__(self, *args, action_dim: Optional[int] = None, **kwargs) -> None:
        self.policy = None  # To be set later
        self.action_dim = action_dim

    def set_action_dim(self, action_dim: int) -> None:
        self.action_dim = action_dim

    def act(self, obs: dict) -> th.Tensor:
        return self.forward(obs)

    def forward(self, obs: dict, *args, **kwargs) -> th.Tensor:
        """
        Directly return a zero action tensor of the specified action dimension.
        """
        if self.policy is not None:
            return self.policy.act(obs).detach().cpu()
        else:
            assert self.action_dim is not None
            return th.zeros(self.action_dim, dtype=th.float32)

    def reset(self) -> None:
        if self.policy is not None:
            self.policy.reset()


class WebsocketPolicy:
    """
    Websocket policy for controlling the robot over a websocket connection.
    """

    def __init__(
        self,
        *args,
        host: Optional[str] = None,
        port: Optional[int] = None,
        allow_reconnect: bool = False,
        **kwargs,
    ) -> None:
        logging.info(f"Creating websocket client policy with host: {host}, port: {port}")
        self.last_action = None
        self.policy = None
        self._allow_reconnect = allow_reconnect
        if host is not None or port is not None:
            self.policy = WebsocketClientPolicy(host=host, port=port, allow_reconnect=allow_reconnect)

    def update_host(self, host: str, port: int) -> None:
        self.policy = WebsocketClientPolicy(host=host, port=port, allow_reconnect=self._allow_reconnect)

    _trace_fh = None
    _trace_step = 0

    def _trace(self, obs, action):
        import json as _json, os as _os
        if WebsocketPolicy._trace_fh is None:
            path = _os.environ.get("TRACE_PATH", "/workspace/traces.jsonl")
            WebsocketPolicy._trace_fh = open(path, "a", buffering=1)
        rec = {"t": WebsocketPolicy._trace_step, "action": [round(float(x), 5) for x in action.flatten().tolist()]}
        for k, v in obs.items():
            if hasattr(v, "shape") and getattr(v, "ndim", 0) == 1 and v.shape[0] < 128:
                rec[k.split("::")[-1]] = [round(float(x), 4) for x in v.flatten().tolist()[:8]]
        # T1 proximity extension: target-object poses (every 10 steps) + EE poses (every step).
        # Diagnostic-only; active only when TARGET_CATS is set (comma-separated category substrings).
        try:
            import omnigibson as og
            cats = _os.environ.get("TARGET_CATS", "")
            if cats and og.sim is not None and len(og.sim.scenes):
                sc = og.sim.scenes[0]
                if WebsocketPolicy._trace_step % 10 == 0:
                    toks = [c.strip() for c in cats.split(",") if c.strip()]
                    objs = {}
                    for o in sc.objects:
                        cat = getattr(o, "category", "")
                        if cat and any(t in cat or cat in t for t in toks):
                            p = o.get_position_orientation()[0]
                            objs[o.name] = [round(float(x), 4) for x in p.tolist()]
                    rec["objs"] = objs
                rb = sc.robots[0] if sc.robots else None
                if rb is not None:
                    ee = {}
                    for arm in getattr(rb, "arm_names", []):
                        try:
                            ee[str(arm)] = [round(float(x), 4) for x in rb.get_eef_position(arm).tolist()]
                        except Exception:
                            pass
                    if ee:
                        rec["ee"] = ee
        except Exception:
            pass
        WebsocketPolicy._trace_fh.write(_json.dumps(rec) + "\n")
        WebsocketPolicy._trace_step += 1

    def forward(self, obs: dict, *args, **kwargs) -> th.Tensor:
        if "need_new_action" in obs and not obs["need_new_action"] and self.last_action is not None:
            return self.last_action
        self.last_action = self.policy.act(obs).detach().cpu()
        try:
            self._trace(obs, self.last_action)
        except Exception:
            pass
        return self.last_action

    def reset(self) -> None:
        if self.policy is not None:
            self.policy.reset()
        self.last_action = None

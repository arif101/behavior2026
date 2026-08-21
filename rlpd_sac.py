"""RLPD learner core per CONTACT_SKILL_SPEC_v1.md §3/§4 — torch, framework-free, co-resident
with sim in the behavior env (skill is ~2M params; learner GPU cost is negligible).

Networks (§3): actor MLP 3x512 + LayerNorm + tanh head (squashed Gaussian);
critic ensemble of 5 MLPs 3x512 + LayerNorm, min-over-random-2 targets (REDQ-style).
Training (§4): SAC, symmetric sampling (50% online / 50% prior per batch), UTD 8 default,
gamma 0.98, entropy auto-tune to -|A|.

Action convention: policy outputs a in [-1,1]^12; the ENV WRAPPER owns physical scaling
(skill_env_wrapper.py) — the learner never sees physical units. Prior-buffer actions are
raw physical (converter meta "action_units"), so the loader normalizes them with the same
scale vector the wrapper uses (wrapper writes it to /root/skill_wrapper_scale.json).
"""

import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def mlp(inp, out, hidden=512, layers=3, layer_norm=True):
    seq, d = [], inp
    for _ in range(layers):
        seq += [nn.Linear(d, hidden)]
        if layer_norm:
            seq += [nn.LayerNorm(hidden)]
        seq += [nn.ReLU()]
        d = hidden
    seq += [nn.Linear(d, out)]
    return nn.Sequential(*seq)


class Actor(nn.Module):
    LOG_STD_MIN, LOG_STD_MAX = -10.0, 2.0

    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.trunk = mlp(obs_dim, 2 * act_dim)
        self.act_dim = act_dim

    def forward(self, obs):
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs, deterministic=False):
        mu, log_std = self(obs)
        if deterministic:
            return torch.tanh(mu), None
        dist = torch.distributions.Normal(mu, log_std.exp())
        x = dist.rsample()
        a = torch.tanh(x)
        logp = (dist.log_prob(x) - torch.log1p(-a.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return a, logp


class CriticEnsemble(nn.Module):
    def __init__(self, obs_dim, act_dim, n=5):
        super().__init__()
        self.qs = nn.ModuleList([mlp(obs_dim + act_dim, 1) for _ in range(n)])

    def forward(self, obs, act, idx=None):
        x = torch.cat([obs, act], -1)
        qs = self.qs if idx is None else [self.qs[i] for i in idx]
        return torch.stack([q(x) for q in qs], 0)


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, capacity):
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.nobs = np.zeros((capacity, obs_dim), np.float32)
        self.act = np.zeros((capacity, act_dim), np.float32)
        self.rew = np.zeros(capacity, np.float32)
        self.done = np.zeros(capacity, np.float32)
        self.capacity, self.idx, self.full = capacity, 0, False

    def add(self, o, a, r, no, d):
        i = self.idx
        self.obs[i], self.act[i], self.rew[i], self.nobs[i], self.done[i] = o, a, r, no, d
        self.idx = (i + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def __len__(self):
        return self.capacity if self.full else self.idx

    def sample(self, n, rng):
        j = rng.integers(0, len(self), n)
        return self.obs[j], self.act[j], self.rew[j], self.nobs[j], self.done[j]

    @classmethod
    def from_prior_npz(cls, path, scale, hold_out_demos=()):
        """Load the converter's prior buffer, normalizing raw physical actions by `scale`
        into [-1,1] (clipped), excluding hold_out_demos (split BY DEMO per spec §5)."""
        z = np.load(path)
        keep = ~np.isin(z["demo"], np.asarray(list(hold_out_demos)))
        n = int(keep.sum())
        buf = cls(z["obs"].shape[1], z["action"].shape[1], n)
        a = np.clip(z["action"][keep] / scale[None, :], -1.0, 1.0).astype(np.float32)
        buf.obs[:n], buf.act[:n] = z["obs"][keep], a
        buf.rew[:n], buf.nobs[:n] = z["reward"][keep], z["next_obs"][keep]
        buf.done[:n] = z["done"][keep].astype(np.float32)
        buf.idx, buf.full = n % buf.capacity, n == buf.capacity
        return buf


class RLPD:
    def __init__(self, obs_dim, act_dim, gamma=0.98, tau=0.005, lr=3e-4,
                 n_critics=5, m_target=2, utd=8, seed=0):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.actor = Actor(obs_dim, act_dim).to(DEVICE)
        self.critic = CriticEnsemble(obs_dim, act_dim, n_critics).to(DEVICE)
        self.critic_tgt = CriticEnsemble(obs_dim, act_dim, n_critics).to(DEVICE)
        self.critic_tgt.load_state_dict(self.critic.state_dict())
        self.log_alpha = torch.tensor(np.log(0.1), device=DEVICE, requires_grad=True)
        self.target_entropy = -float(act_dim)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=lr)
        self.gamma, self.tau, self.utd = gamma, tau, utd
        self.n_critics, self.m_target = n_critics, m_target

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.actor.sample(
                torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)[None],
                deterministic)
        return a[0].cpu().numpy()

    def _batch(self, online, prior, batch_size, seed_buf=None):
        """Symmetric sampling: 50/50 online/prior (RLPD convention). When a seed buffer
        exists (planted demo successes for starved families), it takes a guaranteed 25%
        share carved from the two halves — otherwise ~130 seed tuples drown among 10^4
        online failures at ~2 per batch and TD propagation stalls (v21e observation)."""
        if prior is None:
            # prior retired (v21k: v1 press splices were collected under the spurious
            # left-weld reset — they teach the OLD radio pose and fight relearning).
            # Seeds (honest-world demo successes) inherit the guaranteed share.
            q = batch_size // 4 if (seed_buf is not None and len(seed_buf) > 0) else 0
            specs = [(online, batch_size - q), (seed_buf, q)]
        else:
            specs = [(online, batch_size // 2), (prior, batch_size - batch_size // 2)]
            if seed_buf is not None and len(seed_buf) > 0:
                q = batch_size // 4
                specs = [(online, batch_size // 2 - q // 2),
                         (prior, batch_size - batch_size // 2 - (q - q // 2)),
                         (seed_buf, q)]
        halves = []
        for buf, n in specs:
            if buf is not None and len(buf) > 0 and n > 0:
                halves.append(buf.sample(n, self.rng))
        parts = [np.concatenate(x) for x in zip(*halves)]
        return [torch.as_tensor(p, dtype=torch.float32, device=DEVICE) for p in parts]

    def update(self, online, prior, batch_size=256, seed_buf=None):
        """One UTD round: `utd` critic updates, one actor/alpha update. Returns stats."""
        stats = {}
        for _ in range(self.utd):
            o, a, r, no, d = self._batch(online, prior, batch_size, seed_buf=seed_buf)
            with torch.no_grad():
                na, nlogp = self.actor.sample(no)
                idx = self.rng.choice(self.n_critics, self.m_target, replace=False)
                tq = self.critic_tgt(no, na, idx=list(idx)).min(0).values
                alpha = self.log_alpha.exp()
                target = r[:, None] + self.gamma * (1 - d[:, None]) * (tq - alpha * nlogp)
            q = self.critic(o, a)
            critic_loss = F.mse_loss(q, target.expand_as(q))
            self.opt_c.zero_grad(set_to_none=True)
            critic_loss.backward()
            self.opt_c.step()
            with torch.no_grad():
                for p, tp in zip(self.critic.parameters(), self.critic_tgt.parameters()):
                    tp.lerp_(p, self.tau)
        pa, logp = self.actor.sample(o)
        qpi = self.critic(o, pa).mean(0)
        alpha = self.log_alpha.exp().detach()
        actor_loss = (alpha * logp - qpi).mean()
        self.opt_a.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.opt_a.step()
        alpha_loss = -(self.log_alpha.exp() * (logp.detach() + self.target_entropy)).mean()
        self.opt_alpha.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.opt_alpha.step()
        stats.update(critic_loss=float(critic_loss), actor_loss=float(actor_loss),
                     alpha=float(self.log_alpha.exp()), q_mean=float(q.mean()))
        return stats

    def save(self, path):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                    "critic_tgt": self.critic_tgt.state_dict(),
                    "log_alpha": self.log_alpha.detach().cpu()}, path)

    def load(self, path, reset_alpha=False):
        ck = torch.load(path, map_location=DEVICE, weights_only=False)
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])
        self.critic_tgt.load_state_dict(ck["critic_tgt"])
        with torch.no_grad():
            if reset_alpha:
                # fresh exploration temperature per chunk: a mastered scene tunes alpha
                # down, and resuming that collapsed alpha on a NEW scene freezes
                # exploration (2026-08-13: fresh-init cracked demo 20 in ~10 eps while
                # the resumed policy went 1/21 on demos 40/70)
                self.log_alpha.copy_(torch.tensor(np.log(0.1), device=DEVICE))
            else:
                self.log_alpha.copy_(ck["log_alpha"].to(DEVICE))


def load_meta(path="/root/skill_buffer_prior/skill_buffer_meta.json"):
    return json.load(open(path))

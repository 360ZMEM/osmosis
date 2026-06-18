"""@file eval/train_loop.py
@brief Minimal reference PPO loop (~50 LOC core), Isaac-independent.

Purpose:
  Provides a self-contained training reference so that anyone reading the
  EasyUUV-STDW repo can verify the obs/reward contract without launching
  Isaac Lab. NOT meant to replace `workflows/train_meta.py` for production.

Usage:
  python -m easyuuv_stdw.eval.train_loop --steps 2000

Required: torch, numpy, gymnasium (or any compatible env factory).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class PPOConfig:
    obs_dim: int = 10
    act_dim: int = 4
    hidden: int = 64
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    batch: int = 64
    rollout: int = 256


class ActorCritic(nn.Module):
    def __init__(self, cfg: PPOConfig):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(cfg.obs_dim, cfg.hidden), nn.Tanh(),
            nn.Linear(cfg.hidden, cfg.hidden), nn.Tanh(),
        )
        self.mu = nn.Linear(cfg.hidden, cfg.act_dim)
        self.log_std = nn.Parameter(torch.zeros(cfg.act_dim))
        self.v = nn.Linear(cfg.hidden, 1)

    def forward(self, obs):
        h = self.shared(obs)
        return self.mu(h), self.log_std.expand_as(self.mu(h)), self.v(h).squeeze(-1)


def gae(rewards, values, dones, gamma, lam):
    adv = np.zeros_like(rewards, dtype=np.float32)
    last = 0.0
    for t in reversed(range(len(rewards))):
        nonterm = 1.0 - float(dones[t])
        nextv = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * nextv * nonterm - values[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv


def train(env_factory: Callable, cfg: PPOConfig, total_steps: int = 2000):
    """@brief Reference PPO loop. Returns trained ActorCritic.
    @param env_factory  Callable -> gym-like env with reset()/step().
    @param cfg          PPOConfig.
    @param total_steps  Total environment steps to collect.
    """
    env = env_factory()
    net = ActorCritic(cfg)
    opt = optim.Adam(net.parameters(), lr=cfg.lr)

    obs, _ = env.reset()
    collected = 0
    while collected < total_steps:
        obs_buf, act_buf, lp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
        for _ in range(cfg.rollout):
            o = torch.as_tensor(obs, dtype=torch.float32)
            mu, logstd, v = net(o)
            dist = torch.distributions.Normal(mu, logstd.exp())
            a = dist.sample()
            lp = dist.log_prob(a).sum(-1)
            obs2, r, term, trunc, _ = env.step(a.detach().numpy())
            done = bool(term or trunc)
            obs_buf.append(o); act_buf.append(a.detach())
            lp_buf.append(lp.detach()); rew_buf.append(r)
            val_buf.append(v.detach().item()); done_buf.append(done)
            obs = obs2 if not done else env.reset()[0]
        collected += cfg.rollout

        adv = gae(np.asarray(rew_buf, dtype=np.float32),
                  np.asarray(val_buf + [0.0], dtype=np.float32),
                  np.asarray(done_buf, dtype=np.float32),
                  cfg.gamma, cfg.lam)
        ret = adv + np.asarray(val_buf, dtype=np.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_t = torch.stack(obs_buf)
        act_t = torch.stack(act_buf)
        lp_old = torch.stack(lp_buf)
        adv_t = torch.as_tensor(adv, dtype=torch.float32)
        ret_t = torch.as_tensor(ret, dtype=torch.float32)

        for _ in range(cfg.epochs):
            mu, logstd, v = net(obs_t)
            dist = torch.distributions.Normal(mu, logstd.exp())
            lp = dist.log_prob(act_t).sum(-1)
            ratio = (lp - lp_old).exp()
            s1 = ratio * adv_t
            s2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv_t
            policy_loss = -torch.min(s1, s2).mean()
            value_loss = (v - ret_t).pow(2).mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * dist.entropy().sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()

        print(f"[step {collected}] R={np.mean(rew_buf):.3f} V={value_loss.item():.3f}")
    return net


def _main():
    p = argparse.ArgumentParser(description="Reference PPO loop (Isaac-independent)")
    p.add_argument("--steps", type=int, default=2000)
    args = p.parse_args()
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise SystemExit("gymnasium required for --steps demo") from exc
    train(lambda: gym.make("Pendulum-v1"), PPOConfig(obs_dim=3, act_dim=1), args.steps)


if __name__ == "__main__":
    _main()

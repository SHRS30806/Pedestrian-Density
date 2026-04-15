"""
agents/ppo_agent.py
====================
PPO for 3-action traffic signal control (NS=0, EW=1, PED=2).
PED_CROSSING is action-masked when no pedestrians are waiting.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

logger = logging.getLogger(__name__)

PED_ACTION = 2


class ActorCriticNet(nn.Module):
    def __init__(self, obs_dim: int = 24, action_dim: int = 3) -> None:
        super().__init__()
        self.obs_dim    = obs_dim
        self.action_dim = action_dim

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.LayerNorm(128), nn.Tanh(),
            nn.Linear(128,      64), nn.LayerNorm(64),  nn.Tanh(),
        )
        self.ped_branch = nn.Sequential(
            nn.Linear(8, 32), nn.Tanh(),
            nn.Linear(32, 16), nn.Tanh(),
        )
        fused = 80

        self.actor = nn.Sequential(
            nn.Linear(fused, 32), nn.Tanh(),
            nn.Linear(32, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(fused, 32), nn.Tanh(),
            nn.Linear(32, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor[-1].weight,  gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, x, mask=None):
        fused  = torch.cat([self.backbone(x), self.ped_branch(x[:, 8:16])], dim=-1)
        logits = self.actor(fused)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        return logits, self.critic(fused).squeeze(-1)

    def get_action_value(self, x, mask=None, deterministic=False):
        logits, value = self.forward(x, mask)
        dist = Categorical(logits=logits)
        action = logits.argmax(-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, value, entropy


class RolloutBuffer:
    def __init__(self, capacity: int, obs_dim: int) -> None:
        self.capacity = capacity
        self.ptr = 0; self._full = False
        self.states    = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions   = np.zeros(capacity, dtype=np.int64)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.rewards   = np.zeros(capacity, dtype=np.float32)
        self.values    = np.zeros(capacity, dtype=np.float32)
        self.dones     = np.zeros(capacity, dtype=np.float32)
        self.advantages= np.zeros(capacity, dtype=np.float32)
        self.returns   = np.zeros(capacity, dtype=np.float32)

    def push(self, s, a, lp, r, v, done):
        i = self.ptr % self.capacity
        self.states[i]=s; self.actions[i]=a; self.log_probs[i]=lp
        self.rewards[i]=r; self.values[i]=v; self.dones[i]=float(done)
        self.ptr += 1
        if self.ptr >= self.capacity: self._full = True

    @property
    def full(self): return self._full

    def compute_gae(self, last_val, gamma=0.99, lam=0.95):
        gae = 0.0
        for t in reversed(range(self.capacity)):
            nv  = last_val if t == self.capacity-1 else self.values[t+1]
            nd  = 1.0 - self.dones[t]
            d   = self.rewards[t] + gamma*nv*nd - self.values[t]
            gae = d + gamma*lam*nd*gae
            self.advantages[t] = gae
            self.returns[t]    = gae + self.values[t]

    def batches(self, bs, device, normalize_adv=True):
        n   = self.capacity
        adv = self.advantages[:n]
        if normalize_adv:
            self.advantages[:n] = (adv - adv.mean()) / (adv.std() + 1e-8)
        idx = np.random.permutation(n)
        for s in range(0, n, bs):
            b = idx[s:s+bs]
            yield (
                torch.as_tensor(self.states[b],     device=device),
                torch.as_tensor(self.actions[b],    device=device),
                torch.as_tensor(self.log_probs[b],  device=device),
                torch.as_tensor(self.advantages[b], device=device),
                torch.as_tensor(self.returns[b],    device=device),
            )

    def reset(self): self.ptr = 0; self._full = False


class PPOAgent:
    def __init__(
        self,
        obs_dim=24, action_dim=3,
        lr=3e-4, gamma=0.99, gae_lam=0.95,
        clip_eps=0.2, n_epochs=6, batch_size=64,
        buffer_size=256, entropy_coef=0.05,
        value_coef=0.5, max_grad=0.5, device="auto",
    ):
        self.gamma=gamma; self.gae_lam=gae_lam; self.clip_eps=clip_eps
        self.n_epochs=n_epochs; self.batch_size=batch_size
        self.entropy_coef=entropy_coef; self.value_coef=value_coef
        self.max_grad=max_grad

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if device == "auto" else torch.device(device)

        self.net    = ActorCriticNet(obs_dim, action_dim).to(self.device)
        self.optim  = torch.optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)
        self.buffer = RolloutBuffer(buffer_size, obs_dim)

        self.n_updates = 0; self.total_steps = 0
        self.episode_rewards: list = []

        n = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        logger.info(f"PPOAgent | {self.device} | params={n:,}")

    def select_action(self, state, ped_waiting, deterministic=False):
        x    = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask = torch.ones(1, self.net.action_dim, dtype=torch.bool, device=self.device)
        if not np.any(ped_waiting):
            mask[0, PED_ACTION] = False

        with torch.no_grad():
            logits, value = self.net(x, mask)
            dist   = Categorical(logits=logits)
            action = logits.argmax(-1) if deterministic else dist.sample()
            lp     = dist.log_prob(action)

        self.total_steps += 1
        return int(action.item()), float(lp.item()), float(value.item())

    def store(self, s, a, lp, r, v, done):
        self.buffer.push(s, a, lp, r, v, done)

    def update(self, last_val=0.0):
        self.buffer.compute_gae(last_val, self.gamma, self.gae_lam)
        self.net.train()
        pl=vl=en=nb=0.0
        for _ in range(self.n_epochs):
            for states, actions, old_lp, adv, ret in \
                    self.buffer.batches(self.batch_size, self.device, normalize_adv=True):
                logits, vals = self.net(states)
                dist         = Categorical(logits=logits)
                new_lp       = dist.log_prob(actions)
                entropy      = dist.entropy().mean()
                ratio  = (new_lp - old_lp).exp()
                clip_r = ratio.clamp(1-self.clip_eps, 1+self.clip_eps)
                ploss  = -torch.min(ratio*adv, clip_r*adv).mean()
                vloss  = F.mse_loss(vals, ret)
                loss   = ploss + self.value_coef*vloss - self.entropy_coef*entropy
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad)
                self.optim.step()
                pl+=ploss.item(); vl+=vloss.item(); en+=entropy.item(); nb+=1
        self.buffer.reset(); self.net.eval(); self.n_updates += 1
        return {"policy": pl/max(nb,1), "value": vl/max(nb,1), "entropy": en/max(nb,1)}

    @property
    def buffer_ready(self): return self.buffer.full

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"net": self.net.state_dict(), "optim": self.optim.state_dict(),
                    "updates": self.n_updates, "steps": self.total_steps}, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(ck["net"]); self.optim.load_state_dict(ck["optim"])
        self.n_updates=ck.get("updates",0); self.total_steps=ck.get("steps",0)

    @property
    def n_parameters(self):
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    def set_train_mode(self):
        self.net.train()

    def set_eval_mode(self):
        self.net.eval()

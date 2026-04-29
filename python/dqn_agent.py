"""
agents/dqn_agent.py
--------------------
Double Dueling Deep Q-Network with Prioritized Experience Replay (PER).

Architecture: Dueling DQN (Wang et al., 2016)
    - Shared CNN/MLP backbone
    - Separate value stream V(s) and advantage stream A(s,a)
    - Q(s,a) = V(s) + A(s,a) - mean(A(s,·))

Learning improvements:
    - Double DQN: target network decouples action selection from evaluation
    - PER: prioritised replay by |TD-error| with importance sampling weights
    - Action masking: pedestrian-skip consistent with PPOAgent

Reference:
    Mnih et al. (2015)  — DQN
    Van Hasselt et al. (2016) — Double DQN
    Wang et al. (2016)  — Dueling DQN
    Schaul et al. (2016) — PER
"""

from __future__ import annotations

import logging
import random
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from base import BaseAgent

logger = logging.getLogger(__name__)


# ── Dueling DQN Network ───────────────────────────────────────────────────────

class DuelingQNetwork(nn.Module):
    """
    Dueling architecture with pedestrian feature branch.
    Identical feature extraction to PPO's backbone for fair comparison.
    """

    def __init__(
        self,
        state_dim:   int,
        action_dim:  int,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        ped_dims:    Tuple[int, ...] = (64, 32),
    ) -> None:
        super().__init__()
        self.state_dim  = state_dim
        self.action_dim = action_dim

        # Pedestrian branch (indices 8:16 of state)
        ped_layers: list[nn.Module] = []
        prev = 8
        for d in ped_dims:
            ped_layers += [nn.Linear(prev, d), nn.ReLU()]
            prev = d
        self.ped_branch = nn.Sequential(*ped_layers)
        ped_out = prev

        # Shared backbone
        bb_layers: list[nn.Module] = []
        prev = state_dim
        for d in hidden_dims:
            bb_layers += [nn.Linear(prev, d), nn.LayerNorm(d), nn.ReLU()]
            prev = d
        self.backbone = nn.Sequential(*bb_layers)
        fused = prev + ped_out

        # Value stream: s → scalar V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(fused, fused // 2), nn.ReLU(),
            nn.Linear(fused // 2, 1),
        )
        # Advantage stream: s → A(s, a) for each action
        self.adv_stream = nn.Sequential(
            nn.Linear(fused, fused // 2), nn.ReLU(),
            nn.Linear(fused // 2, action_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(
        self,
        state:  torch.Tensor,
        mask:   Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns Q-values shape (B, action_dim).
        Masked actions receive Q = -1e9.
        """
        ped_feat = self.ped_branch(state[:, 8:16])
        bb_feat  = self.backbone(state)
        fused    = torch.cat([bb_feat, ped_feat], dim=-1)

        value = self.value_stream(fused)                     # (B, 1)
        adv   = self.adv_stream(fused)                       # (B, A)
        q     = value + adv - adv.mean(dim=-1, keepdim=True) # dueling combination

        if mask is not None:
            q = q.masked_fill(~mask, -1e9)
        return q


# ── Prioritised Replay Buffer ─────────────────────────────────────────────────

@dataclass
class _Transition:
    state:      np.ndarray
    action:     int
    reward:     float
    next_state: np.ndarray
    done:       bool
    priority:   float = 1.0


class PrioritizedReplayBuffer:
    """
    Proportional PER with importance-sampling correction.
    Uses a sorted deque for simplicity (suitable for buffer_size ≤ 50k).
    For larger buffers, replace with a SumTree.
    """

    def __init__(
        self,
        capacity:  int,
        alpha:     float = 0.6,   # prioritisation exponent
        beta_init: float = 0.4,   # IS correction initial value
        beta_steps:int   = 100_000,
    ) -> None:
        self.capacity   = capacity
        self.alpha      = alpha
        self.beta       = beta_init
        self.beta_end   = 1.0
        self.beta_inc   = (self.beta_end - beta_init) / beta_steps
        self._buf:  Deque[_Transition] = deque(maxlen=capacity)
        self._max_priority = 1.0

    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        t = _Transition(
            state.copy(), action, reward, next_state.copy(), done,
            priority=self._max_priority,
        )
        self._buf.append(t)

    def sample(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, ...]:
        n = len(self._buf)
        probs = np.array([t.priority ** self.alpha for t in self._buf])
        probs /= probs.sum()

        indices = np.random.choice(n, size=batch_size, replace=False, p=probs)

        # Importance sampling weights
        weights = (n * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(self.beta + self.beta_inc, self.beta_end)

        batch = [self._buf[i] for i in indices]
        return (
            torch.FloatTensor(np.stack([t.state      for t in batch])).to(device),
            torch.LongTensor( np.array( [t.action    for t in batch])).to(device),
            torch.FloatTensor(np.array( [t.reward    for t in batch])).to(device),
            torch.FloatTensor(np.stack([t.next_state for t in batch])).to(device),
            torch.FloatTensor(np.array( [t.done      for t in batch])).to(device),
            torch.FloatTensor(weights).to(device),
            indices,
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        for idx, err in zip(indices, td_errors):
            p = float(abs(err)) + 1e-6
            self._buf[idx].priority = p
            self._max_priority = max(self._max_priority, p)

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def full(self) -> bool:
        return len(self._buf) >= self.capacity


# ── DQN Agent ─────────────────────────────────────────────────────────────────

@dataclass
class DQNConfig:
    state_dim:         int   = 21
    action_dim:        int   = 4
    hidden_dims:       Tuple[int, ...] = (256, 128, 64)
    ped_dims:          Tuple[int, ...] = (64, 32)
    lr:                float = 1e-4
    gamma:             float = 0.99
    buffer_size:       int   = 50_000
    batch_size:        int   = 64
    target_update_freq:int   = 500    # steps between target network hard update
    epsilon_start:     float = 1.0
    epsilon_end:       float = 0.05
    epsilon_decay:     int   = 10_000  # steps over which epsilon decays
    per_alpha:         float = 0.6
    per_beta_init:     float = 0.4
    device:            str   = "auto"
    min_replay_size:   int   = 1_000   # steps before learning begins


class DQNAgent(BaseAgent):
    """Double Dueling DQN with PER and pedestrian-skip masking."""

    PED_PHASE = 2

    def __init__(self, cfg: DQNConfig) -> None:
        self.cfg    = cfg
        self.device = self._resolve_device(cfg.device)

        self.online_net = DuelingQNetwork(
            cfg.state_dim, cfg.action_dim, cfg.hidden_dims, cfg.ped_dims
        ).to(self.device)
        self.target_net = DuelingQNetwork(
            cfg.state_dim, cfg.action_dim, cfg.hidden_dims, cfg.ped_dims
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(), lr=cfg.lr, eps=1.5e-4
        )
        self.replay = PrioritizedReplayBuffer(
            cfg.buffer_size, cfg.per_alpha, cfg.per_beta_init
        )

        self._step         = 0
        self._epsilon      = cfg.epsilon_start
        self.loss_history: Dict[str, list[float]] = {"td_loss": [], "mean_q": []}

        logger.info(
            f"DQNAgent on {self.device} | params={self.n_parameters:,} | "
            f"buffer={cfg.buffer_size}"
        )

    # ── BaseAgent ──────────────────────────────────────────────────────────────

    @property
    def name(self) -> str: return "DQN"

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.online_net.parameters() if p.requires_grad)

    def buffer_ready(self) -> bool:
        return len(self.replay) >= self.cfg.min_replay_size

    def set_train_mode(self) -> None: self.online_net.train()
    def set_eval_mode(self)  -> None: self.online_net.eval()

    def select_action(
        self,
        state: np.ndarray,
        ped_waiting: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        self._epsilon = max(
            self.cfg.epsilon_end,
            self.cfg.epsilon_start
            - (self.cfg.epsilon_start - self.cfg.epsilon_end)
            * self._step / self.cfg.epsilon_decay,
        )

        mask = self._build_mask(ped_waiting)

        if not deterministic and random.random() < self._epsilon:
            # Random valid action (respects mask)
            valid = [i for i, m in enumerate(mask) if m]
            action = random.choice(valid)
        else:
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask_t  = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q      = self.online_net(state_t, mask_t)
                action = q.argmax(-1).item()

        self._step += 1
        return int(action), 0.0, 0.0   # DQN doesn't use log_prob or value

    def store_transition(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
        log_prob:   float = 0.0,
        value:      float = 0.0,
    ) -> None:
        self.replay.push(state, action, reward, next_state, done)

    def update(self, last_value: float = 0.0) -> Dict[str, float]:
        if not self.buffer_ready():
            return {}

        states, actions, rewards, next_states, dones, weights, indices = \
            self.replay.sample(self.cfg.batch_size, self.device)

        # Double DQN target
        with torch.no_grad():
            # Online net selects best action for next state
            next_q_online = self.online_net(next_states)
            best_actions  = next_q_online.argmax(-1, keepdim=True)
            # Target net evaluates that action
            next_q_target = self.target_net(next_states).gather(1, best_actions).squeeze(-1)
            target_q      = rewards + self.cfg.gamma * next_q_target * (1 - dones)

        # Current Q estimates
        current_q = self.online_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        td_errors  = (target_q - current_q).detach().cpu().numpy()
        loss       = (weights * F.huber_loss(current_q, target_q, reduction="none")).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        self.replay.update_priorities(indices, np.abs(td_errors))

        # Hard target update
        if self._step % self.cfg.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
            logger.debug(f"Target network updated at step {self._step}")

        result = {
            "td_loss": loss.item(),
            "mean_q":  current_q.mean().item(),
            "epsilon": self._epsilon,
        }
        self.loss_history["td_loss"].append(result["td_loss"])
        self.loss_history["mean_q"].append(result["mean_q"])
        return result

    # ── Internals ─────────────────────────────────────────────────────────────

    def _build_mask(self, ped_waiting: np.ndarray) -> list[bool]:
        mask = [True] * self.cfg.action_dim
        if not np.any(ped_waiting):
            mask[self.PED_PHASE] = False
        return mask

    @staticmethod
    def _resolve_device(spec: str) -> torch.device:
        if spec != "auto":
            return torch.device(spec)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "online_net":   self.online_net.state_dict(),
            "target_net":   self.target_net.state_dict(),
            "optimizer":    self.optimizer.state_dict(),
            "step":         self._step,
            "epsilon":      self._epsilon,
            "loss_history": self.loss_history,
            "cfg":          self.cfg,
        }, path)
        logger.info(f"DQN checkpoint saved → {path}")

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._step    = ckpt.get("step", 0)
        self._epsilon = ckpt.get("epsilon", self.cfg.epsilon_end)
        self.loss_history = ckpt.get("loss_history", self.loss_history)
        logger.info(f"DQN checkpoint loaded ← {path} (step={self._step})")

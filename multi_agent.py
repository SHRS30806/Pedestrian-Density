"""
agents/multi_agent.py
----------------------
Independent Multi-Agent PPO (IPPO) coordinator for a network of intersections.

Each intersection has its own PPOAgent. Agents are trained independently
(no shared parameters) but share a common rollout collection loop and
synchronised update step — the standard IPPO baseline in MARL literature.

Upstream observation augmentation:
    Each agent's state is augmented with upstream queue pressures from
    adjacent intersections (optional, controlled by `use_neighbour_obs`).

Reference:
    de Witt et al. (2020) — "Is Independent Learning All You Need in the
    StarCraft Multi-Agent Challenge?"

    Coordination topology: each node connected in a 2×2 grid
    (easily extended to arbitrary graphs via adjacency list).
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from ppo_agent import PPOAgent
from config import PPOConfig
from intersection_env import TrafficIntersectionEnv

logger = logging.getLogger(__name__)


# ── Topology ──────────────────────────────────────────────────────────────────

@dataclass
class IntersectionNode:
    node_id:    str
    agent:      PPOAgent
    env:        TrafficIntersectionEnv
    neighbours: List[str] = field(default_factory=list)   # node_ids of adjacent intersections


# ── Coordinator ───────────────────────────────────────────────────────────────

class MultiAgentCoordinator:
    """
    Manages N independent PPO agents across an intersection network.

    Usage:
        coord = MultiAgentCoordinator.from_grid(rows=2, cols=2, cfg=ppo_cfg, env_cfg=env_cfg)
        coord.reset()
        for step in range(720):
            coord.step()
        coord.update_all()
        metrics = coord.collect_metrics()
    """

    def __init__(
        self,
        nodes: List[IntersectionNode],
        use_neighbour_obs: bool = False,
    ) -> None:
        self.nodes:             Dict[str, IntersectionNode] = {n.node_id: n for n in nodes}
        self.use_neighbour_obs: bool = use_neighbour_obs
        self._obs:              Dict[str, np.ndarray] = {}
        self._episode_rewards:  Dict[str, List[float]] = {n.node_id: [] for n in nodes}

        logger.info(
            f"MultiAgentCoordinator: {len(nodes)} agents | "
            f"neighbour_obs={use_neighbour_obs}"
        )

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_grid(
        cls,
        rows:    int,
        cols:    int,
        cfg:     PPOConfig,
        env_cfg,
        seed:    int = 42,
        use_neighbour_obs: bool = False,
    ) -> "MultiAgentCoordinator":
        """
        Create a rows×cols grid of intersections with Manhattan adjacency.
        Node IDs: "R{row}C{col}" (0-indexed).
        """
        from config import EnvConfig

        nodes: List[IntersectionNode] = []
        for r in range(rows):
            for c in range(cols):
                node_id = f"R{r}C{c}"
                neighbours: List[str] = []
                if r > 0:       neighbours.append(f"R{r-1}C{c}")
                if r < rows-1:  neighbours.append(f"R{r+1}C{c}")
                if c > 0:       neighbours.append(f"R{r}C{c-1}")
                if c < cols-1:  neighbours.append(f"R{r}C{c+1}")

                agent = PPOAgent(cfg)
                env   = TrafficIntersectionEnv(env_cfg, seed=seed + r * cols + c)
                nodes.append(IntersectionNode(node_id, agent, env, neighbours))

        return cls(nodes, use_neighbour_obs=use_neighbour_obs)

    # ── Episode Control ───────────────────────────────────────────────────────

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset all environments and return initial observations."""
        self._obs = {}
        for node_id, node in self.nodes.items():
            obs = node.env.reset()
            if self.use_neighbour_obs:
                obs = self._augment(node_id, obs)
            self._obs[node_id] = obs
        return dict(self._obs)

    def step(self) -> Dict[str, float]:
        """
        Single coordinated step across all agents.
        Returns per-agent rewards.
        """
        rewards: Dict[str, float] = {}

        for node_id, node in self.nodes.items():
            obs    = self._obs[node_id]
            action, log_prob, value = node.agent.select_action(
                obs, node.env.ped_waiting, deterministic=False
            )
            next_obs, reward, done, _ = node.env.step(action)

            if self.use_neighbour_obs:
                next_obs = self._augment(node_id, next_obs)

            node.agent.store_transition(
                obs, action, reward, next_obs, done, log_prob, value
            )
            self._obs[node_id] = next_obs
            rewards[node_id]   = reward
            self._episode_rewards[node_id].append(reward)

        return rewards

    def update_all(self) -> Dict[str, Dict[str, float]]:
        """
        Trigger PPO update for each agent whose buffer is full.
        Returns loss dicts keyed by node_id.
        """
        losses: Dict[str, Dict[str, float]] = {}
        for node_id, node in self.nodes.items():
            if node.agent.buffer_ready():
                obs = self._obs[node_id]
                _, _, last_val = node.agent.select_action(
                    obs, node.env.ped_waiting
                )
                losses[node_id] = node.agent.update(last_val)
        return losses

    # ── Neighbour Observation Augmentation ────────────────────────────────────

    def _augment(self, node_id: str, obs: np.ndarray) -> np.ndarray:
        """
        Append mean upstream queue pressure from neighbour intersections.
        Adds len(neighbours) extra features to the observation.
        Falls back to zeros for missing neighbours (boundary nodes).
        """
        node = self.nodes[node_id]
        pressures: List[float] = []
        for nb_id in node.neighbours:
            nb = self.nodes.get(nb_id)
            if nb is None:
                pressures.append(0.0)
            else:
                # Mean normalised queue of the neighbour (indices 0:4 of obs)
                nb_obs = self._obs.get(nb_id, np.zeros(nb.env.obs_dim))
                pressures.append(float(nb_obs[:4].mean()))
        return np.concatenate([obs, pressures], dtype=np.float32)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_all(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        for node_id, node in self.nodes.items():
            node.agent.save(d / f"{node_id}.pt")
        logger.info(f"All {len(self.nodes)} agents saved → {d}")

    def load_all(self, directory: str | Path) -> None:
        d = Path(directory)
        for node_id, node in self.nodes.items():
            ckpt = d / f"{node_id}.pt"
            if ckpt.exists():
                node.agent.load(ckpt)
            else:
                logger.warning(f"No checkpoint found for {node_id} at {ckpt}")

    # ── Metrics ───────────────────────────────────────────────────────────────

    def collect_metrics(self) -> Dict[str, Dict]:
        """Return per-agent episode metrics as nested dicts."""
        return {
            node_id: node.env.metrics.to_dict()
            for node_id, node in self.nodes.items()
        }

    def network_summary(self) -> Dict[str, float]:
        """Aggregate metrics across all agents into a single flat dict."""
        all_metrics = self.collect_metrics()
        keys = list(next(iter(all_metrics.values())).keys())
        summary: Dict[str, float] = {}
        for k in keys:
            vals = [m[k] for m in all_metrics.values()]
            summary[f"network_mean_{k}"] = float(np.mean(vals))
            summary[f"network_sum_{k}"]  = float(np.sum(vals))
        return summary

    def episode_returns(self) -> Dict[str, float]:
        return {
            node_id: float(np.sum(rewards))
            for node_id, rewards in self._episode_rewards.items()
        }

    def reset_episode_rewards(self) -> None:
        self._episode_rewards = {node_id: [] for node_id in self.nodes}

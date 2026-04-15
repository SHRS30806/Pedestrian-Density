"""
agents/base.py
--------------
Abstract base class defining the agent protocol.
Any RL algorithm (PPO, DQN, SAC) must implement this interface,
enabling drop-in replacement of the learning algorithm.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


class BaseAgent(abc.ABC):
    """
    Protocol for all traffic signal control agents.

    Concrete implementations: PPOAgent, DQNAgent, SACAgent.
    """

    # ── Core RL Interface ──────────────────────────────────────────────────────

    @abc.abstractmethod
    def select_action(
        self,
        state: np.ndarray,
        ped_waiting: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        """
        Args:
            state:       Normalised observation vector, shape (state_dim,)
            ped_waiting: Boolean mask of pedestrian presence per crosswalk
            deterministic: If True, returns argmax action (eval mode)

        Returns:
            action:   Chosen signal phase index
            log_prob: Log probability of chosen action (for PPO/SAC)
            value:    State value estimate (for actor-critic methods)
        """

    @abc.abstractmethod
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        """Buffer a single transition for later batch update."""

    @abc.abstractmethod
    def update(self, last_value: float = 0.0) -> Dict[str, float]:
        """
        Perform a learning update over buffered experience.

        Returns:
            Dictionary of loss scalars for logging
            e.g. {"policy_loss": 0.12, "value_loss": 0.34, "entropy": 0.56}
        """

    @abc.abstractmethod
    def buffer_ready(self) -> bool:
        """Returns True when enough experience has been collected to update."""

    # ── Persistence ───────────────────────────────────────────────────────────

    @abc.abstractmethod
    def save(self, path: str | Path) -> None:
        """Serialize agent state to disk."""

    @abc.abstractmethod
    def load(self, path: str | Path) -> None:
        """Restore agent state from disk."""

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Algorithm name string, e.g. 'PPO', 'DQN'."""

    @property
    def n_parameters(self) -> int:
        """Total trainable parameter count. Override in subclass if supported."""
        return -1

    def set_train_mode(self) -> None:
        """Switch network to training mode. Override if needed."""

    def set_eval_mode(self) -> None:
        """Switch network to eval mode. Override if needed."""

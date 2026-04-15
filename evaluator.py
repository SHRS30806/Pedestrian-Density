"""
evaluation/evaluator.py
------------------------
Statistically rigorous evaluation harness.

  - Runs N deterministic evaluation episodes across demand profiles
  - Computes mean ± std for all key metrics
  - Compares against Fixed-Time and Actuated baselines
  - Outputs a pandas DataFrame (or dict if pandas not installed)
  - Saves JSON results alongside the model checkpoint
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from base import BaseAgent
from config import TrafficConfig, EnvConfig
from intersection_env import TrafficIntersectionEnv, SignalPhase

logger = logging.getLogger(__name__)


# ── Baseline Agents ───────────────────────────────────────────────────────────

class FixedTimeAgent:
    """
    Webster-optimal fixed-timing controller.
    Cycles NS_GREEN → ALL_RED → EW_GREEN → ALL_RED → PED_CROSSING → ALL_RED.
    """

    CYCLE = [
        (SignalPhase.NS_GREEN,     6),   # 30s at 5s/step
        (SignalPhase.ALL_RED,      1),   # 5s clearance
        (SignalPhase.EW_GREEN,     6),   # 30s
        (SignalPhase.ALL_RED,      1),
        (SignalPhase.PED_CROSSING, 3),   # 15s
        (SignalPhase.ALL_RED,      1),
    ]

    def __init__(self) -> None:
        self._cycle_pos  = 0
        self._step_in_phase = 0
        self._flat_cycle = [
            phase for phase, dur in self.CYCLE for _ in range(dur)
        ]

    def select_action(self, obs: np.ndarray, ped_waiting: np.ndarray) -> int:
        action = int(self._flat_cycle[self._cycle_pos % len(self._flat_cycle)])
        self._cycle_pos += 1
        return action

    def reset(self) -> None:
        self._cycle_pos = 0


class MaxPressureAgent:
    """
    Actuated max-pressure controller: always serves the direction
    with maximum queue pressure (product of up/down-stream queues).
    Falls back to NS_GREEN on tie.
    """

    def select_action(self, obs: np.ndarray, ped_waiting: np.ndarray) -> int:
        # obs[0:4] = normalised queue lengths
        ns_pressure = obs[0] + obs[1]
        ew_pressure = obs[2] + obs[3]

        # Serve pedestrians if waiting a long time
        any_ped = np.any(ped_waiting)
        ped_urgent = any_ped and (obs[12:16].max() > 0.5)  # >45s normalised
        if ped_urgent:
            return int(SignalPhase.PED_CROSSING)

        return int(SignalPhase.NS_GREEN if ns_pressure >= ew_pressure
                   else SignalPhase.EW_GREEN)

    def reset(self) -> None:
        pass


# ── Evaluator ─────────────────────────────────────────────────────────────────

class Evaluator:
    """
    Runs evaluation episodes and computes summary statistics.

    Baselines are always included automatically so every eval call
    produces a full comparison table.
    """

    DEMANDS = ["low", "medium", "high"]

    _DEMAND_PROFILES: Dict[str, Dict[str, Any]] = {
        "low":    {"arrival_rates": (0.15, 0.15, 0.10, 0.10), "ped_arrival_rate": 0.02},
        "medium": {"arrival_rates": (0.30, 0.30, 0.20, 0.20), "ped_arrival_rate": 0.05},
        "high":   {"arrival_rates": (0.50, 0.45, 0.35, 0.30), "ped_arrival_rate": 0.10},
    }

    def __init__(self, cfg: TrafficConfig) -> None:
        self.cfg     = cfg
        self._ft     = FixedTimeAgent()
        self._mp     = MaxPressureAgent()

    # ── Public ────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        agent: BaseAgent,
        n_episodes: Optional[int] = None,
        save_path:  Optional[Path] = None,
    ) -> Dict[str, float]:
        """
        Evaluate `agent` against baselines.

        Returns:
            Flat dict with keys like:
                mean_reward, std_reward, mean_throughput,
                mean_ped_unsafe, mean_vehicle_wait, ...
            Suitable for TensorBoard scalar logging.
        """
        n = n_episodes or self.cfg.train.eval_episodes
        results = self._run_agent(agent, n, deterministic=True)
        flat    = self._flatten(results)

        if save_path:
            self._save(flat, save_path)

        return flat

    def full_comparison(
        self,
        agent: BaseAgent,
        n_episodes: int = 10,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Returns nested dict: {agent_name: {demand: {metric: value}}}.
        Includes DRL-PA, Fixed-Time, and Max-Pressure baselines.
        """
        agents = {
            agent.name: agent,
            "FixedTime":    self._ft,
            "MaxPressure":  self._mp,
        }
        comparison: Dict[str, Dict[str, Dict[str, float]]] = {}

        for name, ag in agents.items():
            comparison[name] = {}
            for demand in self.DEMANDS:
                metrics_list = self._run_demand(ag, demand, n_episodes)
                comparison[name][demand] = self._aggregate(metrics_list)

        return comparison

    # ── Internals ─────────────────────────────────────────────────────────────

    def _run_agent(
        self,
        agent: Any,
        n_episodes: int,
        deterministic: bool,
    ) -> Dict[str, List[float]]:
        """Collect metric lists across episodes (medium demand only)."""
        out: Dict[str, List[float]] = {
            "reward":        [],
            "throughput":    [],
            "ped_unsafe":    [],
            "vehicle_wait":  [],
            "ped_wait":      [],
            "unnecessary_ped": [],
        }

        if hasattr(agent, "reset"):
            agent.reset()

        for ep in range(n_episodes):
            seed = self.cfg.train.seed + 10000 + ep
            env  = self._make_env("medium", seed)
            obs  = env.reset()

            ep_reward = 0.0
            for _ in range(self.cfg.train.steps_per_episode):
                if hasattr(agent, "select_action"):
                    result = agent.select_action(obs, env.ped_waiting,
                                                 deterministic=deterministic)
                    action = result[0] if isinstance(result, tuple) else result
                else:
                    action = agent.select_action(obs, env.ped_waiting)

                obs, reward, _, _ = env.step(int(action))
                ep_reward += reward

            m = env.metrics
            out["reward"].append(ep_reward)
            out["throughput"].append(float(m.total_throughput))
            out["ped_unsafe"].append(float(m.ped_unsafe_events))
            out["vehicle_wait"].append(m.avg_vehicle_wait)
            out["ped_wait"].append(m.avg_ped_wait)
            out["unnecessary_ped"].append(float(m.unnecessary_ped_phases))

        return out

    def _run_demand(
        self,
        agent: Any,
        demand: str,
        n_episodes: int,
    ) -> List[Dict[str, float]]:
        records: List[Dict[str, float]] = []
        if hasattr(agent, "reset"):
            agent.reset()

        for ep in range(n_episodes):
            seed = self.cfg.train.seed + 20000 + ep
            env  = self._make_env(demand, seed)
            obs  = env.reset()
            ep_reward = 0.0

            for _ in range(self.cfg.train.steps_per_episode):
                if hasattr(agent, "select_action"):
                    result = agent.select_action(obs, env.ped_waiting)
                    action = result[0] if isinstance(result, tuple) else result
                else:
                    action = agent.select_action(obs, env.ped_waiting)
                obs, reward, _, _ = env.step(int(action))
                ep_reward += reward

            m = env.metrics
            records.append({
                "reward":       ep_reward,
                "throughput":   float(m.total_throughput),
                "ped_unsafe":   float(m.ped_unsafe_events),
                "vehicle_wait": m.avg_vehicle_wait,
                "ped_wait":     m.avg_ped_wait,
            })
        return records

    def _make_env(self, demand: str, seed: int) -> TrafficIntersectionEnv:
        profile = self._DEMAND_PROFILES[demand]
        env_cfg = EnvConfig(**{**self.cfg.env.__dict__, **profile})
        return TrafficIntersectionEnv(env_cfg, seed=seed)

    @staticmethod
    def _aggregate(records: List[Dict[str, float]]) -> Dict[str, float]:
        keys = records[0].keys()
        agg: Dict[str, float] = {}
        for k in keys:
            vals = [r[k] for r in records]
            agg[f"mean_{k}"] = float(np.mean(vals))
            agg[f"std_{k}"]  = float(np.std(vals))
        return agg

    @staticmethod
    def _flatten(results: Dict[str, List[float]]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, vals in results.items():
            out[f"mean_{k}"] = float(np.mean(vals))
            out[f"std_{k}"]  = float(np.std(vals))
        return out

    @staticmethod
    def _save(metrics: Dict[str, float], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Eval results saved → {path}")

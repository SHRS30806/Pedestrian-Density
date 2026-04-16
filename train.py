"""
train.py: Adaptive Traffic Signal Control via Proximal Policy Optimization
============================================================================

Implements the training pipeline for PPO-based traffic signal controller.
Supports curriculum learning, multi-demand evaluation, and checkpoint recovery.

Methodology:
  * Curriculum Learning: Progressive difficulty (low -> medium -> high demand)
  * Experience Replay: Rollout buffer with generalized advantage estimation (GAE)
  * Model Checkpointing: Save best-performing policies based on evaluation reward
  * Reproducibility: Full configuration saved with each experiment
  * Graceful Recovery: SIGINT handling to save state before termination

Usage:
    python train.py                              # Default configuration
    python train.py --cfg configs/default.yaml  # Custom YAML config
    python train.py --resume results/run_001/checkpoints/best.pt  # Resume training
    python train.py --quick                     # Quick test (5 episodes)

References:
  [1] Schulman et al. (2017). Proximal Policy Optimization Algorithms.
  [2] Mnih et al. (2016). Asynchronous Methods for Deep RL.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# TensorBoard (optional — falls back to no-op if not installed)
try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False

from ppo_agent import PPOAgent
from config import EnvConfig, TrafficConfig
from intersection_env import TrafficIntersectionEnv
from evaluator import Evaluator

# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(log_dir: Path, run_name: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s: %(message)s"
    handlers: List[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / f"{run_name}.log"),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


logger = logging.getLogger(__name__)


# ── Curriculum ────────────────────────────────────────────────────────────────

_DEMAND_PROFILES: Dict[str, Dict[str, float]] = {
    "low":    {"arrival_rates": (0.15, 0.15, 0.10, 0.10), "ped_rate": 0.02},
    "medium": {"arrival_rates": (0.30, 0.30, 0.20, 0.20), "ped_rate": 0.05},
    "high":   {"arrival_rates": (0.50, 0.45, 0.35, 0.30), "ped_rate": 0.10},
}


def env_for_demand(cfg: TrafficConfig, demand: str, seed: int) -> TrafficIntersectionEnv:
    profile = _DEMAND_PROFILES[demand]
    env_cfg = EnvConfig(
        **{
            **cfg.env.__dict__,
            **profile,
        }
    )
    return TrafficIntersectionEnv(env_cfg, seed=seed)


def curriculum_demand(episode: int, cfg: TrafficConfig) -> str:
    if not cfg.train.curriculum:
        return "medium"
    for threshold, demand in cfg.train.curriculum_stages:
        if episode <= threshold:
            return demand
    return cfg.train.curriculum_stages[-1][1]


# ── TensorBoard wrapper ───────────────────────────────────────────────────────

class Logger:
    """Thin wrapper: writes to TensorBoard if available, else no-ops."""

    def __init__(self, log_dir: Optional[Path]) -> None:
        self._writer = (
            SummaryWriter(log_dir=str(log_dir))
            if _TB_AVAILABLE and log_dir
            else None
        )

    def scalar(self, tag: str, value: float, step: int) -> None:
        if self._writer:
            self._writer.add_scalar(tag, value, step)

    def scalars(self, tag: str, values: Dict[str, float], step: int) -> None:
        if self._writer:
            self._writer.add_scalars(tag, values, step)

    def close(self) -> None:
        if self._writer:
            self._writer.close()


# ── Training Loop ─────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, cfg: TrafficConfig) -> None:
        self.cfg = cfg

        # Paths
        run_dir = Path(cfg.train.output_dir) / cfg.train.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg.train.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        setup_logging(Path(cfg.train.log_dir), cfg.train.run_name)
        cfg.to_yaml(run_dir / "config.yaml")

        # Components
        self.agent     = PPOAgent(cfg.ppo)
        self.tb        = Logger(cfg.train.tensorboard_dir)
        self.evaluator = Evaluator(cfg)

        self._best_eval_reward = -np.inf
        self._interrupted      = False

        # Graceful shutdown
        signal.signal(signal.SIGINT,  self._handle_sigint)
        signal.signal(signal.SIGTERM, self._handle_sigint)

    def _handle_sigint(self, *_) -> None:
        logger.warning("Interrupt received — saving checkpoint and exiting...")
        self._interrupted = True

    # ── Main Entry ────────────────────────────────────────────────────────────

    def train(self, resume_path: Optional[str] = None) -> None:
        if resume_path:
            self.agent.load(resume_path)
            logger.info(f"Resumed from {resume_path}")

        cfg      = self.cfg
        t_start  = time.perf_counter()
        ep_rewards: List[float] = []

        logger.info("=" * 70)
        logger.info(f"  IntelliSignal PPO Training - run: {cfg.train.run_name}")
        logger.info(f"  Episodes:   {cfg.train.total_episodes}")
        logger.info(f"  Steps/ep:   {cfg.train.steps_per_episode}")
        logger.info(f"  Device:     {self.agent.device}")
        logger.info(f"  Parameters: {self.agent.n_parameters:,}")
        logger.info("=" * 70)

        for episode in range(1, cfg.train.total_episodes + 1):
            if self._interrupted:
                break

            demand  = curriculum_demand(episode, cfg)
            env     = env_for_demand(cfg, demand, seed=cfg.train.seed + episode)
            ep_rew  = self._run_episode(env)

            ep_rewards.append(ep_rew)
            self.agent.episode_rewards.append(ep_rew)

            # ── Logging ──────────────────────────────────────────────────────
            if episode % cfg.train.log_interval == 0:
                recent = ep_rewards[-cfg.train.log_interval:]
                elapsed = time.perf_counter() - t_start
                m = env.metrics.to_dict()

                logger.info(
                    f"Ep {episode:5d}/{cfg.train.total_episodes}  "
                    f"demand={demand:<6}  "
                    f"reward={np.mean(recent):+8.2f}  "
                    f"throughput={m['throughput']:5.0f}  "
                    f"ped_unsafe={m['ped_unsafe']:4.0f}  "
                    f"elapsed={elapsed:6.0f}s"
                )

                self.tb.scalar("train/mean_reward",     np.mean(recent), episode)
                self.tb.scalar("train/throughput",      m["throughput"],  episode)
                self.tb.scalar("train/ped_unsafe",      m["ped_unsafe"], episode)
                self.tb.scalar("train/avg_vehicle_wait",m["avg_vehicle_wait_s"], episode)
                self.tb.scalar("train/entropy_coef",    self.agent.entropy_coef, episode)

                if self.agent.loss_history["policy"]:
                    self.tb.scalars("train/losses", {
                        k: v[-1] for k, v in self.agent.loss_history.items() if v
                    }, episode)

            # ── Evaluation ───────────────────────────────────────────────────
            if episode % cfg.train.eval_every_n == 0:
                eval_results = self.evaluator.evaluate(self.agent)
                mean_rew     = eval_results["mean_reward"]
                logger.info(
                    f"  [EVAL] ep={episode}  "
                    f"mean_reward={mean_rew:+.2f}  "
                    f"throughput={eval_results['mean_throughput']:.0f}  "
                    f"ped_unsafe={eval_results['mean_ped_unsafe']:.2f}"
                )
                self.tb.scalars("eval", eval_results, episode)

                if mean_rew > self._best_eval_reward:
                    self._best_eval_reward = mean_rew
                    best_path = cfg.train.checkpoint_dir / "best.pt"
                    self.agent.save(best_path)
                    logger.info(f"  * New best model saved (reward={mean_rew:.2f})")

            # ── Periodic checkpoint ───────────────────────────────────────────
            if episode % cfg.train.checkpoint_every_n == 0:
                ckpt = cfg.train.checkpoint_dir / f"ep_{episode:05d}.pt"
                self.agent.save(ckpt)

        # ── Wrap up ───────────────────────────────────────────────────────────
        final_path = cfg.train.checkpoint_dir / "final.pt"
        self.agent.save(final_path)
        self.tb.close()

        elapsed = time.perf_counter() - t_start
        logger.info(
            f"Training complete in {elapsed/60:.1f}min. "
            f"Best eval reward: {self._best_eval_reward:.2f}"
        )

    # ── Episode ───────────────────────────────────────────────────────────────

    def _run_episode(self, env: TrafficIntersectionEnv) -> float:
        cfg       = self.cfg
        obs       = env.reset()
        ep_reward = 0.0
        self.agent.set_eval_mode()

        for step in range(cfg.train.steps_per_episode):
            action, log_prob, value = self.agent.select_action(
                obs, env.ped_waiting, deterministic=False
            )
            next_obs, reward, done, _ = env.step(action)

            self.agent.store_transition(
                obs, action, reward, next_obs, done, log_prob, value
            )
            obs        = next_obs
            ep_reward += reward

            if self.agent.buffer_ready():
                _, _, last_value = self.agent.select_action(
                    obs, env.ped_waiting, deterministic=False
                )
                losses = self.agent.update(last_value)

        # Update at end of episode
        if self.agent.buffer.ptr > 0:  # If we have any transitions
            _, _, last_value = self.agent.select_action(
                obs, env.ped_waiting, deterministic=False
            )
            losses = self.agent.update(last_value)

        return ep_reward


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train IntelliSignal PPO Agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cfg",    type=str, default=None,
                        help="Path to YAML config (overrides defaults)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--run",    type=str, default=None,
                        help="Override run name")
    parser.add_argument("--seed",   type=int, default=None,
                        help="Override random seed")
    parser.add_argument("--quick",  action="store_true",
                        help="Quick test run with only 5 episodes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = TrafficConfig.from_yaml(args.cfg) if args.cfg else TrafficConfig()
    if args.run:
        cfg.train.run_name = args.run
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.quick:
        cfg.train.total_episodes = 5

    trainer = Trainer(cfg)
    trainer.train(resume_path=args.resume)


if __name__ == "__main__":
    main()

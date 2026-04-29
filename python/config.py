"""
configs/config.py
-----------------
Single source of truth for all hyperparameters.
Uses Python dataclasses + YAML override support.
Fully typed; validated at construction time.

Usage:
    cfg = TrafficConfig.from_yaml("configs/experiment_01.yaml")
    cfg = TrafficConfig()  # defaults
"""

from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Tuple


# ── Environment ────────────────────────────────────────────────────────────────

@dataclass
class EnvConfig:
    n_lanes:             int   = 4
    n_crosswalks:        int   = 4
    step_duration_s:     float = 5.0       # seconds per env step
    step_s:              float = 5.0       # alias for step_duration_s
    max_queue:           float = 30.0      # vehicles
    max_wait_s:          float = 120.0     # seconds
    max_ped_wait_s:      float = 90.0      # seconds

    # Vehicle arrival rates per lane (Poisson λ)
    arrival_rates:       Tuple[float, ...] = (0.3, 0.3, 0.2, 0.2)
    arrival_rate_ns:     float = 0.35      # veh/step/lane
    arrival_rate_ew:     float = 0.28      # veh/step/lane
    ped_rate:            float = 0.06

    # Phase min/max durations (seconds)
    phase_min_dur:       Tuple[int, ...] = (10, 10, 15, 3)
    phase_max_dur:       Tuple[int, ...] = (60, 60, 30, 8)
    min_green_steps:      int = 3        # minimum steps before phase can change

    # Saturation flow
    saturation_flow_vph: float = 1800.0   # vehicles/hour/lane (HCM)
    sat_flow_vph:        float = 1500.0   # alias

    # Pedestrian force
    ped_force_s:         float = 45.0     # override to PED if ped waits this long

    # Reward weights
    w_throughput:        float = 1.0
    w_wait_penalty:      float = 0.05
    w_ped_unsafe:        float = 10.0
    w_ped_served:        float = 3.0
    w_unnecessary_ped:   float = 2.0
    w_clear:             float = 3.0
    w_neglect:           float = 2.0
    w_ped_ok:            float = 5.0
    w_ped_bad:           float = 10.0
    w_switch:            float = 0.3

    # Observation normalisation
    obs_state_dim:       int   = 24        # final state vector length

    def __post_init__(self):
        assert len(self.arrival_rates) == self.n_lanes, \
            "arrival_rates length must equal n_lanes"
        assert len(self.phase_min_dur) == len(self.phase_max_dur) == 4, \
            "phase durations must have 4 entries"
        assert self.saturation_flow_vph > 0


# ── PPO Agent ──────────────────────────────────────────────────────────────────

@dataclass
class PPOConfig:
    # Architecture
    hidden_dims:         Tuple[int, ...] = (256, 128, 64)
    ped_branch_dims:     Tuple[int, ...] = (64, 32)
    action_dim:          int   = 3

    # PPO hyperparameters
    lr:                  float = 3e-4
    lr_schedule:         str   = "cosine"          # "cosine" | "linear" | "constant"
    gamma:               float = 0.99
    gae_lambda:          float = 0.95
    clip_epsilon:        float = 0.2
    clip_value_loss:     bool  = True
    entropy_coef:        float = 0.01
    entropy_decay:       float = 0.995             # multiplied each update
    value_coef:          float = 0.5
    max_grad_norm:       float = 0.5

    # Training
    n_epochs:            int   = 10
    batch_size:          int   = 64
    buffer_size:         int   = 2048
    target_kl:           Optional[float] = 0.015   # early stop epoch if KL > target
    normalize_advantages: bool = True

    # Device
    device:              str   = "auto"            # "auto" | "cpu" | "cuda" | "mps"

    def __post_init__(self):
        assert self.clip_epsilon > 0
        assert 0 < self.gamma <= 1
        assert 0 < self.gae_lambda <= 1
        assert self.lr > 0


# ── Training Loop ─────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    total_episodes:      int   = 1000
    steps_per_episode:   int   = 720           # 1 hour at 5s steps
    eval_every_n:        int   = 50
    eval_episodes:       int   = 5
    checkpoint_every_n:  int   = 100
    log_interval:        int   = 10
    seed:                int   = 42

    # Curriculum learning
    curriculum:          bool  = True
    curriculum_stages: Tuple[Tuple[int, str], ...] = (
        (200,  "low"),
        (600,  "medium"),
        (1000, "high"),
    )

    # Paths
    run_name:            str   = "run_001"
    output_dir:          str   = "results"
    log_dir:             str   = "results/logs"

    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.output_dir) / self.run_name / "checkpoints"

    @property
    def tensorboard_dir(self) -> Path:
        return Path(self.log_dir) / self.run_name


# ── Top-level Config ───────────────────────────────────────────────────────────

@dataclass
class TrafficConfig:
    env:   EnvConfig   = field(default_factory=EnvConfig)
    ppo:   PPOConfig   = field(default_factory=PPOConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrafficConfig":
        """Load config from YAML, overriding only keys present in the file."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        cfg = cls()
        for section, values in data.items():
            if section == "env" and isinstance(values, dict):
                cfg.env = EnvConfig(**{**asdict(cfg.env), **values})
            elif section == "ppo" and isinstance(values, dict):
                cfg.ppo = PPOConfig(**{**asdict(cfg.ppo), **values})
            elif section == "train" and isinstance(values, dict):
                cfg.train = TrainConfig(**{**asdict(cfg.train), **values})
        return cfg

    def to_yaml(self, path: str | Path) -> None:
        import json
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Convert tuples → lists so SafeLoader can round-trip
        data = json.loads(json.dumps(asdict(self)))
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    def __repr__(self) -> str:
        return (
            f"TrafficConfig(\n"
            f"  env={self.env},\n"
            f"  ppo={self.ppo},\n"
            f"  train={self.train}\n)"
        )

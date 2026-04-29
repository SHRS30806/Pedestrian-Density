"""Adaptive Traffic Intersection Environment for Signal Control Research

A Gym-compatible environment for studying deep reinforcement learning
approaches to traffic signal control. Simulates a 4-way signalized
intersection with vehicle and pedestrian traffic.

Environment Dynamics:
  - First-in-first-out (FIFO) vehicle queuing per lane
  - Pedestrian arrival modeling with safety constraints
  - 5-second signal phases with automatic clearance intervals
  - Realistic demand patterns (low/medium/high traffic)

Observation Space (24-dimensional, all in [0, 1]):
  [0:4]   Normalized queue lengths (vehicles per lane)
  [4:8]   Vehicle wait times (seconds, snapshot per lane)
  [8:12]  Pedestrian presence per crosswalk (binary indicators)
  [12:16] Pedestrian wait times (seconds)
  [16:19] Current phase (one-hot encoding: NS/EW/PED)
  [19]    Phase elapsed (fraction of phase duration)
  [20:22] Directional pressure metrics (NS and EW)
  [22]    Pedestrian urgency (boolean: any ped wait > 25s)
  [23]    Clearance state (boolean: in automatic clearance interval)

Action Space (3 discrete actions):
  0: NS_GREEN - Service North/South lanes
  1: EW_GREEN - Service East/West lanes
  2: PED_CROSSING - Pedestrian crossing phase (action-masked when empty)

Reward Function:
  R(s,a) = w_clearing * phase_clearance_bonus
           - w_neglect * directional_neglect_penalty
           + w_ped_served * pedestrians_cleared
           - w_ped_unsafe * unsafe_crossing_events
           - w_switch * phase_switch_penalty

References:
  [1] Guo et al. (2019). Adaptive Traffic Signal Control via Deep RL.
      IEEE Transactions on Systems, Man, and Cybernetics.
  [2] Brockman et al. (2016). OpenAI Gym. arXiv:1606.01540
  [3] Webster, F. V. (1958). Traffic signal settings. Road Research
      Technical Paper No. 39.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class SignalPhase(IntEnum):
    NS_GREEN     = 0
    EW_GREEN     = 1
    PED_CROSSING = 2
    ALL_RED      = 3   # internal only

    @property
    def green_lanes(self) -> Tuple[int, ...]:
        return {
            SignalPhase.NS_GREEN:     (0, 1),
            SignalPhase.EW_GREEN:     (2, 3),
            SignalPhase.PED_CROSSING: (),
            SignalPhase.ALL_RED:      (),
        }[self]

    @property
    def stopped_lanes(self) -> Tuple[int, ...]:
        """Vehicle lanes stopped during this phase."""
        return tuple(i for i in range(4) if i not in self.green_lanes)

    @property
    def is_pedestrian(self) -> bool:
        return self == SignalPhase.PED_CROSSING


@dataclass
class EnvConfig:
    n_lanes:      int   = 4
    n_crosswalks: int   = 4
    step_s:       float = 5.0

    # Symmetric-ish demand so both directions matter equally
    # NS slightly higher to give the agent an asymmetric problem to solve
    arrival_rate_ns: float = 0.35   # veh/step/lane  ≈ 252 vph
    arrival_rate_ew: float = 0.28   # veh/step/lane  ≈ 202 vph
    ped_rate:        float = 0.06

    sat_flow_vph: float = 1500.0    # 2.08 veh cleared per green step

    ped_force_s:  float = 45.0      # override to PED if ped waits this long

    max_queue:     float = 25.0
    max_wait_s:    float = 120.0
    max_ped_wait_s:float = 90.0

    min_green_steps: int = 3        # minimum steps before phase can change

    # Reward
    w_clear:   float = 3.0
    w_neglect: float = 2.0
    w_ped_ok:  float = 5.0
    w_ped_bad: float = 10.0
    w_switch:  float = 0.3


@dataclass
class LaneState:
    queue:  float = 0.0
    wait_s: float = 0.0

    @property
    def pressure(self) -> float:
        return self.queue * self.wait_s


@dataclass
class CrosswalkState:
    waiting: bool  = False
    wait_s:  float = 0.0


@dataclass
class EpisodeMetrics:
    total_steps:       int   = 0
    total_throughput:  int   = 0
    total_ped_served:  int   = 0
    ped_unsafe_events: int   = 0
    forced_ped_phases: int   = 0
    cumulative_reward: float = 0.0

    # Snapshot-based wait: record avg wait of ALL lanes each step
    _wait_snapshots: List[float] = field(default_factory=list)
    _ped_waits:      List[float] = field(default_factory=list)

    def snapshot_wait(self, lanes: List[LaneState]) -> None:
        """Record average wait of all vehicles currently queued."""
        waits = [l.wait_s for l in lanes if l.queue > 0]
        if waits:
            self._wait_snapshots.append(float(np.mean(waits)))

    def log_ped(self, wait_s: float) -> None:
        self._ped_waits.append(wait_s)

    @property
    def avg_vehicle_wait_s(self) -> float:
        """True average wait — includes stuck vehicles, not just served ones."""
        return float(np.mean(self._wait_snapshots)) if self._wait_snapshots else 0.0

    @property
    def avg_ped_wait_s(self) -> float:
        return float(np.mean(self._ped_waits)) if self._ped_waits else 0.0

    @property
    def throughput_per_hour(self) -> float:
        hours = (self.total_steps * 5.0) / 3600.0
        return self.total_throughput / hours if hours > 0 else 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "throughput":          float(self.total_throughput),
            "throughput_per_hour": self.throughput_per_hour,
            "ped_served":          float(self.total_ped_served),
            "ped_unsafe":          float(self.ped_unsafe_events),
            "forced_ped_phases":   float(self.forced_ped_phases),
            "avg_vehicle_wait_s":  self.avg_vehicle_wait_s,
            "avg_ped_wait_s":      self.avg_ped_wait_s,
            "cumulative_reward":   self.cumulative_reward,
        }


class TrafficIntersectionEnv:
    """
    4-way intersection where the agent learns to balance NS and EW traffic
    while managing pedestrian crossings safely.

    Key metric fix: avg_vehicle_wait_s is a snapshot average across ALL
    currently-waiting vehicles, not just those who happened to be served.
    This means an agent that starves NS lanes will see wait times grow
    correctly, providing a truthful signal for paper evaluation.
    """

    OBS_DIM    = 24
    ACTION_DIM = 3   # NS=0, EW=1, PED=2

    def __init__(self, cfg: Optional[EnvConfig] = None, seed: Optional[int] = None):
        self.cfg = cfg or EnvConfig()
        self.rng = np.random.default_rng(seed)
        self._init()

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._init()
        for i, lane in enumerate(self._lanes):
            lane.queue  = float(self.rng.uniform(1, 6))
            lane.wait_s = float(self.rng.uniform(5, 25))
        return self._observe()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        cfg = self.cfg

        # Auto clearance step
        if self._in_clearance:
            obs, r, done, info = self._run_phase(SignalPhase.ALL_RED, switched=False)
            self._clearance_left -= 1
            if self._clearance_left <= 0:
                self._in_clearance = False
                self._phase        = self._next_phase
            info["in_clearance"] = True
            return obs, r, done, info

        # Safety override
        forced = False
        if any(cw.waiting and cw.wait_s >= cfg.ped_force_s for cw in self._crosswalks):
            action = 2
            forced = True
            self._metrics.forced_ped_phases += 1

        requested = SignalPhase(action)

        # Minimum green enforcement
        if requested != self._phase and self._phase_steps < cfg.min_green_steps:
            requested = self._phase

        switched = (requested != self._phase)

        if switched:
            # NS↔EW conflict: insert one ALL_RED clearance step
            ns_ew = (
                {self._phase, requested} == {SignalPhase.NS_GREEN, SignalPhase.EW_GREEN}
            )
            if ns_ew:
                self._in_clearance  = True
                self._clearance_left= 1
                self._next_phase    = requested
                obs, r, done, info  = self._run_phase(SignalPhase.ALL_RED, switched=True)
                info["in_clearance"] = True
                info["forced_ped"]   = forced
                return obs, r, done, info

            self._phase       = requested
            self._phase_steps = 0
        else:
            self._phase_steps += 1

        obs, r, done, info = self._run_phase(self._phase, switched)
        info["in_clearance"] = False
        info["forced_ped"]   = forced
        return obs, r, done, info

    def _run_phase(self, phase: SignalPhase, switched: bool) -> Tuple[np.ndarray, float, bool, Dict]:
        cfg = self.cfg

        pressure_before = {i: self._lanes[i].pressure for i in range(4)}

        self._arrive()
        throughput     = self._depart(phase)
        peds, unsafe   = self._update_peds(phase)

        for lane in self._lanes:
            if lane.queue > 0:
                lane.wait_s = min(lane.wait_s + cfg.step_s, cfg.max_wait_s)
            else:
                lane.wait_s = 0.0

        # Snapshot wait AFTER update — includes all waiting vehicles
        self._metrics.snapshot_wait(self._lanes)

        reward = self._reward(phase, pressure_before, peds, unsafe, switched)

        m = self._metrics
        m.total_steps      += 1
        m.total_throughput += throughput
        m.total_ped_served += peds
        m.ped_unsafe_events+= unsafe
        m.cumulative_reward+= reward

        info = {
            "phase":       phase.name,
            "throughput":  throughput,
            "peds_served": peds,
            "ped_unsafe":  unsafe,
            "total_queue": sum(l.queue for l in self._lanes),
            "ns_pressure": self._lanes[0].pressure + self._lanes[1].pressure,
            "ew_pressure": self._lanes[2].pressure + self._lanes[3].pressure,
        }
        return self._observe(), reward, False, info

    def _arrive(self) -> None:
        cfg = self.cfg
        for i, lane in enumerate(self._lanes):
            rate = cfg.arrival_rate_ns if i < 2 else cfg.arrival_rate_ew
            lane.queue = min(lane.queue + int(self.rng.poisson(rate)), cfg.max_queue)
        for cw in self._crosswalks:
            if not cw.waiting and self.rng.random() < cfg.ped_rate:
                cw.waiting = True
                cw.wait_s  = 0.0

    def _depart(self, phase: SignalPhase) -> int:
        sat = (self.cfg.sat_flow_vph / 3600.0) * self.cfg.step_s
        n   = 0
        for i in phase.green_lanes:
            lane = self._lanes[i]
            if lane.queue <= 0:
                continue
            cleared = max(0, int(round(min(lane.queue, sat * self.rng.uniform(0.8, 1.2)))))
            if cleared:
                lane.queue  = max(0.0, lane.queue - cleared)
                if lane.queue == 0:
                    lane.wait_s = 0.0
                n += cleared
        return n

    def _update_peds(self, phase: SignalPhase) -> Tuple[int, int]:
        served = unsafe = 0
        for cw in self._crosswalks:
            if not cw.waiting:
                continue
            cw.wait_s += self.cfg.step_s
            if phase.is_pedestrian:
                self._metrics.log_ped(cw.wait_s)
                cw.waiting = False
                cw.wait_s  = 0.0
                served    += 1
            elif cw.wait_s > self.cfg.ped_force_s:
                unsafe += 1
        return served, unsafe

    def _reward(self, phase, p_before, peds, unsafe, switched) -> float:
        cfg   = self.cfg
        max_p = cfg.max_queue * cfg.max_wait_s * 2  # 2 lanes per direction

        # Pressure cleared this step by the green phase
        cleared_p = sum(
            max(0.0, p_before[i] - self._lanes[i].pressure)
            for i in phase.green_lanes
        )
        clear_term = cfg.w_clear * (cleared_p / (max_p + 1e-9))

        # Neglect penalty: pressure in the OTHER vehicle direction right now
        # NS_GREEN → penalise EW pressure; EW_GREEN → penalise NS pressure
        # PED/ALL_RED → penalise whichever direction has higher pressure
        ns_p = self._lanes[0].pressure + self._lanes[1].pressure
        ew_p = self._lanes[2].pressure + self._lanes[3].pressure

        if phase == SignalPhase.NS_GREEN:
            neglect_p = ew_p
        elif phase == SignalPhase.EW_GREEN:
            neglect_p = ns_p
        else:
            neglect_p = ns_p + ew_p   # both directions stopped

        neglect_term = -cfg.w_neglect * (neglect_p / (max_p + 1e-9))

        ped_term    = cfg.w_ped_ok * peds - cfg.w_ped_bad * unsafe
        switch_term = -cfg.w_switch if switched else 0.0

        return clear_term + neglect_term + ped_term + switch_term

    def _observe(self) -> np.ndarray:
        cfg = self.cfg
        obs = np.zeros(self.OBS_DIM, dtype=np.float32)

        for i, lane in enumerate(self._lanes):
            obs[i]      = min(lane.queue  / cfg.max_queue,  1.0)
            obs[4 + i]  = min(lane.wait_s / cfg.max_wait_s, 1.0)

        for i, cw in enumerate(self._crosswalks):
            obs[8 + i]  = float(cw.waiting)
            obs[12 + i] = min(cw.wait_s / cfg.max_ped_wait_s, 1.0)

        phase_idx = min(int(self._phase), 2)
        obs[16 + phase_idx] = 1.0

        obs[19] = min(self._phase_steps / (cfg.min_green_steps * 4 + 1e-9), 1.0)

        max_p   = cfg.max_queue * cfg.max_wait_s * 2
        obs[20] = min((self._lanes[0].pressure + self._lanes[1].pressure) / (max_p + 1e-9), 1.0)
        obs[21] = min((self._lanes[2].pressure + self._lanes[3].pressure) / (max_p + 1e-9), 1.0)

        obs[22] = float(any(cw.waiting and cw.wait_s > 25.0 for cw in self._crosswalks))
        obs[23] = float(self._in_clearance)

        return obs

    def _init(self) -> None:
        self._lanes      = [LaneState()     for _ in range(self.cfg.n_lanes)]
        self._crosswalks = [CrosswalkState() for _ in range(self.cfg.n_crosswalks)]
        self._phase             = SignalPhase.NS_GREEN
        self._phase_steps       = 0
        self._in_clearance      = False
        self._clearance_left    = 0
        self._next_phase        = SignalPhase.NS_GREEN
        self._metrics           = EpisodeMetrics()

    @property
    def ped_waiting(self) -> np.ndarray:
        return np.array([cw.waiting for cw in self._crosswalks], dtype=bool)

    @property
    def metrics(self) -> EpisodeMetrics:
        return self._metrics

    @property
    def obs_dim(self) -> int:
        return self.OBS_DIM

    @property
    def action_dim(self) -> int:
        return self.ACTION_DIM

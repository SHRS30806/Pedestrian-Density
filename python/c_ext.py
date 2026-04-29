"""
utils/c_ext.py
--------------
ctypes wrapper around queue_stats.so (compiled from c/queue_stats.c).

Provides:
  - Webster uniform delay formula
  - Moving-average queue history
  - Batch reward signal from C-side computation

Usage:
    from utils.c_ext import QueueStatsLib

    lib = QueueStatsLib.load()          # auto-finds .so/.dll
    delay = lib.webster_delay(90, 45, 0.4, 0.5)
    lib.update_lane(block, lane=0, green=True, arrivals=3.0)
    avg   = lib.moving_avg_queue(window=10)
"""

from __future__ import annotations

import ctypes
import logging
import os
import platform
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── C struct mirrors ───────────────────────────────────────────────────────────

MAX_LANES   = 4
MAX_HISTORY = 1000


class CLaneStatsBlock(ctypes.Structure):
    """
    Must match the C struct layout in queue_stats.c exactly.

    typedef struct {
        double queue_lengths[MAX_LANES];
        double wait_times[MAX_LANES];
        double arrival_rates[MAX_LANES];
        double departure_rates[MAX_LANES];
        int    n_lanes;
    } LaneStatsBlock;
    """
    _fields_ = [
        ("queue_lengths",   ctypes.c_double * MAX_LANES),
        ("wait_times",      ctypes.c_double * MAX_LANES),
        ("arrival_rates",   ctypes.c_double * MAX_LANES),
        ("departure_rates", ctypes.c_double * MAX_LANES),
        ("n_lanes",         ctypes.c_int),
    ]


class CIntersectionMetrics(ctypes.Structure):
    """
    typedef struct {
        double total_delay;
        double throughput;
        double avg_queue;
        double max_queue;
        double efficiency_ratio;
    } IntersectionMetrics;
    """
    _fields_ = [
        ("total_delay",      ctypes.c_double),
        ("throughput",       ctypes.c_double),
        ("avg_queue",        ctypes.c_double),
        ("max_queue",        ctypes.c_double),
        ("efficiency_ratio", ctypes.c_double),
    ]


# ── Loader ────────────────────────────────────────────────────────────────────

class QueueStatsLib:
    """
    Thin Python wrapper around the compiled C shared library.
    Handles type coercion; raises RuntimeError if library not found.
    """

    _LIB_NAME = {
        "Linux":  "queue_stats.so",
        "Darwin": "queue_stats.dylib",
        "Windows":"queue_stats.dll",
    }

    def __init__(self, lib: ctypes.CDLL) -> None:
        self._lib = lib
        self._configure_signatures()
        self._block = CLaneStatsBlock()
        self._lib.init_lane_stats(
            ctypes.byref(self._block), ctypes.c_int(MAX_LANES)
        )

    @classmethod
    def load(cls, search_dirs: Optional[list[str]] = None) -> "QueueStatsLib":
        """
        Locate and load the shared library.

        Compile with:
            gcc -O2 -shared -fPIC -o queue_stats.so queue_stats.c -lm   # Linux/Mac
            gcc -O2 -shared -o queue_stats.dll queue_stats.c -lm         # Windows
        """
        system    = platform.system()
        lib_name  = cls._LIB_NAME.get(system, "queue_stats.so")
        candidates = [
            Path(__file__).parent.parent.parent / "c" / lib_name,
            Path("c") / lib_name,
            Path(lib_name),
        ]
        if search_dirs:
            candidates += [Path(d) / lib_name for d in search_dirs]

        for candidate in candidates:
            if candidate.exists():
                try:
                    lib = ctypes.CDLL(str(candidate))
                    logger.info(f"Loaded C extension: {candidate}")
                    return cls(lib)
                except OSError as e:
                    logger.warning(f"Failed to load {candidate}: {e}")

        raise RuntimeError(
            f"Could not find {lib_name}. "
            "Compile with: gcc -O2 -shared -fPIC -o queue_stats.so c/queue_stats.c -lm"
        )

    def _configure_signatures(self) -> None:
        """Set explicit argtypes + restype for every exported symbol."""
        lib = self._lib
        db  = ctypes.c_double
        pi  = ctypes.POINTER(ctypes.c_double)
        pb  = ctypes.POINTER(CLaneStatsBlock)
        pm  = ctypes.POINTER(CIntersectionMetrics)

        lib.compute_webster_delay.argtypes  = [db, db, db, db]
        lib.compute_webster_delay.restype   = db

        lib.update_lane_queue.argtypes      = [pb, ctypes.c_int, ctypes.c_int, db]
        lib.update_lane_queue.restype       = db

        lib.compute_intersection_metrics.argtypes = [pb, pm]
        lib.compute_intersection_metrics.restype  = None

        lib.record_queue_snapshot.argtypes  = [pb]
        lib.record_queue_snapshot.restype   = None

        lib.compute_moving_avg_queue.argtypes = [ctypes.c_int, pi]
        lib.compute_moving_avg_queue.restype  = None

        lib.compute_queue_reward.argtypes   = [pb, db, db]
        lib.compute_queue_reward.restype    = db

        lib.init_lane_stats.argtypes        = [pb, ctypes.c_int]
        lib.init_lane_stats.restype         = None

    # ── Public API ────────────────────────────────────────────────────────────

    def webster_delay(
        self,
        cycle_s:       float,
        green_s:       float,
        demand_vps:    float,
        sat_flow_vps:  float,
    ) -> float:
        """Webster uniform + incremental delay (seconds)."""
        return float(self._lib.compute_webster_delay(
            cycle_s, green_s, demand_vps, sat_flow_vps
        ))

    def update_lane(
        self,
        lane_idx: int,
        is_green: bool,
        arrivals: float,
    ) -> float:
        """Update one lane's queue. Returns vehicles cleared this step."""
        return float(self._lib.update_lane_queue(
            ctypes.byref(self._block),
            ctypes.c_int(lane_idx),
            ctypes.c_int(int(is_green)),
            ctypes.c_double(arrivals),
        ))

    def intersection_metrics(self) -> dict[str, float]:
        """Compute throughput, delay, efficiency for current block state."""
        m = CIntersectionMetrics()
        self._lib.compute_intersection_metrics(
            ctypes.byref(self._block), ctypes.byref(m)
        )
        return {
            "total_delay":      m.total_delay,
            "throughput":       m.throughput,
            "avg_queue":        m.avg_queue,
            "max_queue":        m.max_queue,
            "efficiency_ratio": m.efficiency_ratio,
        }

    def record_snapshot(self) -> None:
        """Push current block state into the circular history buffer."""
        self._lib.record_queue_snapshot(ctypes.byref(self._block))

    def moving_avg_queue(self, window: int = 10) -> np.ndarray:
        """
        Returns (4,) float64 array of per-lane average queue over last
        `window` steps recorded via record_snapshot().
        """
        out = (ctypes.c_double * MAX_LANES)()
        self._lib.compute_moving_avg_queue(
            ctypes.c_int(window), ctypes.cast(out, ctypes.POINTER(ctypes.c_double))
        )
        return np.frombuffer(out, dtype=np.float64).copy()

    def queue_reward(
        self,
        throughput_weight: float = 1.0,
        delay_weight:      float = 0.05,
    ) -> float:
        """Compute reward delta from C-side queue state."""
        return float(self._lib.compute_queue_reward(
            ctypes.byref(self._block),
            ctypes.c_double(throughput_weight),
            ctypes.c_double(delay_weight),
        ))

    def sync_from_env(self, queues: np.ndarray, waits: np.ndarray) -> None:
        """
        Synchronise block state from Python env arrays.
        Call before any metric or reward computation.
        """
        for i in range(min(len(queues), MAX_LANES)):
            self._block.queue_lengths[i] = float(queues[i])
            self._block.wait_times[i]    = float(waits[i])

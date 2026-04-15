"""
scripts/run_hardware_loop.py
-----------------------------
End-to-end control loop connecting the trained PPO agent to the
Java TrafficControlServer via HTTP.

Flow:
    Java Server (sensor aggregation + safety enforcement)
        ↕  HTTP (GET /state, POST /action)
    Python PPO Agent (inference only, no training)
        ↕  Optional: live metrics to TensorBoard

This script is the deployment entry point for real-world or
SUMO-hardware-in-the-loop evaluation.

Usage:
    # Terminal 1: start Java server
    make java-run

    # Terminal 2: run the control loop
    python scripts/run_hardware_loop.py \
        --checkpoint results/run_001/checkpoints/best.pt \
        --server http://localhost:8765 \
        --duration 3600

Requires: requests, torch, numpy
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False
    print("[warn] 'requests' not installed. Install with: pip install requests")

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── Java server client ────────────────────────────────────────────────────────

class SignalServerClient:
    """Thin HTTP client wrapping the Java TrafficControlServer REST API."""

    def __init__(self, base_url: str, timeout: float = 2.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout
        self._session = requests.Session() if _HAS_REQUESTS else None

    def health_check(self) -> bool:
        try:
            r = self._session.get(f"{self.base_url}/health", timeout=self.timeout)
            return r.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def get_state(self) -> Optional[dict]:
        try:
            r = self._session.get(f"{self.base_url}/state", timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"GET /state failed: {e}")
            return None

    def post_action(self, action: int) -> Optional[dict]:
        try:
            r = self._session.post(
                f"{self.base_url}/action",
                json={"action": action},
                timeout=self.timeout,
            )
            return r.json()
        except Exception as e:
            logger.error(f"POST /action failed: {e}")
            return None

    def get_metrics(self) -> Optional[dict]:
        try:
            r = self._session.get(f"{self.base_url}/metrics", timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"GET /metrics failed: {e}")
            return None


# ── Control loop ──────────────────────────────────────────────────────────────

class HardwareControlLoop:
    """
    Runs the trained PPO agent against the live Java sensor/controller server.

    State vector is received from the Java server (already assembled from
    real sensor readings). Action is posted back, with the safety constraint
    layer enforced server-side.
    """

    def __init__(
        self,
        checkpoint_path: str,
        server_url:      str,
        step_interval_s: float = 5.0,
        log_interval:    int   = 12,   # log every 12 steps = 1 minute
    ) -> None:
        self.client         = SignalServerClient(server_url)
        self.step_interval  = step_interval_s
        self.log_interval   = log_interval
        self._running       = False

        # Load agent (inference only)
        self.agent = self._load_agent(checkpoint_path)

        # Metrics
        self._step           = 0
        self._total_reward   = 0.0
        self._action_history: list[int] = []
        self._phase_counts   = {i: 0 for i in range(4)}

    @staticmethod
    def _load_agent(path: str):
        """Load PPO agent for inference (no gradient, eval mode)."""
        sys.path.insert(0, str(Path(__file__).parent))
        from ppo_agent import PPOAgent
        from config import PPOConfig

        ckpt = __import__("torch").load(path, map_location="cpu", weights_only=False)
        cfg  = ckpt.get("cfg", PPOConfig())
        agent = PPOAgent(cfg)
        agent.load(path)
        agent.set_eval_mode()
        logger.info(f"Loaded PPO agent from {path} (params={agent.n_parameters:,})")
        return agent

    def run(self, duration_s: Optional[float] = None) -> None:
        if not _HAS_REQUESTS:
            raise RuntimeError("Install 'requests': pip install requests")

        # Connectivity check
        if not self.client.health_check():
            raise ConnectionError(
                f"Java server not reachable at {self.client.base_url}. "
                "Start with: make java-run"
            )
        logger.info(f"Connected to Java server at {self.client.base_url}")
        logger.info(f"Control loop starting | step={self.step_interval}s")

        self._running = True
        signal.signal(signal.SIGINT,  lambda *_: self._shutdown())
        signal.signal(signal.SIGTERM, lambda *_: self._shutdown())

        t_start   = time.monotonic()
        t_deadline = t_start + duration_s if duration_s else float("inf")

        while self._running and time.monotonic() < t_deadline:
            t_step = time.monotonic()

            # 1. Fetch state from Java server
            server_state = self.client.get_state()
            if server_state is None:
                logger.warning("No state received — holding current phase")
                time.sleep(self.step_interval)
                continue

            state_vec   = np.array(server_state.get("state", [0.0] * 21), dtype=np.float32)
            ped_present = np.array(server_state.get("ped_waiting", False))
            ped_arr     = np.array([bool(ped_present)] * 4, dtype=bool)

            # 2. Agent selects action (deterministic for deployment)
            action, _, _ = self.agent.select_action(
                state_vec, ped_arr, deterministic=True
            )

            # 3. Post action to Java server
            response = self.client.post_action(action)
            if response:
                applied = response.get("applied", "unknown")
                status  = response.get("status",  "unknown")
            else:
                applied = status = "error"

            self._step += 1
            self._action_history.append(action)
            self._phase_counts[action] = self._phase_counts.get(action, 0) + 1

            # 4. Periodic logging
            if self._step % self.log_interval == 0:
                elapsed = time.monotonic() - t_start
                metrics = self.client.get_metrics() or {}
                logger.info(
                    f"Step {self._step:5d}  "
                    f"elapsed={elapsed:6.0f}s  "
                    f"action={action} ({applied})  "
                    f"status={status}  "
                    f"ped={ped_present}  "
                    f"queue_size={metrics.get('phase_elapsed_ms', 'N/A')}"
                )

            # 5. Sleep to maintain step cadence
            elapsed_step = time.monotonic() - t_step
            sleep_time   = max(0, self.step_interval - elapsed_step)
            if sleep_time < 0.1:
                logger.warning(
                    f"Step took {elapsed_step:.2f}s > interval {self.step_interval}s"
                )
            time.sleep(sleep_time)

        self._print_summary()

    def _shutdown(self) -> None:
        logger.info("Shutdown signal received.")
        self._running = False

    def _print_summary(self) -> None:
        total_steps = max(self._step, 1)
        logger.info("=" * 60)
        logger.info(f"  Control loop completed: {self._step} steps")
        logger.info(f"  Duration: {self._step * self.step_interval:.0f}s")
        logger.info("  Phase distribution:")
        phase_names = ["NS_GREEN", "EW_GREEN", "PED_CROSSING", "ALL_RED"]
        for i, name in enumerate(phase_names):
            count = self._phase_counts.get(i, 0)
            pct   = 100.0 * count / total_steps
            logger.info(f"    {name:<16}: {count:5d} ({pct:.1f}%)")
        logger.info("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run trained PPO agent against Java signal server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to best.pt checkpoint"
    )
    parser.add_argument(
        "--server", default="http://localhost:8765",
        help="Java server base URL"
    )
    parser.add_argument(
        "--duration", type=float, default=None,
        help="Run duration in seconds (None = indefinite)"
    )
    parser.add_argument(
        "--step", type=float, default=5.0,
        help="Control step interval (seconds)"
    )
    args = parser.parse_args()

    loop = HardwareControlLoop(
        checkpoint_path=args.checkpoint,
        server_url=args.server,
        step_interval_s=args.step,
    )
    loop.run(duration_s=args.duration)


if __name__ == "__main__":
    main()

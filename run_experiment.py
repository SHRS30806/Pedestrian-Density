"""
run_experiment.py
==================
Full experiment runner — trains PPO and produces all paper results.

Usage:
    cd python
    python run_experiment.py                 # full run (500 eps, ~30 min CPU)
    python run_experiment.py --fast          # quick test (100 eps, ~5 min)
    python run_experiment.py --episodes 300  # custom length

On Google Colab / Kaggle GPU:
    !pip install torch numpy matplotlib
    !python run_experiment.py               # runs in ~8 min on T4 GPU

The script produces:
  - Full results table (paste into paper)
  - Key percentage improvements (for abstract)
  - Phase distribution analysis
  - Training curve + comparison figures (results/figures/)
  - JSON results file (results/paper_results.json)

NOTE ON CONVERGENCE
--------------------
The PPO agent needs approximately 300-500 training episodes to converge fully.
The --fast mode (100 episodes) shows correct behaviour but not peak performance.
For the paper, use the full run (500 episodes).

Expected results at 500 episodes (Medium demand):
  Fixed-Time:   wait ~93s, throughput ~909 v/h, ped_unsafe ~20
  Actuated:     wait ~60s, throughput ~960 v/h, ped_unsafe  ~0
  DRL-PA (Ours):wait ~45s, throughput ~940 v/h, ped_unsafe  ~0, ped_wait ~12s
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from intersection_env import (
    EnvConfig, SignalPhase, TrafficIntersectionEnv,
)
from ppo_agent import PPOAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Demand profiles  (3 levels used in paper)
# ─────────────────────────────────────────────────────────────────────────────

DEMANDS = {
    "Low":    dict(arrival_rate_ns=0.20, arrival_rate_ew=0.15, ped_rate=0.03),
    "Medium": dict(arrival_rate_ns=0.35, arrival_rate_ew=0.28, ped_rate=0.06),
    "High":   dict(arrival_rate_ns=0.55, arrival_rate_ew=0.42, ped_rate=0.10),
}

# Reward weights — validated: reward is higher for correct phase given pressure imbalance
REWARD_CFG = dict(w_clear=2.0, w_neglect=5.0, w_ped_ok=5.0, w_ped_bad=10.0, w_switch=0.1)


def make_env(demand: str, seed: int) -> TrafficIntersectionEnv:
    return TrafficIntersectionEnv(
        EnvConfig(**DEMANDS[demand], **REWARD_CFG), seed=seed
    )


def curriculum_demand(ep: int, total: int) -> str:
    f = ep / total
    if f < 0.25: return "Low"
    if f < 0.65: return "Medium"
    return "High"


# ─────────────────────────────────────────────────────────────────────────────
# Baseline controllers (3-action versions)
# ─────────────────────────────────────────────────────────────────────────────

class FixedTimeController:
    """
    Webster-optimal fixed cycle (3-action):
        NS_GREEN 30s (6 steps) → EW_GREEN 30s → PED_CROSSING 15s → repeat
    No response to real-time conditions.
    """
    CYCLE = [0]*6 + [1]*6 + [2]*3   # 15-step cycle

    def __init__(self): self._step = 0
    def act(self, obs, ped_waiting):
        a = self.CYCLE[self._step % len(self.CYCLE)]
        self._step += 1
        return a
    def reset(self): self._step = 0


class ActuatedController:
    """
    Pressure-based actuated signal (current industry standard).
    Serves the direction with highest queue pressure.
    Serves pedestrians when urgency flag is set.
    Minimum 15s green before switching.
    """
    MIN_STEPS = 3

    def __init__(self):
        self._phase = 0
        self._steps = 0

    def act(self, obs, ped_waiting):
        self._steps += 1
        if self._steps < self.MIN_STEPS:
            return self._phase
        # obs[22] = pedestrian urgency flag
        if obs[22] > 0.5 and np.any(ped_waiting):
            if self._phase != 2:
                self._phase = 2; self._steps = 0
            return 2
        # obs[20] = NS pressure, obs[21] = EW pressure
        chosen = 0 if obs[20] >= obs[21] else 1
        if chosen != self._phase:
            self._phase = chosen; self._steps = 0
        return self._phase

    def reset(self):
        self._phase = 0; self._steps = 0


class DRLVehicleOnly:
    """
    Same trained PPO agent but PED_CROSSING always masked.
    Used to isolate the contribution of pedestrian awareness.
    """
    def __init__(self, agent: PPOAgent):
        self._agent = agent

    def act(self, obs, ped_waiting):
        no_peds = np.zeros(4, dtype=bool)
        action, _, _ = self._agent.select_action(obs, no_peds, deterministic=True)
        return action

    def reset(self): pass


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(
    n_episodes:   int,
    steps_per_ep: int,
    seed:         int = 42,
    log_every:    int = 50,
) -> PPOAgent:
    agent = PPOAgent(
        obs_dim=24, action_dim=3,
        lr=3e-4,
        buffer_size=256,    # small buffer = frequent updates = faster learning
        n_epochs=6,
        entropy_coef=0.05,  # high entropy = explores all 3 phases
        batch_size=64,
    )
    log.info(
        f"PPO agent | params={agent.n_parameters:,} | "
        f"buffer=256 | ~{n_episodes} updates over {n_episodes} episodes"
    )

    ep_rewards: List[float] = []
    t0 = time.perf_counter()

    for ep in range(1, n_episodes + 1):
        demand = curriculum_demand(ep, n_episodes)
        env    = make_env(demand, seed=seed + ep)
        obs    = env.reset()
        ep_r   = 0.0

        for _ in range(steps_per_ep):
            a, lp, v = agent.select_action(obs, env.ped_waiting)
            obs, r, _, _ = env.step(a)
            agent.store(obs, a, lp, r, v, False)
            ep_r += r
            if agent.buffer_ready():
                _, _, lv = agent.select_action(obs, env.ped_waiting)
                agent.update(lv)

        ep_rewards.append(ep_r)
        agent.episode_rewards.append(ep_r)

        if ep % log_every == 0:
            m       = env.metrics
            elapsed = time.perf_counter() - t0
            log.info(
                f"  Ep {ep:4d}/{n_episodes} | {demand:<6} | "
                f"reward={np.mean(ep_rewards[-log_every:]):+8.1f} | "
                f"throughput={m.total_throughput:5d} | "
                f"ped_unsafe={m.ped_unsafe_events:3d} | "
                f"elapsed={elapsed:5.0f}s"
            )

    log.info(
        f"Training done | {agent.n_updates} updates | "
        f"final reward={np.mean(ep_rewards[-max(1, n_episodes//10):]):.1f}"
    )
    return agent


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    controller,
    demand:  str,
    n_eps:   int = 10,
    n_steps: int = 720,
    seed:    int = 99999,
) -> Dict:
    if hasattr(controller, "reset"):
        controller.reset()

    all_m: List[Dict] = []
    phases = Counter()

    for ep in range(n_eps):
        env = make_env(demand, seed=seed + ep)
        obs = env.reset()

        for _ in range(n_steps):
            if hasattr(controller, "act"):
                a = controller.act(obs, env.ped_waiting)
            else:
                a, _, _ = controller.select_action(
                    obs, env.ped_waiting, deterministic=True
                )
            phases[SignalPhase(a).name] += 1
            obs, _, _, _ = env.step(a)

        all_m.append(env.metrics.to_dict())

    means = {k: float(np.mean([m[k] for m in all_m])) for k in all_m[0]}
    stds  = {f"{k}_std": float(np.std([m[k] for m in all_m])) for k in all_m[0]}

    total = sum(phases.values())
    for ph in SignalPhase:
        if ph.name != "ALL_RED":
            means[f"pct_{ph.name.lower()}"] = phases[ph.name] / total * 100 if total else 0

    return {**means, **stds}


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

def print_table(results: Dict) -> None:
    methods = list(results.keys())
    demands = ["Low", "Medium", "High"]

    METRICS = [
        ("avg_vehicle_wait_s",  "Avg Vehicle Wait (s)",   True),
        ("throughput_per_hour", "Throughput (veh/hour)",   False),
        ("ped_unsafe",          "Ped Unsafe Events",       True),
        ("avg_ped_wait_s",      "Avg Ped Wait (s)",        True),
    ]

    print()
    print("=" * 100)
    print("  TABLE II — Performance Comparison  (★ = best per column)")
    print("=" * 100)

    for key, label, lower_better in METRICS:
        print(f"\n  {label}  ({'↓' if lower_better else '↑'})")
        print(f"  {'Method':<22}" + "".join(f"  {d:<16}" for d in demands))
        print(f"  {'-'*22}" + "".join(f"  {'-'*16}" for _ in demands))

        best = {}
        for d in demands:
            vals = [results[m][d].get(key, 0) for m in methods]
            best[d] = min(vals) if lower_better else max(vals)

        for method in methods:
            row = f"  {method:<22}"
            for d in demands:
                val = results[method][d].get(key, 0)
                std = results[method][d].get(f"{key}_std", 0)
                star = "★" if abs(val - best[d]) < 0.5 else " "
                row += f"  {val:6.1f}±{std:4.1f}{star}   "
            print(row)

    print("\n  ★ = best result for that demand level")
    print("=" * 100)


def print_abstract_numbers(results: Dict) -> None:
    if "DRL-PA (Ours)" not in results or "Fixed-Time" not in results:
        return

    print()
    print("─" * 65)
    print("  KEY IMPROVEMENTS vs FIXED-TIME (for your abstract)")
    print("─" * 65)

    COMPARE = [
        ("avg_vehicle_wait_s",  "Vehicle wait time",    True),
        ("throughput_per_hour", "Vehicle throughput",   False),
        ("ped_unsafe",          "Ped unsafe events",    True),
        ("avg_ped_wait_s",      "Pedestrian wait time", True),
    ]

    for key, label, lower in COMPARE:
        print(f"\n  {label}:")
        for demand in ["Low", "Medium", "High"]:
            ours = results["DRL-PA (Ours)"][demand].get(key, 0)
            base = results["Fixed-Time"][demand].get(key, 0)
            if base < 0.01:
                print(f"    {demand:<8}: baseline ≈ 0")
                continue
            pct = (base - ours) / base * 100 if lower else (ours - base) / base * 100
            arrow = "↓" if lower else "↑"
            print(f"    {demand:<8}: {arrow} {abs(pct):.1f}%  ({base:.1f} → {ours:.1f})")

    print("─" * 65)


def print_phase_distribution(results: Dict) -> None:
    methods = list(results.keys())
    print()
    print("  PHASE DISTRIBUTION — Medium Demand (% of steps)")
    print(f"  {'Method':<22} {'NS%':>7} {'EW%':>7} {'PED%':>7}")
    print(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*7}")
    for m in methods:
        d = results[m]["Medium"]
        ns  = d.get("pct_ns_green", 0)
        ew  = d.get("pct_ew_green", 0)
        ped = d.get("pct_ped_crossing", 0)
        print(f"  {m:<22} {ns:>7.1f} {ew:>7.1f} {ped:>7.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def save_figures(agent: PPOAgent, results: Dict, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed — skipping plots")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3,
    })

    COLORS = {
        "Fixed-Time":    "#F44336",
        "Actuated":      "#FF9800",
        "DRL (no ped)":  "#9C27B0",
        "DRL-PA (Ours)": "#1565C0",
    }

    # ── Training curve ────────────────────────────────────────────────────────
    if agent.episode_rewards:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        r = np.array(agent.episode_rewards, dtype=float)
        ax.plot(r, alpha=0.2, color="#1565C0", lw=0.8)
        ema, a = [r[0]], 0.05
        for v in r[1:]: ema.append(a*v + (1-a)*ema[-1])
        ax.plot(ema, color="#1565C0", lw=2, label="DRL-PA (PPO)")

        n = len(r)
        for frac, label in [(0.25, "Low→Med"), (0.65, "Med→High")]:
            ep = int(frac * n)
            ax.axvline(ep, color="gray", lw=1, ls="--", alpha=0.5)
            ax.text(ep+1, ax.get_ylim()[0], label, fontsize=7.5, color="gray")

        ax.set_xlabel("Training Episode")
        ax.set_ylabel("Cumulative Episode Reward")
        ax.set_title("Training Convergence — PPO with Curriculum Learning")
        ax.legend(loc="lower right")
        fig.tight_layout()
        for ext in ["pdf", "png"]:
            fig.savefig(out_dir / f"training_curve.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
        log.info("  ✓ training_curve.pdf/png")

    # ── Bar chart: all metrics, all demands ───────────────────────────────────
    methods = list(results.keys())
    demands = ["Low", "Medium", "High"]
    METRICS = [
        ("avg_vehicle_wait_s",  "Avg Vehicle\nWait (s)"),
        ("throughput_per_hour", "Throughput\n(veh/h)"),
        ("ped_unsafe",          "Ped Unsafe\nEvents"),
        ("avg_ped_wait_s",      "Avg Ped\nWait (s)"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))
    x = np.arange(len(demands))
    w = 0.18
    offsets = np.linspace(-(len(methods)-1)/2, (len(methods)-1)/2, len(methods)) * w

    for ax, (key, label) in zip(axes, METRICS):
        for i, (method, off) in enumerate(zip(methods, offsets)):
            vals = [results[method][d].get(key, 0) for d in demands]
            stds = [results[method][d].get(f"{key}_std", 0) for d in demands]
            col  = COLORS.get(method, "#607D8B")
            ec   = "#0D47A1" if "Ours" in method else "white"
            lw   = 2 if "Ours" in method else 0.5
            ax.bar(
                x+off, vals, w, color=col, alpha=0.85,
                edgecolor=ec, linewidth=lw,
                label=method if ax == axes[0] else None,
                yerr=stds, error_kw=dict(capsize=2.5, capthick=1, elinewidth=1),
            )
        ax.set_xticks(x)
        ax.set_xticklabels(demands, fontsize=9)
        ax.set_title(label, fontsize=9)
        ax.set_ylim(0)

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(methods),
               bbox_to_anchor=(0.5, 1.03), fontsize=9)
    fig.suptitle(
        "Performance Comparison Across Methods and Demand Levels",
        fontsize=11, y=1.08
    )
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(out_dir / f"comparison.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("  ✓ comparison.pdf/png")

    # ── Phase distribution (horizontal stacked bar) ───────────────────────────
    ph_keys   = ["pct_ns_green", "pct_ew_green", "pct_ped_crossing"]
    ph_labels = ["NS Green", "EW Green", "Ped Crossing"]
    ph_colors = ["#1565C0", "#7B1FA2", "#2E7D32"]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    y   = np.arange(len(methods))
    bot = np.zeros(len(methods))
    for pk, pl, pc in zip(ph_keys, ph_labels, ph_colors):
        vals = [results[m]["Medium"].get(pk, 0) for m in methods]
        ax.barh(y, vals, left=bot, color=pc, label=pl, alpha=0.85)
        bot += np.array(vals)
    ax.set_yticks(y)
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel("Phase Usage (%)")
    ax.set_title("Signal Phase Distribution — Medium Demand Scenario")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(out_dir / f"phase_dist.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("  ✓ phase_dist.pdf/png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast",     action="store_true",
                        help="100 training eps, 5 eval eps (quick test)")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--out",      type=str, default="results")
    args = parser.parse_args()

    N_TRAIN = 100 if args.fast else (args.episodes or 500)
    N_EVAL  = 5   if args.fast else 15
    STEPS   = 240 if args.fast else 720
    OUT     = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)

    log.info("=" * 65)
    log.info("  IntelliSignal — Paper Experiment")
    log.info(f"  Training:  {N_TRAIN} episodes × {STEPS} steps/ep")
    log.info(f"  Eval:      {N_EVAL} episodes × {STEPS} steps/ep per method")
    log.info(f"  Output:    {OUT.resolve()}")
    log.info("=" * 65)

    # 1. Train
    log.info("\n[1/3] Training DRL-PA (PPO + ped masking + curriculum)...")
    agent = train(N_TRAIN, STEPS, seed=args.seed, log_every=max(10, N_TRAIN // 10))
    agent.save(str(OUT / "drl_pa.pt"))
    log.info(f"  Saved: {OUT / 'drl_pa.pt'}")

    # 2. Evaluate
    log.info("\n[2/3] Evaluating all methods...")
    controllers = {
        "Fixed-Time":    FixedTimeController(),
        "Actuated":      ActuatedController(),
        "DRL (no ped)":  DRLVehicleOnly(agent),
        "DRL-PA (Ours)": agent,
    }

    results: Dict = {}
    for name, ctrl in controllers.items():
        results[name] = {}
        for demand in ["Low", "Medium", "High"]:
            log.info(f"    {name:<22} | {demand}...")
            results[name][demand] = evaluate(
                ctrl, demand,
                n_eps=N_EVAL, n_steps=STEPS,
                seed=args.seed + 50000,
            )

    # 3. Print results
    log.info("\n[3/3] Results:")
    print_table(results)
    print_abstract_numbers(results)
    print_phase_distribution(results)

    # Save JSON
    jpath = OUT / "paper_results.json"
    with open(jpath, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\n  Results saved: {jpath}")

    # Figures
    log.info("  Generating figures...")
    save_figures(agent, results, OUT / "figures")

    log.info(f"\n  Done. All files in: {OUT.resolve()}")
    log.info("  Copy the table above into your paper.")


if __name__ == "__main__":
    main()

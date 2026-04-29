#!/usr/bin/env python3
"""
Inference Demo: Using the Trained PPO Agent for Traffic Signal Control

This script demonstrates how to:
1. Load a trained PPO agent
2. Run inference on a traffic intersection
3. Visualize the decision-making process
4. Evaluate performance metrics

Usage:
    python inference_demo.py --checkpoint results/run_001/checkpoints/best.pt
"""

import sys
import argparse
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ppo_agent import PPOAgent
from intersection_env import TrafficIntersectionEnv
from config import EnvConfig


def load_agent(checkpoint_path: str) -> PPOAgent:
    """Load trained PPO agent from checkpoint."""
    agent = PPOAgent()
    agent.load(checkpoint_path)
    print(f"✓ Loaded agent from {checkpoint_path}")
    print(f"  - Network parameters: {agent.n_parameters:,}")
    print(f"  - Updates performed: {agent.n_updates}")
    return agent


def run_inference_episode(agent: PPOAgent, env: TrafficIntersectionEnv, max_steps: int = 100):
    """Run one episode of inference and collect detailed information."""
    obs = env.reset()
    episode_data = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'log_probs': [],
        'values': [],
        'phase_changes': [],
        'metrics': []
    }

    current_phase = env.phase
    total_reward = 0.0

    print("\n🚦 Starting inference episode...")
    print(f"Initial phase: {current_phase.name}")

    for step in range(max_steps):
        # Get agent decision
        action, log_prob, value = agent.select_action(
            obs, env.ped_waiting, deterministic=True  # Use deterministic for inference
        )

        # Store pre-action data
        episode_data['observations'].append(obs.copy())
        episode_data['log_probs'].append(log_prob)
        episode_data['values'].append(value)

        # Execute action
        next_obs, reward, done, info = env.step(action)

        # Store post-action data
        episode_data['actions'].append(action)
        episode_data['rewards'].append(reward)
        total_reward += reward

        # Track phase changes
        if env.phase != current_phase:
            episode_data['phase_changes'].append({
                'step': step,
                'from_phase': current_phase.name,
                'to_phase': env.phase.name,
                'reason': 'agent_decision' if action != current_phase.value else 'phase_timeout'
            })
            current_phase = env.phase

        # Store metrics snapshot
        metrics = env.metrics.to_dict()
        episode_data['metrics'].append(metrics)

        # Print progress every 10 steps
        if step % 10 == 0:
            phase_name = env.phase.name
            throughput = metrics['throughput']
            ped_unsafe = metrics['ped_unsafe']
            print(f"Step {step:3d}: Phase={phase_name:12s} | "
                  f"Reward={reward:+6.2f} | Throughput={throughput:5.1f} | "
                  f"Ped Unsafe={ped_unsafe:3.0f}")

        obs = next_obs

        if done:
            break

    print("\n🏁 Episode completed!")
    print(f"Total steps: {len(episode_data['actions'])}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final metrics: Throughput={metrics['throughput']:.1f} veh/h, "
          f"Ped Unsafe={metrics['ped_unsafe']:.1f}")

    return episode_data, total_reward


def analyze_decisions(episode_data: dict):
    """Analyze the agent's decision-making patterns."""
    actions = np.array(episode_data['actions'])
    rewards = np.array(episode_data['rewards'])
    values = np.array(episode_data['values'])

    print("\n📊 Decision Analysis:")
    print(f"  - Total actions taken: {len(actions)}")
    print(f"  - Action distribution:")
    unique, counts = np.unique(actions, return_counts=True)
    action_names = ['NS_GREEN', 'EW_GREEN', 'PED_CROSSING']
    for action_id, count in zip(unique, counts):
        percentage = (count / len(actions)) * 100
        print(f"    {action_names[action_id]:12s}: {count:3d} ({percentage:5.1f}%)")

    print(f"  - Average reward per action: {rewards.mean():.3f}")
    print(f"  - Average value estimate: {values.mean():.3f}")

    # Phase change analysis
    phase_changes = episode_data['phase_changes']
    print(f"  - Phase changes: {len(phase_changes)}")
    if phase_changes:
        print("    Recent changes:")
        for change in phase_changes[-3:]:  # Show last 3
            print(f"      Step {change['step']:3d}: {change['from_phase']} → {change['to_phase']}")


def main():
    parser = argparse.ArgumentParser(description="Traffic Signal Control Inference Demo")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--demand", type=str, default="medium",
                       choices=["low", "medium", "high"],
                       help="Traffic demand level")
    parser.add_argument("--steps", type=int, default=100,
                       help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")

    args = parser.parse_args()

    # Set up environment
    demand_profiles = {
        "low":    {"arrival_rates": (0.15, 0.15, 0.10, 0.10), "ped_rate": 0.02},
        "medium": {"arrival_rates": (0.30, 0.30, 0.20, 0.20), "ped_rate": 0.05},
        "high":   {"arrival_rates": (0.50, 0.45, 0.35, 0.30), "ped_rate": 0.10},
    }

    env_cfg = EnvConfig(**demand_profiles[args.demand])
    env = TrafficIntersectionEnv(env_cfg, seed=args.seed)

    print(f"🚗 Traffic Demand: {args.demand.upper()}")
    print(f"   - Vehicle arrival rates: {env_cfg.arrival_rates}")
    print(f"   - Pedestrian arrival rate: {env_cfg.ped_rate}")

    # Load trained agent
    agent = load_agent(args.checkpoint)

    # Run inference
    episode_data, total_reward = run_inference_episode(agent, env, args.steps)

    # Analyze results
    analyze_decisions(episode_data)

    print("\n✅ Inference demo completed!")
    print(f"💡 The trained PPO agent successfully controlled traffic signals,")
    print(f"   balancing vehicle throughput and pedestrian safety through learned policies.")


if __name__ == "__main__":
    main()
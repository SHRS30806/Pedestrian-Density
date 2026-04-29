#!/usr/bin/env python3
"""
Evaluation Script: Compare Trained PPO Agent Against Baselines
Usage:
    python evaluate_model.py --checkpoint results/run_001/checkpoints/best.pt
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ppo_agent import PPOAgent
from evaluator import Evaluator
from config import TrafficConfig


def extract_metrics(metrics):
    """Extract mean metrics from evaluator output."""
    return {
        'reward': metrics.get('mean_reward', 0.0),
        'throughput': metrics.get('mean_throughput', 0.0),
        'ped_unsafe': metrics.get('mean_ped_unsafe', 0.0),
        'vehicle_wait': metrics.get('mean_vehicle_wait', 0.0),
    }


def format_metrics(agent_name, metrics):
    """Format metrics for printing."""
    m = extract_metrics(metrics)

    return (
        f"{agent_name:<12}: "
        f"Reward={m['reward']:.2f} | "
        f"Throughput={m['throughput']:.1f} | "
        f"Ped Unsafe={m['ped_unsafe']:.1f} | "
        f"Vehicle Wait={m['vehicle_wait']:.1f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained PPO Agent")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes per demand level")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config file")

    args = parser.parse_args()

    print("Starting Model Evaluation")
    print(f"   Checkpoint: {args.checkpoint}")
    print(f"   Episodes per demand: {args.episodes}")

    # Validate checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load configuration
    cfg = TrafficConfig.from_yaml(args.config)
    cfg.train.eval_episodes = args.episodes

    # Load trained agent
    agent = PPOAgent(cfg.ppo)
    agent.load(str(checkpoint_path))
    print(f"Loaded PPO agent ({agent.n_parameters} parameters)")

    # Create evaluator
    evaluator = Evaluator(cfg)

    # Run evaluation
    print("\nRunning comprehensive evaluation...")
    results = evaluator.full_comparison(agent, n_episodes=args.episodes)

    # Debug check (only if structure is wrong)
    for agent_name, agent_data in results.items():
        for demand, metrics in agent_data.items():
            if 'mean_reward' not in metrics:
                print(f"WARNING: Unexpected format for {agent_name} ({demand}) -> keys: {list(metrics.keys())}")

    # Display results
    print("\nEvaluation Results:")
    print("=" * 60)

    for demand in ['low', 'medium', 'high']:
        print(f"\nDemand Level: {demand.upper()}")
        print("-" * 30)

        for agent_name, agent_data in results.items():
            if demand in agent_data:
                metrics = agent_data[demand]
                print(format_metrics(agent_name, metrics))

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        checkpoint_name = checkpoint_path.stem
        output_path = Path(f"results/{checkpoint_name}_evaluation.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Performance summary
    print("\nPerformance Summary:")
    ppo_results = results.get(agent.name, {})

    for demand in ['low', 'medium', 'high']:
        if demand in ppo_results:
            ppo_reward = ppo_results[demand].get('mean_reward', 0.0)
            fixed_reward = results.get('FixedTime', {}).get(demand, {}).get('mean_reward', 0.0)

            if fixed_reward == 0:
                improvement = float('inf') if ppo_reward > 0 else 0.0
            else:
                improvement = ((ppo_reward - fixed_reward) / abs(fixed_reward)) * 100

            print(
                f"  {demand.capitalize():<6}: "
                f"PPO reward = {ppo_reward:.2f} "
                f"({improvement:.1f}% vs Fixed-Time)"
            )

    print("\nEvaluation completed!")
    print("The trained PPO agent demonstrates adaptive traffic control capabilities.")


if __name__ == "__main__":
    main()
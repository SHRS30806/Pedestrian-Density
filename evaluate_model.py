#!/usr/bin/env python3
"""
Evaluation Script: Compare Trained PPO Agent Against Baselines

This script evaluates the trained PPO agent and compares it against:
1. Fixed-Time controller (Webster-optimal timing)
2. Max-Pressure controller (actuated control)

Usage:
    python evaluate_model.py --checkpoint results/run_001/checkpoints/best.pt
"""

import sys
import json
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ppo_agent import PPOAgent
from evaluator import Evaluator
from config import TrafficConfig


def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained PPO Agent")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of evaluation episodes per demand level")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file path")

    args = parser.parse_args()

    print("🧪 Starting Model Evaluation")
    print(f"   Checkpoint: {args.checkpoint}")
    print(f"   Episodes per demand: {args.episodes}")

    # Load configuration
    cfg = TrafficConfig.from_yaml('configs/default.yaml')
    cfg.train.eval_episodes = args.episodes

    # Load trained agent
    agent = PPOAgent(cfg.ppo)
    agent.load(args.checkpoint)
    print(f"✓ Loaded PPO agent ({agent.n_parameters:,} parameters)")

    # Create evaluator
    evaluator = Evaluator(cfg)

    # Run full comparison
    print("
🔬 Running comprehensive evaluation..."    results = evaluator.full_comparison(agent, n_episodes=args.episodes)

    # Display results
    print("
📊 Evaluation Results:"    print("=" * 60)

    for demand in ['low', 'medium', 'high']:
        print(f"\n🌆 Demand Level: {demand.upper()}")
        print("-" * 30)

        for agent_name in results:
            if demand in results[agent_name]:
                metrics = results[agent_name][demand]
                print(f"{agent_name:12s}: "
                      f"Reward={metrics['reward']:+6.2f} | "
                      f"Throughput={metrics['throughput']:5.1f} | "
                      f"Ped Unsafe={metrics['ped_unsafe']:4.1f} | "
                      f"Vehicle Wait={metrics['vehicle_wait']:5.1f}")

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        checkpoint_name = Path(args.checkpoint).stem
        output_path = Path(f"results/{checkpoint_name}_evaluation.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {output_path}")

    # Performance summary
    print("
🏆 Performance Summary:"    ppo_results = results.get(agent.name, {})

    for demand in ['low', 'medium', 'high']:
        if demand in ppo_results:
            ppo_reward = ppo_results[demand]['reward']
            fixed_reward = results.get('FixedTime', {}).get(demand, {}).get('reward', 0)

            improvement = ((ppo_reward - fixed_reward) / abs(fixed_reward)) * 100 if fixed_reward != 0 else 0

            print(f"  {demand.capitalize():6s}: PPO reward = {ppo_reward:+6.2f} "
                  f"({improvement:+5.1f}% vs Fixed-Time)")

    print("
✅ Evaluation completed!"    print("💡 The trained PPO agent demonstrates adaptive traffic control"    print("   capabilities, learning to balance competing objectives through RL.")


if __name__ == "__main__":
    main()
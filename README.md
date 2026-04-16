# Pedestrian-Aware Deep Reinforcement Learning for Adaptive Traffic Signal Control

**Authors:** Shreyas Rastogi et al.  
**Institution:** [Your Institution]  
**Status:** Research Implementation (arXiv pre-print)  
**License:** MIT

---

## Abstract

We present IntelliSignal, a deep reinforcement learning framework for adaptive traffic signal control that explicitly optimizes for pedestrian safety while maintaining vehicle throughput. Our approach uses Proximal Policy Optimization (PPO) with an Actor-Critic architecture extended to handle multi-modal objectives (vehicle efficiency, pedestrian safety, and signal coordination). The agent learns a curriculum-based policy that generalizes across varying traffic demand levels. Evaluation on a custom 4-way intersection simulator demonstrates **X% improvement in vehicle throughput** and **Y% reduction in unsafe pedestrian events** compared to fixed-time and demand-responsive baselines. Code and evaluation protocols are provided for reproducibility.

**Keywords:** traffic signal control, deep reinforcement learning, pedestrian safety, adaptive systems, urban computing

---

## 1. Introduction

Adaptive traffic signal control (ATSC) is a critical urban infrastructure problem affecting congestion, emissions, and safety. While Deep Reinforcement Learning has shown promise for traffic optimization, existing approaches often:

1. Neglect pedestrian safety as a primary objective
2. Lack explicit mechanisms for curriculum learning across demand levels
3. Provide limited reproducibility (closed-source implementations)

IntelliSignal addresses these limitations through:
- **Pedestrian-aware reward design** with explicit safety constraints
- **Curriculum learning** that progressively increases traffic demand
- **Full reproducibility** via typed Python configurations and multi-seed evaluation

This repository contains the complete implementation, evaluation suite, and experimental protocols used in our research.

---

## 2. Methods

### 2.1 Environment Model

We simulate a 4-way signalized intersection with:

**State Representation** (24-dimensional, normalized to [0,1]):
- Queue lengths and vehicle wait times (per lane)
- Pedestrian presence and wait times (per crosswalk)
- Current signal phase (3 actions: NS_GREEN, EW_GREEN, PED_CROSSING)
- Phase elapsed time and directional pressure metrics

**Reward Function:**
```
R(s,a) = α·ΔPressure(a) - β·Neglect(a) + γ·Ped_Served(a) - λ·Unsafe_Events(a)
```
Where:
- ΔPressure: vehicles cleared this phase
- Neglect: penalty for not serving high-pressure directions
- Ped_Served: bonus for crossing pedestrians
- Unsafe_Events: penalty for violations of pedestrian safety constraints

**Demand Profiles:**
We evaluate across three traffic demand levels:
- **Low demand:** λ_arrival ∈ [0.10, 0.15] vehicles/sec
- **Medium demand:** λ_arrival ∈ [0.20, 0.30] vehicles/sec  
- **High demand:** λ_arrival ∈ [0.30, 0.50] vehicles/sec

### 2.2 Learning Algorithm: Proximal Policy Optimization

**Architecture:**
- Actor-Critic with shared feature extraction (2 hidden layers, 128 units)
- Separate pedestrian-aware branch for safety-critical features
- Categorical action distribution with action masking

**Training:**
- Rollout buffer: 2048 transitions per update
- GAE with λ=0.95 for variance reduction
- Multi-epoch optimization (K_epochs=3)
- Entropy regularization: β ∈ [0.001, 0.005]

**Curriculum Learning:**
```
Episode 1-500:     low demand only
Episode 501-750:   medium demand only  
Episode 751-1000:  high demand only
```

### 2.3 Implementation Details

**Reproducibility:**
- All hyperparameters specified in `configs/default.yaml`
- Configuration objects serialized alongside results
- Multi-seed evaluation (seed ∈ {42, 123, 456})

**Evaluation Metrics:**
- **Throughput:** vehicles/hour
- **Safety:** unsafe crossing events per 1000 timesteps
- **Efficiency:** average vehicle wait time (seconds)
- **Pedestrian service:** fraction of pedestrians served within 30 seconds

---

## 3. Architecture & Code Organization

```
.
├── configs/
│   ├── config.py                    # Type-safe configuration (dataclasses)
│   └── default.yaml                 # YAML configuration override
├── ppo_agent.py                     # PPO implementation (Actor-Critic)
├── intersection_env.py              # Gym-compatible environment
├── evaluator.py                     # Baseline comparison & metrics
├── train.py                         # Training pipeline (curriculum, checkpointing)
├── test_ppo_agent.py                # Comprehensive test suite (24 tests)
└── results/                         # Checkpoints and logs
    └── run_001/
        ├── checkpoints/
        │   ├── best.pt              # Best model by eval reward
        │   └── final.pt             # Final model
        ├── logs/
        │   └── run_001.log          # Structured logging
        └── config.yaml              # Exact config used in this run
```

### Core Components

**1. Proximal Policy Optimization Agent** (`ppo_agent.py`)
- ActorCriticNet: Shared trunk + separate policy/value heads
- RolloutBuffer: Experience storage with GAE computation
- Gradient clipping and entropy regularization

**2. Traffic Intersection Environment** (`intersection_env.py`)
- FIFO lane queuing with realistic vehicle dynamics
- Pedestrian arrival modeling with safety constraints
- Action masking for disabled pedestrian crossing
- Metrics collection (throughput, safety, wait times)

**3. Training Engine** (`train.py`)
- Episode rollout with curriculum learning
- Periodic model checkpointing (best by eval reward)
- TensorBoard logging integration
- Graceful interrupt handling (SIGINT saves checkpoint)

**4. Evaluation Suite** (`evaluator.py`)
- Deterministic policy evaluation across demand profiles
- Baseline comparisons (fixed-time, demand-responsive)
- Statistical aggregation (mean, std, CI) across seeds

---

## 4. Usage

### Installation
```bash
pip install -r requirements.txt
pytest test_ppo_agent.py -v          # Run test suite (24 tests)
```

### Training

**Default (1000 episodes):**
```bash
python train.py
```

**Quick test (5 episodes):**
```bash
python train.py --quick
```

**Custom configuration:**
```bash
python train.py --cfg configs/default.yaml --run my_experiment --seed 42
```

**Resume from checkpoint:**
```bash
python train.py --resume results/run_001/checkpoints/best.pt
```

### Evaluation
```bash
python -c "
import sys; sys.path.insert(0, '.')
from train import TrafficConfig, Trainer
from evaluator import Evaluator

cfg = TrafficConfig.from_yaml('configs/default.yaml')
evaluator = Evaluator(cfg)
results = evaluator.evaluate(agent)
print(results)
"
```

---

## 5. Reproducing Paper Results

```bash
# Run 3 seeds to match paper protocol
for seed in 42 123 456; do
    python train.py --run paper_seed_$seed --seed $seed
done

# Aggregate results
python -c "
import json, pathlib
results = {}
for seed in [42, 123, 456]:
    with open(f'results/paper_seed_{seed}/eval_results.json') as f:
        results[f'seed_{seed}'] = json.load(f)
print(json.dumps(results, indent=2))
" > results/aggregated_results.json
```

---

## 6. Extending

### Adding a new RL algorithm
```python
from ppo_agent import PPOAgent

class MyAgent(PPOAgent):  # or subclass BaseAgent
    def update(self, last_val=0.0):
        # Your optimization logic here
        self.net.train()
        # ... compute losses ...
        self.loss_history["policy"].append(policy_loss)
        return {"policy": policy_loss, ...}
```

### Adding a new demand profile
```python
# In train.py
_DEMAND_PROFILES["custom"] = {
    "arrival_rates": (0.35, 0.30, 0.25, 0.20),
    "ped_rate": 0.08,
}
# Use: python train.py (curriculum automatically includes custom profile)
```

### Evaluating on custom environment
```python
from intersection_env import TrafficIntersectionEnv
from config import EnvConfig

env_cfg = EnvConfig(arrival_rates=(0.5, 0.5, 0.3, 0.3), ped_rate=0.1)
env = TrafficIntersectionEnv(env_cfg, seed=42)
obs = env.reset()
# env.step(), env.render(), etc.
```

---

## 7. Test Suite

Comprehensive pytest coverage (24 tests):
```bash
pytest test_ppo_agent.py -v

# Expected output:
# test_ppo_agent.py::TestPPOAgent::test_select_action PASSED
# test_ppo_agent.py::TestPPOAgent::test_update PASSED
# ... [24 tests total]
```

Tests cover:
- Agent initialization and save/load
- Buffer operations and GAE computation
- Environment dynamics and reward calculation
- Integration tests (full episode rollout)

---

## 8. References

[1] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).  
Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.

[2] Guo, X., Liu, Z., Wang, J., & Qiao, Y. (2019).  
Adaptive traffic signal control via deep reinforcement learning with discrete proximal policy optimization. *IEEE Transactions on Systems, Man, and Cybernetics: Systems*, 51(1), 19-30.

[3] Brockman, G., Cheung, V., Pettersson, L., Schneider, J., Schulman, J., Tang, J., & Zaremba, W. (2016).  
OpenAI gym. *arXiv preprint arXiv:1606.01540*.

[4] Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., ... & Kavukcuoglu, K. (2016).  
Asynchronous methods for deep reinforcement learning. In *International conference on machine learning* (pp. 1928-1937). PMLR.

[5] Webster, F. V. (1958).  
Traffic signal settings. *Road Research Technical Paper No. 39*, HMSO, London.

---

## 9. Citation

If you use IntelliSignal in your research, please cite:

```bibtex
@article{rastogi2025intelligsignal,
  title={Pedestrian-Aware Deep Reinforcement Learning for Adaptive Traffic Signal Control},
  author={Rastogi, Shreyas and Islam, Mamunur and Yadav, Yash and 
          Narayan, Lavish and Patel, Pranav and Rajput, Ayush},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

---

## License

MIT License - See LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue or contact [contact information].


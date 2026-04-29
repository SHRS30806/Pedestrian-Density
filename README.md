# Pedestrian-Aware Deep Reinforcement Learning for Adaptive Traffic Signal Control

**Authors:** Shreyas Rastogi et al.  
**Institution:** IILM University  

## Abstract

We present a deep reinforcement learning framework for adaptive traffic signal control that explicitly optimizes for pedestrian safety while maintaining vehicle throughput. Our approach utilizes Proximal Policy Optimization (PPO) with an Actor-Critic architecture extended to handle multi-modal objectives (vehicle efficiency, pedestrian safety, and signal coordination). The agent is trained using curriculum learning to generalize across varying traffic demand levels.

---

## 1. Introduction

Adaptive traffic signal control (ATSC) is a critical urban infrastructure problem affecting congestion, emissions, and safety. While Deep Reinforcement Learning has shown promise for traffic optimization, existing approaches often neglect pedestrian safety or lack explicit curriculum learning for scaling across demands.

This research addresses these limitations through:
- A pedestrian-aware reward design with explicit safety constraints.
- Curriculum learning that progressively increases traffic demand.
- A fully reproducible evaluation framework for comparing baseline methods.

---

## 2. Model Architecture

The framework relies on a multi-objective reinforcement learning pipeline modeled as a Markov Decision Process (MDP).

### State Representation
The intersection environment captures a 24-dimensional normalized state vector:
- Vehicle queue lengths and wait times across all approaching lanes.
- Pedestrian presence and waiting durations at crosswalks.
- The current active signal phase and elapsed phase time.

### Reward Function
The reward mechanism balances vehicle throughput and pedestrian safety:
- **Pressure cleared:** Rewards the clearing of queued vehicles.
- **Unsafe Events penalty:** Heavily penalizes signal changes that compromise crossing pedestrians.
- **Pedestrian Served bonus:** Provides a sparse reward for safely completing pedestrian crossings.

### Learning Method (PPO)
- The agent employs an **Actor-Critic** architecture with shared feature extraction layers and separate policy and value heads.
- A dedicated pedestrian-aware network branch processes safety-critical features independently to prevent feature dilution.
- **Curriculum Learning Pipeline:**
  - Episodes 1-500: Low traffic demand
  - Episodes 501-750: Medium traffic demand
  - Episodes 751-1000: High traffic demand

---

## 3. Training the Model

The training process leverages curriculum learning to progressively expose the agent to higher traffic densities.

### Standard Training
To train the PPO model from scratch:
```bash
cd python
python train.py --cfg configs/default.yaml
```

The training script outputs checkpoints and logs into the `results/` directory. Training progress, including the **training curve** (reward over time, policy loss, and value loss), is logged and can be plotted to visualize the convergence and stability of the model.

### Quick Verification
To verify the environment and training loop function correctly:
```bash
cd python
python train.py --quick
```

---

## 4. Results & Comparison

A core component of this research is demonstrating the benefits of the proposed PPO-based method over alternative approaches.

### Baseline Comparisons
The framework compares the trained PPO agent against traditional and alternative RL baselines:
- **Fixed-Time Control:** Traditional static phase rotation.
- **DQN Agent:** Deep Q-Network baseline to demonstrate the advantages of policy gradient methods in highly stochastic environments.

### Running the Evaluations
To evaluate a trained checkpoint and generate the quantitative results (throughput vs. safety metrics):
```bash
cd python
python run_experiment.py
```
This orchestration script runs the multi-seed evaluation, tests the trained policy against baselines across varying demand profiles, and aggregates the results for statistical comparison.

### Generating Training Curves and Plots
To visualize the training curve and comparative performance:
```bash
cd python
python plot_results.py
```
These plots illustrate the algorithm's convergence and explicitly show the trade-off improvements in vehicle throughput versus pedestrian safety constraints.

---

## 5. File Structure

The repository is structured to support the RL training pipeline and experimentation:

```
.
├── python/                     
│   ├── configs/                # Hyperparameter configurations (default.yaml)
│   ├── ppo_agent.py            # Core Actor-Critic PPO architecture
│   ├── dqn_agent.py            # DQN baseline implementation
│   ├── intersection_env.py     # Intersection environment and state representation
│   ├── train.py                # Model training and curriculum orchestration
│   ├── run_experiment.py       # Baseline comparison and statistical evaluation
│   ├── evaluator.py            # Metric calculation and validation
│   └── plot_results.py         # Generation of training curves and comparison charts
├── tests/python/               # Unit testing for RL components
├── c/                          # High-performance C extensions for queue mechanics
│   └── queue_stats.c           
└── results/                    # Generated output folder for runs (checkpoints, logs)
```

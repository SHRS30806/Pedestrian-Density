# Pedestrian-Aware Deep Reinforcement Learning for Adaptive Traffic Signal Control

**Authors:** Shreyas Rastogi et al.  
**Institution:** IILM University  

## Abstract

This project uses deep reinforcement learning (DRL) to control traffic lights. Most traffic AI models only focus on moving cars quickly, which can be dangerous for pedestrians. Our model uses Proximal Policy Optimization (PPO) to balance vehicle throughput with pedestrian safety. We also use curriculum learning to train the agent on different traffic volumes.

---

## 1. Introduction

Traffic signal control is an important part of urban infrastructure. Traditional systems run on fixed timers or simple sensors, which struggle during heavy traffic. While reinforcement learning works well for this, many researchers ignore crosswalks.

This project attempts to fix this by:
- Adding specific safety constraints for pedestrians in the reward function.
- Using curriculum learning to step up the traffic density during training.
- Comparing our PPO model against fixed-time and actuated signal baselines.

---

## 2. Model Architecture

The environment is built as a Markov Decision Process (MDP).

### State Representation
The intersection returns a 24-dimensional array:
- Number of cars queued and their waiting times.
- Whether pedestrians are waiting at the crosswalk.
- The current green light phase and how long it has been green.

### Reward Function
The reward math tries to maximize throughput while preventing unsafe crossings:
- **Pressure cleared:** Points given for clearing long vehicle queues.
- **Unsafe Events penalty:** Heavy point deductions if the light turns green for cars while pedestrians are crossing.
- **Pedestrian Served:** Points given for safely clearing a crosswalk.

### Learning Method (PPO)
- We use an **Actor-Critic** setup.
- The neural network has a separate branch just for pedestrian data so it doesn't get ignored by the larger vehicle numbers.
- **Curriculum Training:**
  - Episodes 1-500: Low traffic
  - Episodes 501-750: Medium traffic
  - Episodes 751-1000: High traffic

---

## 3. Training the Model

To train the PPO model from scratch, open your terminal and run:

```bash
cd python
python train.py --cfg configs/default.yaml
```

The script will save the model weights and training logs into the `results/` folder. 

To run a fast test just to check if your Python environment is set up correctly:
```bash
cd python
python train.py --quick
```

---

## 4. Results & Comparison

We test our PPO agent against two standard traffic systems:
- **Fixed-Time Control:** A simple repeating timer.
- **Actuated:** A sensor-based system that changes lights based on queue limits.

### Running the Evaluations
To run the evaluation script and print the performance metrics (throughput vs. wait times):
```bash
cd python
python run_experiment.py
```
This tests the model on 3 different traffic demands (Low, Medium, High).

### Generating Plots
To create the PNG and PDF charts for the paper:
```bash
cd python
python plot_results.py
```
This will output the convergence graphs and phase distribution charts into the `python/results/figures` folder.

---

## 5. File Structure

```
.
├── python/                     
│   ├── configs/                # Configuration parameters (default.yaml)
│   ├── ppo_agent.py            # The PPO math and neural network
│   ├── dqn_agent.py            # DQN baseline
│   ├── intersection_env.py     # The traffic simulation math
│   ├── train.py                # Main training loop
│   ├── run_experiment.py       # Evaluation and testing script
│   ├── evaluator.py            # Metric calculations
│   └── plot_results.py         # Graph generation script
├── tests/python/               # Unit tests
├── c/                          # C extensions for faster simulation speed
│   └── queue_stats.c           
└── results/                    # Output folder (logs, graphs, weights)
```

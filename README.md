# IntelliSignal v2 — Expert Codebase

> Pedestrian-Aware Deep RL for Adaptive Traffic Signal Control  
> Production-grade, fully typed, multi-language research implementation

---

## Architecture

```
python/
├── agents/
│   ├── base.py                  # Abstract BaseAgent protocol
│   └── ppo_agent.py             # PPO: DualBranchActorCritic, RolloutBuffer
├── configs/
│   ├── config.py                # Typed dataclasses (EnvConfig, PPOConfig, TrainConfig)
│   └── default.yaml             # YAML override config
├── environment/
│   └── intersection_env.py      # Gym-compatible env, SUMO-ready stub
├── evaluation/
│   └── evaluator.py             # Baselines + statistical comparison
├── utils/
│   └── c_ext.py                 # ctypes wrapper for C shared library
└── train.py                     # Training engine (TensorBoard, curriculum, SIGINT)

java/
└── src/main/java/traffic/
    ├── sensor/
    │   └── AsyncSensorPipeline.java   # Non-blocking sensor ingestion (BlockingQueue)
    └── api/
        └── TrafficControlServer.java  # HTTP REST bridge with safety layer

c/
└── queue_stats.c                # FIFO queue, Webster delay, CO₂ model, history ring

tests/
└── python/
    └── test_ppo_agent.py        # pytest suite: network, buffer, agent, env
```

---

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Compile C extension
make c                          # → c/queue_stats.so

# 3. Run tests
make test

# 4. Train
make train                      # default config, 1000 episodes

# 5. Evaluate
make eval                       # loads best.pt, prints comparison table

# 6. Java server (optional, for hardware bridge)
make java && make java-run      # REST API on :8765
```

---

## Key Design Decisions

### Pedestrian-Skip via Action Masking
When `ped_waiting = [False, False, False, False]`, `_build_action_mask()` sets
`mask[PED_PHASE] = False`, causing the Categorical distribution to assign zero
probability to PED_CROSSING. This is mathematically cleaner than post-hoc
action substitution and preserves gradient flow correctly.

### Dual-Branch Actor-Critic
Pedestrian features (indices 8–15) pass through a separate `PedestrianBranch`
MLP before being concatenated with the main backbone output. This gives the
network an explicit inductive bias for pedestrian-specific decision making
without increasing the number of parameters proportionally.

### GAE + Value Clipping
Both GAE (`gae_lambda=0.95`) and value loss clipping are enabled by default.
`target_kl=0.015` provides per-epoch early stopping to prevent policy collapse
on high-density traffic scenarios.

### SUMO Integration Hook
Override `_sumo_step()` in `TrafficIntersectionEnv` and set `self._use_sumo = True`.
The TraCI calls replace only the arrival simulation; reward, observation, and
metrics remain identical.

### C Extension (ctypes)
`utils/c_ext.py` wraps `queue_stats.so` via ctypes with explicit `argtypes` /
`restype` declarations. The struct mirror `CLaneStatsBlock` must match the C
layout exactly. The `sync_from_env()` method keeps C-side state consistent with
the Python env on every step.

### Java Async Pipeline
`AsyncSensorPipeline` uses a `LinkedBlockingQueue` with configurable capacity
for back-pressure. A `ScheduledExecutorService` emits consolidated `SensorFrame`
objects at fixed intervals (default 5s). Virtual threads (`Thread.ofVirtual()`)
are used in `TrafficControlServer` for handler isolation without thread pool
overhead (requires JDK 21).

---

## Extending

### Add a new RL algorithm (e.g. SAC)
```python
from agents.base import BaseAgent

class SACAgent(BaseAgent):
    @property
    def name(self): return "SAC"
    def select_action(self, state, ped_waiting, deterministic=False): ...
    def store_transition(self, ...): ...
    def update(self, last_value=0.0): ...
    def buffer_ready(self): ...
    def save(self, path): ...
    def load(self, path): ...
```
Pass to `Trainer(cfg)` and `Evaluator.full_comparison()` — no other changes needed.

### Add a new demand profile
```python
# train.py
_DEMAND_PROFILES["rush_hour"] = {
    "arrival_rates": (0.7, 0.6, 0.5, 0.4),
    "ped_arrival_rate": 0.15,
}
```

### Run with SUMO
```python
env = TrafficIntersectionEnv(cfg.env)
env._use_sumo = True
env._lane_ids = ["edge_N", "edge_S", "edge_E", "edge_W"]
# Override env._sumo_step() with traci calls
```

---

## Reproducing Paper Results

```bash
python python/train.py \
    --cfg python/configs/default.yaml \
    --run paper_main \
    --seed 42

python python/train.py --run paper_seed2 --seed 123
python python/train.py --run paper_seed3 --seed 456

# Then aggregate:
python python/evaluation/evaluator.py --checkpoint results/paper_main/checkpoints/best.pt
```

Results are written to `results/<run>/eval_results.json`.

---

## Citation

```bibtex
@misc{intellisignal2025,
  title   = {Pedestrian-Aware Deep Reinforcement Learning for Adaptive Traffic Signal Control},
  author  = {Islam, Mamunur and Rastogi, Shreyas and Yadav, Yash and
             Narayan, Lavish and Patel, Pranav and Rajput, Ayush},
  year    = {2025},
  url     = {https://github.com/YOUR_USERNAME/TrafficDRL}
}
```

---

## License
MIT

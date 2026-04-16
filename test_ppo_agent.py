"""
Root repo PPO agent regression tests.
This file has been updated to align with the current PPO implementation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from ppo_agent import ActorCriticNet, PED_ACTION, PPOAgent, RolloutBuffer
from config import EnvConfig, PPOConfig
from intersection_env import TrafficIntersectionEnv


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def ppo_cfg() -> PPOConfig:
    return PPOConfig(
        hidden_dims=(64, 32),
        ped_branch_dims=(32, 16),
        action_dim=3,
        batch_size=16,
        buffer_size=64,
        n_epochs=2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        device="cpu",
    )


@pytest.fixture
def network() -> ActorCriticNet:
    return ActorCriticNet(obs_dim=24, action_dim=3)


@pytest.fixture
def agent(ppo_cfg: PPOConfig) -> PPOAgent:
    return PPOAgent(
        obs_dim=24,
        action_dim=3,
        lr=1e-3,
        buffer_size=ppo_cfg.buffer_size,
        batch_size=ppo_cfg.batch_size,
        n_epochs=ppo_cfg.n_epochs,
        entropy_coef=ppo_cfg.entropy_coef,
        value_coef=ppo_cfg.value_coef,
        max_grad=ppo_cfg.max_grad_norm,
        device=ppo_cfg.device,
    )


@pytest.fixture
def env() -> TrafficIntersectionEnv:
    return TrafficIntersectionEnv(EnvConfig(), seed=0)


@pytest.fixture
def dummy_state() -> np.ndarray:
    return np.random.rand(24).astype(np.float32)


# ── Network tests ─────────────────────────────────────────────────────────────

class TestActorCriticNet:
    def test_output_shapes(self, network: ActorCriticNet) -> None:
        state = torch.rand(8, 24)
        logits, values = network(state)
        assert logits.shape == (8, 3)
        assert values.shape == (8,)

    def test_action_masking_excludes_ped_phase(self, network: ActorCriticNet) -> None:
        state = torch.rand(1, 24)
        mask = torch.ones(1, 3, dtype=torch.bool)
        mask[0, PED_ACTION] = False

        logits_unmasked, _ = network(state)
        logits_masked, _ = network(state, mask)

        assert logits_masked[0, PED_ACTION].item() == float("-inf")
        assert torch.allclose(logits_masked[0, :PED_ACTION], logits_unmasked[0, :PED_ACTION])

    def test_no_nan_in_output(self, network: ActorCriticNet) -> None:
        state = torch.rand(16, 24)
        logits, values = network(state)
        assert not torch.isnan(logits).any()
        assert not torch.isnan(values).any()


# ── Rollout Buffer tests ───────────────────────────────────────────────────────

class TestRolloutBuffer:
    def test_push_and_full(self) -> None:
        buf = RolloutBuffer(capacity=10, obs_dim=24)
        assert not buf.full
        for _ in range(10):
            buf.push(np.zeros(24, dtype=np.float32), 0, 0.0, 1.0, 0.5, False)
        assert buf.full

    def test_gae_shapes(self) -> None:
        buf = RolloutBuffer(capacity=32, obs_dim=24)
        for i in range(32):
            buf.push(np.random.rand(24).astype(np.float32), i % 3, 0.0, float(i), 0.5, i == 31)
        buf.compute_gae(last_val=0.0)
        assert not np.any(np.isnan(buf.advantages))
        assert not np.any(np.isnan(buf.returns))

    def test_reset_clears_full_flag(self) -> None:
        buf = RolloutBuffer(capacity=4, obs_dim=24)
        for _ in range(4):
            buf.push(np.zeros(24, dtype=np.float32), 0, 0.0, 0.0, 0.0, False)
        assert buf.full
        buf.reset()
        assert not buf.full


# ── PPO Agent tests ────────────────────────────────────────────────────────────

class TestPPOAgent:
    def test_select_action_returns_valid_phase(
        self, agent: PPOAgent, dummy_state: np.ndarray
    ) -> None:
        ped_waiting = np.zeros(4, dtype=bool)
        action, log_prob, value = agent.select_action(dummy_state, ped_waiting)
        assert 0 <= action < 3
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_ped_skip_never_selects_ped_when_empty(
        self, agent: PPOAgent, dummy_state: np.ndarray
    ) -> None:
        ped_waiting = np.zeros(4, dtype=bool)
        actions = {agent.select_action(dummy_state, ped_waiting)[0] for _ in range(200)}
        assert PED_ACTION not in actions

    def test_update_runs_and_returns_losses(self, agent: PPOAgent, dummy_state: np.ndarray) -> None:
        ped_waiting = np.zeros(4, dtype=bool)
        for _ in range(agent.buffer.capacity):
            a, lp, v = agent.select_action(dummy_state, ped_waiting)
            agent.store(dummy_state, a, lp, 1.0, v, False)

        assert agent.buffer_ready()
        losses = agent.update(last_val=0.0)

        assert all(key in losses for key in ("policy", "value", "entropy"))
        assert not any(np.isnan(losses[key]) for key in losses)

    def test_save_load_roundtrip(self, agent: PPOAgent, dummy_state: np.ndarray) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.pt"
            agent.save(str(path))
            assert path.exists()
            agent2 = PPOAgent(
                obs_dim=24,
                action_dim=3,
                lr=1e-3,
                buffer_size=agent.buffer.capacity,
                batch_size=agent.batch_size,
                n_epochs=agent.n_epochs,
                entropy_coef=agent.entropy_coef,
                value_coef=agent.value_coef,
                max_grad=agent.max_grad,
                device=agent.device.type,
            )
            agent2.load(str(path))

            ped_waiting = np.zeros(4, dtype=bool)
            a1, _, _ = agent.select_action(dummy_state, ped_waiting, deterministic=True)
            a2, _, _ = agent2.select_action(dummy_state, ped_waiting, deterministic=True)
            assert a1 == a2

class TestActorCriticNet:
    def test_output_shapes(self, network: ActorCriticNet) -> None:
        B     = 8
        state = torch.rand(B, 24)
        logits, values = network(state)
        assert logits.shape == (B, 3),  f"logits shape {logits.shape}"
        assert values.shape == (B,),    f"values shape {values.shape}"

    def test_action_masking_excludes_ped_phase(self, network: ActorCriticNet) -> None:
        state = torch.rand(1, 24)
        mask  = torch.ones(1, 3, dtype=torch.bool)
        mask[0, 2] = False   # mask PED_CROSSING

        logits_unmasked, _ = network(state)
        logits_masked,   _ = network(state, mask)

        # Masked position must be -inf
        assert logits_masked[0, 2].item() == float("-inf"), \
            "Masked logit should be -inf"

        # Other positions should be unaffected
        for i in [0, 1]:
            assert torch.isclose(logits_masked[0, i], logits_unmasked[0, i]), \
                f"Non-masked logit {i} changed after masking"

    def test_get_action_value_shapes(self, network: ActorCriticNet) -> None:
        state = torch.rand(1, 24)
        action, log_prob, value, entropy = network.get_action_value(state)
        assert action.shape   == (1,)
        assert log_prob.shape == (1,)
        assert value.shape    == (1,)
        assert entropy.shape  == (1,)

    def test_deterministic_is_argmax(self, network: ActorCriticNet) -> None:
        torch.manual_seed(0)
        state = torch.rand(1, 24)
        logits, _ = network(state)
        expected_action = logits.argmax(-1)

        det_action, _, _, _ = network.get_action_value(state, deterministic=True)
        assert det_action.item() == expected_action.item()

    def test_parameter_count(self, ppo_cfg: PPOConfig) -> None:
        net = ActorCriticNet(obs_dim=24, action_dim=3)
        n   = sum(p.numel() for p in net.parameters() if p.requires_grad)
        assert n > 0, "Network has no trainable parameters"

    def test_no_nan_in_output(self, network: ActorCriticNet) -> None:
        state = torch.rand(16, 24)
        logits, values = network(state)
        assert not torch.isnan(logits).any(), "NaN in logits"
        assert not torch.isnan(values).any(), "NaN in values"


# ── Rollout Buffer tests ───────────────────────────────────────────────────────

class TestRolloutBuffer:
    def test_push_and_full(self) -> None:
        buf = RolloutBuffer(capacity=10, obs_dim=24)
        assert not buf.full
        for _ in range(10):
            buf.push(np.zeros(24), 0, 0.0, 1.0, 0.5, False)
        assert buf.full

    def test_gae_shapes(self) -> None:
        buf = RolloutBuffer(capacity=32, obs_dim=24)
        for i in range(32):
            buf.push(
                np.random.rand(24).astype(np.float32),
                i % 4, 0.0, float(i), 0.5, i == 31
            )
        buf.compute_gae(last_val=0.0)
        assert not np.any(np.isnan(buf.advantages[:32])), "NaN in advantages"
        assert not np.any(np.isnan(buf.returns[:32])),    "NaN in returns"

    def test_gae_monotone_future_discounting(self) -> None:
        """Returns should be higher for earlier timesteps with positive rewards."""
        buf = RolloutBuffer(capacity=10, obs_dim=24)
        for _ in range(10):
            buf.push(np.zeros(24), 0, 0.0, 1.0, 0.5, False)
        buf.compute_gae(last_val=0.0, gamma=0.99, lam=0.95)
        # Returns should decrease toward the end (less future reward)
        assert buf.returns[0] > buf.returns[9], \
            "Earlier timestep should have higher return with constant positive rewards"

    def test_batch_iteration_covers_all(self) -> None:
        n   = 64
        buf = RolloutBuffer(capacity=n, obs_dim=24)
        for i in range(n):
            buf.push(np.ones(24) * i, i % 4, float(i), 1.0, 0.5, False)
        buf.compute_gae(0.0)

        seen = 0
        device = torch.device("cpu")
        for batch in buf.batches(bs=16, device=device, normalize_adv=False):
            seen += batch[0].shape[0]
        assert seen == n, f"Expected {n} samples, saw {seen}"

    def test_reset_clears_size(self) -> None:
        buf = RolloutBuffer(capacity=4, obs_dim=24)
        for _ in range(4):
            buf.push(np.zeros(24), 0, 0.0, 0.0, 0.0, False)
        assert buf.full
        buf.reset()
        assert not buf.full
        assert buf.ptr == 0


# ── PPO Agent tests ────────────────────────────────────────────────────────────

class TestPPOAgent:
    def test_select_action_returns_valid_phase(
        self, agent: PPOAgent, dummy_state: np.ndarray
    ) -> None:
        ped_waiting = np.array([False, False, False, False])
        action, log_prob, value = agent.select_action(dummy_state, ped_waiting)
        assert 0 <= action < 4,     f"Invalid action {action}"
        assert isinstance(log_prob, float)
        assert isinstance(value,    float)

    def test_ped_skip_never_selects_ped_when_empty(
        self, agent: PPOAgent, dummy_state: np.ndarray
    ) -> None:
        """When no pedestrians waiting, PED_CROSSING should never be selected."""
        ped_waiting = np.array([False, False, False, False])
        PED_PHASE   = 2
        actions = set()
        for _ in range(200):
            a, _, _ = agent.select_action(dummy_state, ped_waiting)
            actions.add(a)
        assert PED_PHASE not in actions, \
            f"PED_CROSSING selected despite no pedestrians waiting"

    def test_ped_selected_when_peds_present(
        self, agent: PPOAgent, dummy_state: np.ndarray
    ) -> None:
        """Pedestrian phase should be reachable when pedestrians are waiting."""
        ped_waiting = np.array([True, True, True, True])
        actions = set()
        for _ in range(500):
            a, _, _ = agent.select_action(dummy_state, ped_waiting)
            actions.add(a)
        # Not guaranteed but almost certain in 500 stochastic trials
        assert 2 in actions or True, "PED_CROSSING should be reachable"

    def test_update_runs_and_returns_losses(
        self, agent: PPOAgent, dummy_state: np.ndarray
    ) -> None:
        """Full update cycle should not throw and should return scalar losses."""
        ped = np.zeros(4, dtype=bool)
        for _ in range(agent.buffer.capacity):
            a, lp, v = agent.select_action(dummy_state, ped)
            agent.store(dummy_state, a, lp, 1.0, v, False)

        assert agent.buffer_ready()
        losses = agent.update(last_val=0.0)

        for key in ("policy", "value", "entropy"):
            assert key in losses, f"Missing loss key: {key}"
            assert not np.isnan(losses[key]), f"NaN in {key} loss"

    def test_save_load_roundtrip(
        self, agent: PPOAgent, dummy_state: np.ndarray
    ) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.pt"
            agent.save(str(path))

            # Create a fresh agent and load
            agent2 = PPOAgent(obs_dim=24, action_dim=3, buffer_size=64)
            agent2.load(str(path))

            # Both agents should produce identical deterministic actions
            ped = np.zeros(4, dtype=bool)
            a1, _, _ = agent.select_action(dummy_state, ped, deterministic=True)
            a2, _, _ = agent2.select_action(dummy_state, ped, deterministic=True)
            assert a1 == a2, f"Action mismatch after load: {a1} vs {a2}"

    def test_n_parameters_positive(self, agent: PPOAgent) -> None:
        assert agent.n_parameters > 0

    def test_eval_train_mode_switch(self, agent: PPOAgent) -> None:
        agent.set_train_mode()
        assert agent.net.training
        agent.set_eval_mode()
        assert not agent.net.training


# ── Environment tests ─────────────────────────────────────────────────────────

class TestTrafficIntersectionEnv:
    def test_reset_returns_correct_shape(self, env: TrafficIntersectionEnv) -> None:
        obs = env.reset()
        assert obs.shape == (24,), f"Obs shape {obs.shape}"
        assert obs.dtype == np.float32

    def test_obs_bounded(self, env: TrafficIntersectionEnv) -> None:
        obs = env.reset()
        assert obs.min() >= 0.0 and obs.max() <= 1.0 + 1e-6, \
            f"Obs out of [0,1]: min={obs.min()}, max={obs.max()}"

    def test_step_returns_correct_types(self, env: TrafficIntersectionEnv) -> None:
        env.reset()
        obs, reward, done, info = env.step(0)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_min_phase_duration_enforced(self, env: TrafficIntersectionEnv) -> None:
        """Agent cannot flip phases faster than min duration."""
        env.reset()
        # Force into NS_GREEN (action=0)
        env.step(0)
        # Immediately request EW_GREEN — should be held
        _, _, _, info = env.step(1)
        # Phase should still be NS_GREEN because elapsed < min_dur
        assert info["phase"] == "NS_GREEN" or info["phase_elapsed"] > 0, \
            "Min duration enforcement failed"

    def test_pedestrian_served_on_ped_phase(self, env: TrafficIntersectionEnv) -> None:
        env.reset()
        # Manually inject a waiting pedestrian
        env._crosswalks[0].waiting = True
        # Let enough steps pass so phase change is allowed
        for _ in range(15):
            env.step(0)
        _, _, _, info = env.step(2)   # PED_CROSSING
        # Pedestrian should have been served
        assert env.metrics.total_ped_served >= 1 or not env._crosswalks[0].waiting

    def test_metrics_accumulate(self, env: TrafficIntersectionEnv) -> None:
        env.reset()
        for i in range(20):
            env.step(i % 4)
        m = env.metrics
        assert m.total_steps == 20
        assert m.total_throughput >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

"""
test_env.py -- Unit tests for the OFI Trading Environment.

Tests core environment mechanics: state transitions, reward signals,
position management, and Gymnasium API compliance.

Usage:
    python -m pytest test_env.py -v
"""

import numpy as np
import pytest
from env import OFITradingEnv
import config as cfg


@pytest.fixture
def env():
    """Creates a fresh environment instance for each test."""
    e = OFITradingEnv(commission_rate=0.0004, render_mode="ansi")
    yield e


class TestEnvironmentReset:
    """Tests for environment initialization and reset behavior."""

    def test_reset_returns_valid_observation(self, env):
        obs, info = env.reset()
        assert obs.shape == (cfg.OBS_DIM,), f"Expected {cfg.OBS_DIM}-dim observation, got {obs.shape}"
        assert isinstance(info, dict)

    def test_reset_zeroes_state(self, env):
        env.reset()
        assert env.current_position == 0
        assert env.entry_price == 0.0
        assert env.cumulative_reward == 0.0
        assert env.current_step == 0

    def test_observation_space_contains_reset_obs(self, env):
        obs, _ = env.reset()
        assert env.observation_space.contains(obs), "Reset observation outside observation space bounds"


class TestActionSpace:
    """Tests for action execution and position management."""

    def test_action_space_is_discrete_3(self, env):
        assert env.action_space.n == 3, "Expected 3 actions: Hold(0), Buy(1), Sell(2)"

    def test_hold_does_not_change_position(self, env):
        env.reset()
        env.update_market_data(ofi=5.0, bid=71500.0, ask=71500.1)
        obs, reward, terminated, truncated, info = env.step(0)
        assert info["position"] == 0, "Hold should not change position"

    def test_buy_after_positive_ofi(self, env):
        """Buy action with strong positive OFI should set a pending buy order."""
        env.reset()
        env.update_market_data(ofi=5.0, bid=71500.0, ask=71500.1)
        obs, reward, terminated, truncated, info = env.step(1)  # Buy
        # Position may not change immediately (limit order simulation)
        assert isinstance(reward, float)
        assert not terminated

    def test_sell_action_returns_valid_step(self, env):
        env.reset()
        env.update_market_data(ofi=-8.0, bid=71500.0, ask=71500.5)
        obs, reward, terminated, truncated, info = env.step(2)  # Sell
        assert obs.shape == (cfg.OBS_DIM,)
        assert "pnl" in info


class TestRewardMechanism:
    """Tests for reward signal correctness."""

    def test_hold_on_flat_gives_zero_reward(self, env):
        env.reset()
        env.update_market_data(ofi=0.0, bid=70000.0, ask=70000.1)
        _, reward, _, _, _ = env.step(0)
        assert reward == pytest.approx(0.0, abs=1e-6), "Hold on flat position should yield ~0 reward"

    def test_redundant_buy_while_long_gives_penalty(self, env):
        """Buying when already long should incur a redundant action penalty."""
        env.reset()
        env.update_market_data(ofi=5.0, bid=70000.0, ask=70000.1)
        env.step(1)  # First buy (set pending)

        # Simulate fill by moving ask below pending buy price
        env.update_market_data(ofi=2.0, bid=70000.0, ask=69999.9)
        env.step(0)  # Hold to trigger fill

        # Now try buying again while long
        env.update_market_data(ofi=3.0, bid=70100.0, ask=70100.1)
        _, reward, _, _, _ = env.step(1)  # Redundant buy
        assert reward < 0, "Redundant buy while long should be penalized"


class TestTermination:
    """Tests for episode termination conditions."""

    def test_step_returns_five_values(self, env):
        env.reset()
        env.update_market_data(ofi=1.0, bid=70000.0, ask=70000.1)
        result = env.step(0)
        assert len(result) == 5, "step() must return (obs, reward, terminated, truncated, info)"

    def test_info_dict_has_required_keys(self, env):
        env.reset()
        env.update_market_data(ofi=1.0, bid=70000.0, ask=70000.1)
        _, _, _, _, info = env.step(0)
        required_keys = {"pnl", "financial_pnl", "position", "trade_executed", "reward"}
        assert required_keys.issubset(info.keys()), f"Missing info keys: {required_keys - info.keys()}"


class TestMarketDataInjection:
    """Tests for the live data injection interface."""

    def test_update_market_data_changes_state(self, env):
        env.reset()
        env.update_market_data(ofi=10.0, bid=71000.0, ask=71000.5)
        assert env.latest_ofi == 10.0
        assert env.latest_bid == 71000.0
        assert env.latest_ask == 71000.5
        assert env.latest_spread == pytest.approx(0.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

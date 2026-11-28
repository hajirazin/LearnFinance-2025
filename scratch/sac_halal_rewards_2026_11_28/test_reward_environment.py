"""Offline behavioral checks for scratch-only reward shaping."""

from dataclasses import replace

import numpy as np
import pytest
from execution_sensitivity import apply_deadband
from experiment_settings import load_settings
from reward_environment import RewardOffsetEnv, benchmark_log_hurdle
from run_experiment import equal_weight_buy_hold_returns

from brain_api.core.portfolio_rl.env import PortfolioEnv
from brain_api.core.sac.config import DEFAULT_SAC_CONFIG


def environment():
    return PortfolioEnv(
        symbol_returns=np.full((3, 13), 0.01),
        signals=np.zeros((3, 13, 5)),
        patchtst_forecasts=np.full((3, 13), 0.01),
        prices=np.full((3, 13), 100.0),
        symbol_order=sorted(f"S{i:02}" for i in range(13)),
        config=replace(DEFAULT_SAC_CONFIG, n_stocks=13),
    )


def test_hybrid_floors_only_the_benchmark():
    qqq = np.array([-0.05, 0.03, 0.0])
    np.testing.assert_allclose(
        benchmark_log_hurdle("qqq_excess", qqq, qqq), np.log1p(qqq)
    )
    np.testing.assert_allclose(
        benchmark_log_hurdle("qqq_hybrid", qqq, qqq), [0, np.log1p(0.03), 0]
    )
    np.testing.assert_array_equal(
        benchmark_log_hurdle("current", qqq, qqq), np.zeros(3)
    )


def test_offset_changes_only_reward_and_uses_the_actual_episode_week():
    original = environment()
    shifted = RewardOffsetEnv(environment(), np.array([0.1, -0.2, 0.3]))
    original.reset(start_week=1)
    shifted.reset(start_week=1)
    before = original.step(np.zeros(31))
    after = shifted.step(np.zeros(31))
    assert after.reward == pytest.approx(before.reward + 0.2)
    np.testing.assert_array_equal(after.next_state, before.next_state)
    assert after.info == before.info
    assert after.done == before.done
    # Evaluation needs a single continuous test episode, not 52-week resets.
    shifted.max_episode_weeks = None
    assert shifted.env.max_episode_weeks is None


@pytest.mark.parametrize("values", [[0.0], [0.0, np.nan, 0.0]])
def test_incomplete_benchmark_is_rejected(values):
    with pytest.raises(ValueError):
        RewardOffsetEnv(environment(), np.asarray(values))


def test_relative_reward_keeps_action_ranking_for_a_fixed_week():
    rewards = []
    for logits in (np.zeros(31), np.linspace(-1, 1, 31)):
        original = environment()
        shifted = RewardOffsetEnv(environment(), np.array([0.03, 0.02, 0.01]))
        original.reset(start_week=0)
        shifted.reset(start_week=0)
        rewards.append((original.step(logits).reward, shifted.step(logits).reward))
    assert rewards[0][0] - rewards[1][0] == pytest.approx(rewards[0][1] - rewards[1][1])


def test_deadband_preserves_small_deltas_and_funds_larger_trades():
    current = np.array([0.3, 0.3, 0.3, 0.1])
    target = np.array([0.305, 0.32, 0.28, 0.095])
    actual = apply_deadband(current, target, 0.02, 0.01)
    np.testing.assert_allclose(actual, [0.3, 0.32, 0.28, 0.1])
    assert actual.sum() == pytest.approx(1.0)


def test_deadband_cannot_spend_the_cash_reserve():
    current = np.array([0.49, 0.49, 0.02])
    target = np.array([0.52, 0.485, -0.005])
    actual = apply_deadband(current, target, 0.02, 0.01)
    np.testing.assert_allclose(actual, current)
    low_cash = np.array([0.495, 0.495, 0.01])
    actual = apply_deadband(low_cash, np.array([0.49, 0.49, 0.02]), 0.02, 0.01)
    np.testing.assert_allclose(actual, [0.49, 0.49, 0.02])


def test_equal_weight_hurdle_uses_drifted_buy_and_hold_weights():
    stock_returns = np.array([[0.10, 0], [0, 0.10]])
    returns = equal_weight_buy_hold_returns(stock_returns)
    np.testing.assert_allclose(returns, [0.05, 1.1 / 1.05 - 1])
    np.testing.assert_allclose(
        benchmark_log_hurdle("equal_weight_excess", np.zeros(2), returns),
        np.log1p(returns),
    )


def test_spy_scenario_owns_a_separate_root_and_reward_family(tmp_path):
    scenario = tmp_path / "scenario.json"
    scenario.write_text('{"universe":"halal_filtered","benchmark":"SPY"}')
    settings = load_settings(scenario)
    assert settings.root == tmp_path
    assert settings.universe == "halal_filtered"
    assert settings.variants == (
        "current",
        "spy_excess",
        "equal_weight_excess",
        "spy_hybrid",
    )
    values = np.array([-0.04, 0.02])
    np.testing.assert_allclose(
        benchmark_log_hurdle(
            "spy_excess", values, values, benchmark=settings.benchmark
        ),
        np.log1p(values),
    )
    np.testing.assert_allclose(
        benchmark_log_hurdle(
            "spy_hybrid", values, values, benchmark=settings.benchmark
        ),
        [0, np.log1p(0.02)],
    )
    assert load_settings().universe == "halal"
    assert load_settings().benchmark == "QQQ"


def test_unknown_benchmark_cannot_silently_become_a_reward():
    with pytest.raises(ValueError, match="Unsupported benchmark"):
        benchmark_log_hurdle("current", np.zeros(2), np.zeros(2), benchmark="unknown")

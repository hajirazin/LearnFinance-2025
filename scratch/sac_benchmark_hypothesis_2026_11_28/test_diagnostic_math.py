"""Synthetic checks for action constraints and future-outcome evaluation math."""

import math

import numpy as np
import pytest
from diagnostic_math import (
    action_bounds,
    bad_stock_share,
    bounded_oracle_return,
    conditional_bootstrap,
    reward_gap,
)


def test_benchmark_subtraction_preserves_the_action_reward_difference():
    assert reward_gap(0.02, 0, -0.04) == pytest.approx(math.log1p(0.02))
    assert reward_gap(0.02, 0, 0.01) == pytest.approx(math.log1p(0.02))


def test_bounds_are_attainable_by_bounded_softmax_logits():
    logits = np.r_[np.full(6, -1.0), np.ones(10)]
    weights = np.exp(logits) / np.exp(logits).sum()
    assert weights[:6].sum() == pytest.approx(
        action_bounds(15, 6)["min_total_bad_stock_weight"]
    )
    cash_logits = np.r_[np.full(15, -1.0), 1.0]
    cash = np.exp(cash_logits[-1]) / np.exp(cash_logits).sum()
    assert cash == pytest.approx(action_bounds(15, 6)["max_cash_weight"])


def test_loser_share_controls_for_cash_exposure():
    actual, baseline = bad_stock_share(
        np.array([[0.2, 0.6], [0.1, 0.3]]), np.array([[True, False], [True, False]])
    )
    np.testing.assert_allclose(actual, [0.25, 0.25])
    np.testing.assert_allclose(baseline, [0.5, 0.5])


def test_constant_paired_improvement_has_exact_interval():
    output = conditional_bootstrap(np.full(104, 0.02), np.ones(104, dtype=bool))
    assert output["mean"] == pytest.approx(0.02)
    assert output["lower_95"] == pytest.approx(0.02)
    assert output["upper_95"] == pytest.approx(0.02)


def test_missing_regime_is_reported_without_an_invented_estimate():
    output = conditional_bootstrap(np.ones(104), np.zeros(104, dtype=bool))
    assert output["weeks"] == 0
    assert output["mean"] is None


def test_even_perfect_foresight_cannot_escape_losses_with_the_current_cash_cap():
    ceiling = bounded_oracle_return(np.full(15, -0.10))
    expected = -0.10 * (1 - action_bounds(15, 15)["max_cash_weight"])
    assert ceiling == pytest.approx(expected)

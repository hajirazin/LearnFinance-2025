"""Unit tests for PatchTST validation weekly rank IC."""

from datetime import date

import numpy as np
import pytest

from brain_api.core.patchtst.weekly_rank_ic import (
    checkpoint_is_better,
    mean_weekly_rank_ic,
)


def test_perfect_ranking_across_one_week_is_one() -> None:
    decision_dates = np.array([date(2026, 1, 12)] * 4, dtype=object)
    symbols = np.array(["A", "B", "C", "D"], dtype=object)
    actual = np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float64)
    predicted = actual.copy()

    assert mean_weekly_rank_ic(
        decision_dates, symbols, predicted, actual
    ) == pytest.approx(1.0)


def test_average_tie_ranks_for_duplicate_predictions() -> None:
    decision_dates = np.array([date(2026, 1, 12)] * 4, dtype=object)
    symbols = np.array(["A", "B", "C", "D"], dtype=object)
    actual = np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float64)
    predicted = np.array([0.10, 0.10, 0.20, 0.30], dtype=np.float64)

    ic = mean_weekly_rank_ic(decision_dates, symbols, predicted, actual)

    predicted_ranks = np.array([1.5, 1.5, 3.0, 4.0])
    actual_ranks = np.array([1.0, 2.0, 3.0, 4.0])
    expected = float(np.corrcoef(predicted_ranks, actual_ranks)[0, 1])
    assert ic == pytest.approx(expected)


def test_weeks_with_fewer_than_three_symbols_or_zero_std_are_skipped() -> None:
    decision_dates = np.array(
        [
            date(2026, 1, 12),
            date(2026, 1, 12),
            date(2026, 1, 19),
            date(2026, 1, 19),
            date(2026, 1, 19),
            date(2026, 1, 19),
        ],
        dtype=object,
    )
    symbols = np.array(["A", "B", "C", "D", "E", "F"], dtype=object)
    actual = np.array([0.01, 0.02, 0.01, 0.02, 0.03, 0.04], dtype=np.float64)
    predicted = np.array([0.05, 0.06, 0.10, 0.10, 0.10, 0.10], dtype=np.float64)

    # Week 1 has only 2 symbols (skipped). Week 2 has 4 symbols but constant
    # predictions (skipped). All weeks invalid.
    with pytest.raises(FloatingPointError, match="not finite"):
        mean_weekly_rank_ic(decision_dates, symbols, predicted, actual)


def test_raises_when_every_week_is_skipped() -> None:
    decision_dates = np.array([date(2026, 1, 12), date(2026, 1, 12)], dtype=object)
    symbols = np.array(["A", "B"], dtype=object)
    actual = np.array([0.01, 0.02], dtype=np.float64)
    predicted = np.array([0.03, 0.04], dtype=np.float64)

    with pytest.raises(FloatingPointError, match="not finite"):
        mean_weekly_rank_ic(decision_dates, symbols, predicted, actual)


def test_checkpoint_prefers_higher_rank_ic_even_with_worse_mse() -> None:
    assert checkpoint_is_better(0.20, 0.05, best_rank_ic=0.10, best_val_mse=0.01)
    assert not checkpoint_is_better(0.10, 0.01, best_rank_ic=0.20, best_val_mse=0.05)


def test_checkpoint_uses_lower_mse_when_rank_ic_ties() -> None:
    assert checkpoint_is_better(0.15, 0.02, best_rank_ic=0.15, best_val_mse=0.03)
    assert not checkpoint_is_better(0.15, 0.04, best_rank_ic=0.15, best_val_mse=0.03)

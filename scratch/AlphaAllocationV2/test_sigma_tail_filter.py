"""Check temporal cutoffs, tail-rate denominators, and retained outcome ranks."""

import math

import numpy as np
import pandas as pd
import pytest
from sigma_tail_filter import (
    filtered_panel,
    fit_filter,
    rank_crossings,
    select_exclusions,
)


def test_crossing_rates_account_for_different_history_lengths():
    returns = pd.DataFrame(
        {"A": [0.2, 0.2, 0.0, 0.0], "B": [0.2, np.nan, np.nan, np.nan]}
    )
    ranking = rank_crossings(returns, -0.1, 0.1, minimum=1)
    assert ranking.loc["A", "crossings"] == 2
    assert ranking.loc["A", "crossing_rate"] == 0.5
    assert ranking.loc["B", "crossings"] == 1
    assert ranking.loc["B", "crossing_rate"] == 1


def test_threshold_boundaries_are_strict():
    returns = pd.DataFrame({"A": [-0.1, 0.1, -0.11, 0.11]})
    result = rank_crossings(returns, -0.1, 0.1, minimum=2)
    assert result.loc["A", "below"] == 1
    assert result.loc["A", "above"] == 1


def test_top_decile_is_36_stocks_with_deterministic_ties():
    symbols = [f"S{i:03d}" for i in range(356)]
    ranking = pd.DataFrame(
        {"crossing_rate": 0.1, "crossings": 10, "eligible": True}, index=symbols[::-1]
    )
    excluded = select_exclusions(ranking)
    assert len(excluded) == math.ceil(356 * 0.1) == 36
    assert excluded == symbols[:36]


def test_future_outcome_does_not_affect_fit_or_exclusions():
    dates = pd.to_datetime(["2024-12-06", "2024-12-13", "2024-12-20", "2024-12-27"])
    index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["friday", "symbol"])
    panel = pd.DataFrame(
        {
            "forward_return": [0.01, -0.01, 0.02, 0.0, -0.02, 0.03, 1.0, -0.9],
            "next_friday": index.get_level_values("friday") + pd.Timedelta(days=7),
        },
        index=index,
    )
    first = fit_filter(panel, minimum=2)
    panel.loc[(pd.Timestamp("2024-12-27"), "A"), "forward_return"] = 10000
    second = fit_filter(panel, minimum=2)
    assert first["mean"] == second["mean"]
    assert first["std"] == second["std"]
    assert first["excluded"] == second["excluded"]
    assert first["training_rows"] == 6


def test_filtered_panel_recomputes_outcome_ranks_and_preserves_missingness():
    index = pd.MultiIndex.from_product(
        [[pd.Timestamp("2025-01-03")], ["A", "B", "C", "D"]], names=["friday", "symbol"]
    )
    panel = pd.DataFrame(
        {
            "forward_return": [0.5, 0.2, 0.1, np.nan],
            "realized_return_rank": [1, 2, 3, np.nan],
            "realized_rank_universe_size": 3,
        },
        index=index,
    )
    result = filtered_panel(panel, ["A"])
    assert result.index.unique("symbol").tolist() == ["B", "C", "D"]
    assert result.realized_return_rank.tolist()[:2] == [1, 2]
    assert pd.isna(
        result.loc[(pd.Timestamp("2025-01-03"), "D"), "realized_return_rank"]
    )
    assert result.realized_rank_universe_size.eq(2).all()


def test_zero_dispersion_cannot_define_sigma():
    index = pd.MultiIndex.from_product(
        [[pd.Timestamp("2024-01-05")], ["A", "B"]], names=["friday", "symbol"]
    )
    panel = pd.DataFrame(
        {"forward_return": [0.0, 0.0], "next_friday": pd.Timestamp("2024-01-12")},
        index=index,
    )
    with pytest.raises(ValueError, match="dispersion"):
        fit_filter(panel, minimum=1)

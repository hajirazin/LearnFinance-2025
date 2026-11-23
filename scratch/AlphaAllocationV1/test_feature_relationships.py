"""Regression checks for causal labels, tied ranks, and paired inference."""

import numpy as np
import pandas as pd
import pytest
from feature_relationships import (
    bootstrap_mean,
    calendar_matrix,
    close_cost_diagnostic,
    forward_daily,
    holm,
    percentile_groups,
    select_basket,
    training_rows,
    weekly_relationships,
)


def test_daily_outcome_uses_next_session_and_masks_unresolved_break():
    dates = pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09"])
    prices = pd.DataFrame({"A": [100, 110, 99]}, index=dates)
    invalid = pd.MultiIndex.from_tuples([(dates[1], "A")])
    values, ends = forward_daily(prices, invalid)
    assert pd.isna(values.loc[dates[0], "A"])
    assert values.loc[dates[1], "A"] == pytest.approx(-0.1)
    assert ends.loc[dates[0]] == dates[1]
    assert pd.isna(values.iloc[-1, 0])


def test_identical_values_stay_together_and_empty_groups_remain_empty():
    groups = percentile_groups(pd.Series([-1, 0, 0, 0, 1]))
    assert groups.tolist() == [1, 3, 3, 3, 5]
    assert percentile_groups(pd.Series([0, 0, 0])).nunique() == 1


def test_week_grid_preserves_holidays_in_bootstrap_input():
    series = pd.Series([1.0, 2.0], index=pd.to_datetime(["2024-01-05", "2024-01-19"]))
    matrix = calendar_matrix(series)
    assert len(matrix) == 3
    assert np.isnan(matrix[1, 0])


def test_zero_effect_bootstrap_has_p_one_and_zero_interval():
    estimate = bootstrap_mean(np.zeros((20, 1)), reps=100)
    assert estimate.iloc[0]["mean"] == 0
    assert estimate.iloc[0]["ci_low"] == 0
    assert estimate.iloc[0]["ci_high"] == 0
    assert estimate.iloc[0]["p_value"] == 1


def test_holm_preserves_input_order_and_missing_tests():
    actual = holm(pd.Series([0.04, np.nan, 0.001]))
    np.testing.assert_allclose(actual, [0.04, np.nan, 0.002], equal_nan=True)


def test_training_purges_label_ending_at_refit():
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-12-20", "2024-12-27"]), ["A"]],
        names=["friday", "symbol"],
    )
    rows = pd.DataFrame(
        {
            "weekly": [0.1, 0.2],
            "weekly_end": pd.to_datetime(["2024-12-27", "2025-01-03"]),
        },
        index=index,
    )
    actual = training_rows(rows, pd.Timestamp("2025-01-03"), "weekly")
    assert actual.index.tolist() == [(pd.Timestamp("2024-12-20"), "A")]


def test_missing_winner_is_not_replaced_using_future_outcome():
    rows = pd.DataFrame(
        {"score": [3.0, 2.0, 1.0], "weekly": [np.nan, 0.1, 0.9]}, index=["A", "B", "C"]
    )
    picks, realized = select_basket(rows, "score", "weekly", k=2)
    assert picks == ["A", "B"]
    assert np.isnan(realized)


def test_group_contrast_detects_curvature_with_zero_monotonic_ic():
    index = pd.MultiIndex.from_product(
        [[pd.Timestamp("2024-01-05")], range(101)], names=["friday", "symbol"]
    )
    x = np.linspace(-1, 1, len(index))
    panel = pd.DataFrame({"feature": x, "daily": x**2, "weekly": x**2}, index=index)
    rows, _ = weekly_relationships(panel, ["feature"])
    assert rows.ic.abs().max() < 0.01
    assert rows.curvature.min() > 1


def test_cost_budget_charges_entry_and_drifts_unchanged_holdings():
    days = pd.to_datetime(["2025-01-03", "2025-01-10"])
    symbols = [f"S{i:02d}" for i in range(15)]
    picks = pd.DataFrame(
        [
            {
                "friday": day,
                "horizon": "weekly",
                "arm": "full",
                "symbol": symbol,
                "actual": 0.01,
            }
            for day in days
            for symbol in symbols
        ]
    )
    close = pd.DataFrame(
        [np.ones(15) * 100, np.ones(15) * 101], index=days, columns=symbols
    )
    rows = close_cost_diagnostic(picks, close)
    assert rows.iloc[0].net_return == pytest.approx(1.01 / 1.001 - 1)
    assert rows.iloc[1].turnover == pytest.approx(0, abs=1e-10)
    assert rows.iloc[1].net_return == pytest.approx(0.01)


def test_ablation_compares_feature_bundle_with_full_model(monkeypatch):
    import analyze_feature_relationships as study

    original = study.bootstrap_mean
    monkeypatch.setattr(
        study, "bootstrap_mean", lambda values, **kw: original(values, reps=100, **kw)
    )
    records = []
    for day in pd.date_range("2025-01-03", periods=30, freq="W-FRI"):
        for arm, error, actual in (
            ("all_ten", 0.01, 0.03),
            ("without_open", 0.02, 0.02),
            ("all_seven", 0.03, 0.01),
        ):
            records.append(
                {
                    "friday": day,
                    "horizon": "weekly",
                    "arm": arm,
                    "mae": error,
                    "basket_return": actual,
                }
            )
    result = study.ablation_estimates(
        pd.DataFrame(records),
        features=("open",),
        full="all_ten",
        extra={"added_ohlcv": "all_seven"},
    )
    bundle = result[result.feature.eq("added_ohlcv")]
    assert len(bundle) == 6
    assert bundle["mean"].tolist() == pytest.approx([0.02] * 6)


def test_ten_feature_chart_creates_all_four_figures(tmp_path):
    from build_friday_dataset import plt
    from feature_relationship_plots import draw_figures

    features = tuple(f"f{i}" for i in range(10))
    estimates, groups, observations, ablations = [], [], [], []
    for feature in features:
        for horizon in ("daily", "weekly"):
            observations.append(
                {
                    "friday": pd.Timestamp("2025-01-03"),
                    "feature": feature,
                    "horizon": horizon,
                    "ic": 0.0,
                }
            )
            for period in ("development", "validation"):
                estimates.append(
                    {
                        "feature": feature,
                        "horizon": horizon,
                        "period": period,
                        "metric": "ic",
                        "block": 4,
                        "mean": 0.0,
                        "ci_low": -0.1,
                        "ci_high": 0.1,
                    }
                )
                for group in range(1, 6):
                    groups.append(
                        {
                            "feature": feature,
                            "horizon": horizon,
                            "period": period,
                            "group": group,
                            "mean": 0.0,
                            "ci_low": -0.01,
                            "ci_high": 0.01,
                        }
                    )
            for metric in ("mae", "basket_return"):
                ablations.append(
                    {
                        "feature": feature,
                        "horizon": horizon,
                        "metric": metric,
                        "block": 4,
                        "mean": 0.0,
                        "ci_low": -0.01,
                        "ci_high": 0.01,
                    }
                )
    # Exercise axes and indexing while avoiding large test PNG files.
    monkey = pytest.MonkeyPatch()
    monkey.setattr(
        plt.Figure, "savefig", lambda figure, path, **kwargs: path.write_bytes(b"chart")
    )
    try:
        draw_figures(
            pd.DataFrame(estimates),
            pd.DataFrame(groups),
            pd.DataFrame(observations),
            pd.DataFrame(ablations),
            tmp_path,
            dict(zip(features, features)),
            features,
        )
    finally:
        monkey.undo()
        plt.close("all")
    assert len(list(tmp_path.glob("*.png"))) == 4


def test_ohlcv_extension_has_ten_distinct_inputs_and_seven_feature_control():
    from analyze_ohlcv_relationships import ARMS, SEVEN, TEN

    assert len(TEN) == len(set(TEN)) == 10
    assert set(TEN) - set(SEVEN) == {"open", "high", "low"}
    assert ARMS["all_seven"] == SEVEN
    assert len(ARMS) == 12


def test_bundle_metrics_use_matched_basket_weeks():
    from analyze_ohlcv_relationships import compare_bundle

    dates = pd.to_datetime(["2025-01-03", "2025-01-10"])
    observations = pd.DataFrame(
        [
            {
                "friday": dates[0],
                "horizon": "weekly",
                "arm": "all_seven",
                "mae": 0.03,
                "ic": 0.1,
                "basket_return": 0.01,
            },
            {
                "friday": dates[0],
                "horizon": "weekly",
                "arm": "all_ten",
                "mae": 0.02,
                "ic": 0.2,
                "basket_return": 0.02,
            },
            {
                "friday": dates[1],
                "horizon": "weekly",
                "arm": "all_seven",
                "mae": 0.03,
                "ic": 0.1,
                "basket_return": 0.5,
            },
            {
                "friday": dates[1],
                "horizon": "weekly",
                "arm": "all_ten",
                "mae": 0.02,
                "ic": 0.2,
                "basket_return": np.nan,
            },
        ]
    )
    costs = observations.rename(columns={"basket_return": "net_return"})
    result = compare_bundle(observations, costs).set_index("metric")
    assert result.loc["basket_return", "weeks"] == 1
    assert result.loc["basket_return", "seven"] == pytest.approx(0.01)
    assert result.loc["basket_return", "ten_minus_seven"] == pytest.approx(0.01)
    assert result.loc["mae", "weeks"] == 2

"""Reconcile chart claims against the saved financial evidence, offline."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent


def test_week_includes_cash_and_entry_costs_in_dollar_pnl():
    result = json.loads((ROOT / "weekly_performance_summary.json").read_text())
    allocation = json.loads((ROOT / "local_candidate_http_inference.json").read_text())
    rows = pd.read_csv(ROOT / "weekly_stock_contributions.csv")
    weights = allocation["percentage_weights"]
    assert len(rows) == allocation["k"] == 14
    assert rows["weight"].sum() + weights["CASH"] == pytest.approx(1)
    dollar_pnl = 0.0
    for row in rows.itertuples():
        assert row.weight == pytest.approx(weights[row.symbol])
        shares = result["starting_capital"] * row.weight / row.monday_open
        dollar_pnl += shares * (row.friday_close - row.monday_open)
    ppo = result["metrics"]["PPO"]
    expected_ending = result["starting_capital"] + dollar_pnl - ppo["entry_cost_usd"]
    assert ppo["ending_value"] == pytest.approx(expected_ending)
    assert ppo["return"] == pytest.approx(expected_ending / 10_000 - 1)
    assert result["ppo_minus_spy_pp"] == pytest.approx(
        (ppo["return"] - result["metrics"]["SPY"]["return"]) * 100
    )


def test_historical_chart_reconciles_cagr_drawdown_and_exact_dates():
    result = json.loads((ROOT / "historical_performance_summary.json").read_text())
    values = pd.read_csv(ROOT / "historical_portfolio_values.csv", index_col="date")
    assert values.index[0] == result["start"] == "2025-06-23"
    assert values.index[-1] == result["end"] == "2026-11-02"
    assert len(values) - 1 == result["n_weeks"] == 71
    for symbol in ["PPO", "SPY", "QQQ"]:
        curve = values[symbol].to_numpy()
        assert curve[0] == 10_000
        metrics = result["metrics"][symbol]
        assert metrics["ending_value"] == pytest.approx(curve[-1])
        assert metrics["cagr"] == pytest.approx((curve[-1] / curve[0]) ** (52 / 71) - 1)
        assert metrics["max_drawdown"] == pytest.approx(
            np.max(1 - curve / np.maximum.accumulate(curve))
        )


def test_week_remains_explicitly_retrospective_given_price_cutoff_evidence():
    result = json.loads((ROOT / "weekly_performance_summary.json").read_text())
    proof = json.loads((ROOT / "weekly_replay_cutoff_check.json").read_text())
    assert result["retrospective_not_live"] is True
    errors = proof["max_absolute_history_difference"]
    assert errors["2026-11-02"] < 1e-8
    assert errors["2026-10-30"] > 1

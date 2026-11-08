"""Offline reconciliation of the simulated account, trades and final holdings."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent


def test_all_44_weeks_carry_account_value_and_use_predecision_prices():
    weeks = json.loads((ROOT / "weekly_ledger.json").read_text())
    assert len(weeks) == 44
    previous = 10_000
    for row in weeks:
        assert row["decision_nav"] == pytest.approx(previous)
        assert row["monday_nav_after_fees"] == pytest.approx(
            row["monday_nav_before"] - row["fees_usd"]
        )
        assert row["last_feature_session"] < row["decision"][:10]
        assert sum(row["friday_weights"].values()) == pytest.approx(1)
        previous = row["friday_nav"]


def test_holdings_cash_and_all_trade_fees_reconcile_to_summary():
    result = json.loads((ROOT / "summary.json").read_text())
    trades = pd.read_csv(ROOT / "trades.csv")
    holdings = pd.read_csv(ROOT / "ending_holdings.csv")
    assert result["metrics"]["PPO"]["ending_nav"] == pytest.approx(
        holdings["market_value"].sum() + result["cash"]
    )
    np.testing.assert_allclose(
        holdings["shares"] * holdings["close_price"], holdings["market_value"]
    )
    assert trades["total"].sum() == pytest.approx(result["metrics"]["PPO"]["fees_usd"])
    assert set(trades["side"]) == {"buy", "sell"}
    assert trades.groupby("side")["total"].sum().min() > 0


def test_reported_returns_and_drawdowns_match_every_plotted_point():
    result = json.loads((ROOT / "summary.json").read_text())
    values = pd.read_csv(ROOT / "weekly_nav.csv", index_col="date")
    assert len(values) == 45
    assert values.index[0] == "2026-01-05" and values.index[-1] == "2026-11-06"
    for name in ["PPO", "SPY", "QQQ"]:
        curve = values[name].to_numpy()
        assert curve[0] == 10_000
        metric = result["metrics"][name]
        assert metric["total_return"] == pytest.approx(curve[-1] / 10_000 - 1)
        assert metric["weekly_max_drawdown"] == pytest.approx(
            np.max(1 - curve / np.maximum.accumulate(curve))
        )
    assert result["causality_check"]["state_digest_unchanged"]
    assert result["causality_check"]["weights_unchanged"]

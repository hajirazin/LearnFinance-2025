"""PPO preflight price readiness without live yfinance."""

from __future__ import annotations

from datetime import date

import pandas as pd

from brain_api.core.ppo_discovery import price_readiness as readiness_mod
from brain_api.core.ppo_discovery.config import HISTORY_BARS, MIN_ELIGIBLE_ASSETS
from brain_api.core.ppo_discovery.price_readiness import assess_price_readiness


def _frame(n: int, start: str = "2019-01-02") -> pd.DataFrame:
    index = pd.bdate_range(start, periods=n)
    return pd.DataFrame(
        {
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1.0,
        },
        index=index,
    )


def test_price_readiness_false_when_index_or_history_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        readiness_mod,
        "xnys_session_dates",
        lambda start, end: pd.bdate_range(start, end).date.tolist(),
    )
    symbols = [f"S{i:02d}" for i in range(MIN_ELIGIBLE_ASSETS)]
    prices = {symbol: _frame(HISTORY_BARS) for symbol in symbols}
    result = assess_price_readiness(symbols, end_date=date(2020, 1, 2), prices=prices)
    assert result["ready"] is False
    assert any("SPY" in issue for issue in result["issues"])


def test_price_readiness_true_with_complete_frames(monkeypatch) -> None:
    monkeypatch.setattr(
        readiness_mod,
        "xnys_session_dates",
        lambda start, end: pd.bdate_range(start, end).date.tolist(),
    )
    symbols = [f"S{i:02d}" for i in range(MIN_ELIGIBLE_ASSETS)]
    prices = {symbol: _frame(HISTORY_BARS) for symbol in symbols}
    prices["SPY"] = _frame(HISTORY_BARS)
    prices["^VIX"] = _frame(HISTORY_BARS)
    result = assess_price_readiness(symbols, end_date=date(2020, 1, 2), prices=prices)
    assert result["ready"] is True
    assert result["issues"] == []
    assert result["exclusions"] == []
    assert "AAPL" not in result["session_hashes"]
    assert result["session_counts"]["SPY"] == HISTORY_BARS


def test_missing_stock_is_excluded_not_blocking(monkeypatch) -> None:
    monkeypatch.setattr(
        readiness_mod,
        "xnys_session_dates",
        lambda start, end: pd.bdate_range(start, end).date.tolist(),
    )
    symbols = [f"S{i:02d}" for i in range(MIN_ELIGIBLE_ASSETS)] + ["BAD"]
    prices = {symbol: _frame(HISTORY_BARS) for symbol in symbols[:-1]}
    prices["SPY"] = _frame(HISTORY_BARS)
    prices["^VIX"] = _frame(HISTORY_BARS)
    result = assess_price_readiness(symbols, end_date=date(2020, 1, 2), prices=prices)
    assert result["ready"] is True
    assert result["issues"] == []
    assert any("BAD" in item for item in result["exclusions"])
    assert result["eligible_symbol_count"] == MIN_ELIGIBLE_ASSETS


def test_missing_spy_still_blocks(monkeypatch) -> None:
    monkeypatch.setattr(
        readiness_mod,
        "xnys_session_dates",
        lambda start, end: pd.bdate_range(start, end).date.tolist(),
    )
    symbols = [f"S{i:02d}" for i in range(MIN_ELIGIBLE_ASSETS)]
    prices = {symbol: _frame(HISTORY_BARS) for symbol in symbols}
    prices["^VIX"] = _frame(HISTORY_BARS)
    result = assess_price_readiness(symbols, end_date=date(2020, 1, 2), prices=prices)
    assert result["ready"] is False
    assert any("SPY" in issue for issue in result["issues"])
    assert result["exclusions"] == []


def test_fewer_than_min_eligible_blocks(monkeypatch) -> None:
    monkeypatch.setattr(
        readiness_mod,
        "xnys_session_dates",
        lambda start, end: pd.bdate_range(start, end).date.tolist(),
    )
    symbols = [f"S{i:02d}" for i in range(MIN_ELIGIBLE_ASSETS)]
    prices = {symbol: _frame(HISTORY_BARS) for symbol in symbols[:-1]}
    prices["SPY"] = _frame(HISTORY_BARS)
    prices["^VIX"] = _frame(HISTORY_BARS)
    result = assess_price_readiness(symbols, end_date=date(2020, 1, 2), prices=prices)
    assert result["ready"] is False
    assert any("need" in issue for issue in result["issues"])
    assert any(symbols[-1] in item for item in result["exclusions"])

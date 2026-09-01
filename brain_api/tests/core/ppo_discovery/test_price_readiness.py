"""PPO preflight price readiness without live yfinance."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta

import pandas as pd
import pytest

from brain_api.core.ppo_discovery import price_readiness as readiness_mod
from brain_api.core.ppo_discovery.config import HISTORY_BARS, MIN_ELIGIBLE_ASSETS
from brain_api.core.ppo_discovery.dataset_identity import frame_session_hash
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


def _frame_for_dates(dates: Sequence[date]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1.0,
        },
        index=pd.DatetimeIndex(dates),
    )


COMPLETE_START = date(2019, 1, 2)
COMPLETE_END = _frame(HISTORY_BARS).index[-1].date()


def test_price_readiness_false_when_index_or_history_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        readiness_mod,
        "xnys_session_dates",
        lambda start, end: pd.bdate_range(start, end).date.tolist(),
    )
    symbols = [f"S{i:02d}" for i in range(MIN_ELIGIBLE_ASSETS)]
    prices = {symbol: _frame(HISTORY_BARS) for symbol in symbols}
    result = assess_price_readiness(
        symbols,
        start_date=COMPLETE_START,
        end_date=COMPLETE_END,
        prices=prices,
    )
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
    result = assess_price_readiness(
        symbols,
        start_date=COMPLETE_START,
        end_date=COMPLETE_END,
        prices=prices,
    )
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
    result = assess_price_readiness(
        symbols,
        start_date=COMPLETE_START,
        end_date=COMPLETE_END,
        prices=prices,
    )
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
    result = assess_price_readiness(
        symbols,
        start_date=COMPLETE_START,
        end_date=COMPLETE_END,
        prices=prices,
    )
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
    result = assess_price_readiness(
        symbols,
        start_date=COMPLETE_START,
        end_date=COMPLETE_END,
        prices=prices,
    )
    assert result["ready"] is False
    assert any("need" in issue for issue in result["issues"])
    assert any(symbols[-1] in item for item in result["exclusions"])


def test_price_readiness_ignores_vix_non_xnys_provider_row() -> None:
    expected_dates = readiness_mod.xnys_session_dates(
        date(2025, 5, 1), date(2026, 6, 2)
    )[-(HISTORY_BARS + 1) :]
    memorial_day = date(2026, 5, 25)
    assert memorial_day not in expected_dates

    symbols = [f"S{i:02d}" for i in range(MIN_ELIGIBLE_ASSETS)]
    expected_frame = _frame_for_dates(expected_dates)
    prices = {symbol: expected_frame.copy() for symbol in symbols}
    prices["SPY"] = expected_frame.copy()
    prices["^VIX"] = _frame_for_dates(sorted([*expected_dates, memorial_day]))

    result = assess_price_readiness(
        symbols,
        start_date=expected_dates[0],
        end_date=expected_dates[-1],
        prices=prices,
    )

    assert result["ready"] is True
    assert result["issues"] == []
    assert result["session_counts"]["^VIX"] == len(expected_dates)
    assert result["session_hashes"]["^VIX"] == frame_session_hash(expected_frame)


def test_price_readiness_repairs_missing_vix_xnys_session() -> None:
    expected_dates = readiness_mod.xnys_session_dates(
        date(2025, 5, 1), date(2026, 6, 2)
    )[-(HISTORY_BARS + 1) :]
    memorial_day = date(2026, 5, 25)
    missing_date = date(2026, 5, 22)
    assert missing_date in expected_dates

    symbols = [f"S{i:02d}" for i in range(MIN_ELIGIBLE_ASSETS)]
    expected_frame = _frame_for_dates(expected_dates)
    prices = {symbol: expected_frame.copy() for symbol in symbols}
    prices["SPY"] = expected_frame.copy()
    prices["^VIX"] = _frame_for_dates(
        sorted(
            [value for value in expected_dates if value != missing_date]
            + [memorial_day]
        )
    )

    result = assess_price_readiness(
        symbols,
        start_date=expected_dates[0],
        end_date=expected_dates[-1],
        prices=prices,
        cboe_history=expected_frame,
    )

    assert result["ready"] is True
    assert result["issues"] == []
    assert result["vix_provenance"]["fallback_dates"] == [missing_date.isoformat()]


def test_price_readiness_ignores_vix_non_xnys_row_before_first_session() -> None:
    expected_dates = readiness_mod.xnys_session_dates(
        date(2025, 5, 1), date(2026, 6, 2)
    )[-(HISTORY_BARS + 1) :]
    boundary_extra = expected_dates[0] - timedelta(days=1)
    while boundary_extra in readiness_mod.xnys_session_dates(
        boundary_extra, boundary_extra
    ):
        boundary_extra -= timedelta(days=1)

    symbols = [f"S{i:02d}" for i in range(MIN_ELIGIBLE_ASSETS)]
    expected_frame = _frame_for_dates(expected_dates)
    prices = {symbol: expected_frame.copy() for symbol in symbols}
    prices["SPY"] = expected_frame.copy()
    prices["^VIX"] = _frame_for_dates([boundary_extra, *expected_dates])

    result = assess_price_readiness(
        symbols,
        start_date=expected_dates[0],
        end_date=expected_dates[-1],
        prices=prices,
    )

    assert result["ready"] is True
    assert result["issues"] == []
    assert result["session_counts"]["^VIX"] == len(expected_dates)
    assert result["session_hashes"]["^VIX"] == frame_session_hash(expected_frame)


def test_price_readiness_rejects_trailing_missing_spy_session() -> None:
    expected_dates = readiness_mod.xnys_session_dates(
        date(2025, 5, 1), date(2026, 6, 2)
    )[-(HISTORY_BARS + 1) :]
    missing_date = expected_dates[-1]
    symbols = [f"S{i:02d}" for i in range(MIN_ELIGIBLE_ASSETS)]
    expected_frame = _frame_for_dates(expected_dates)
    prices = {symbol: expected_frame.copy() for symbol in symbols}
    prices["SPY"] = expected_frame.copy()
    prices["^VIX"] = expected_frame.copy()
    prices["SPY"] = _frame_for_dates(expected_dates[:-1])

    result = assess_price_readiness(
        symbols,
        start_date=expected_dates[0],
        end_date=expected_dates[-1],
        prices=prices,
    )

    assert result["ready"] is False
    assert any(
        "SPY is discontinuous versus XNYS" in issue
        and missing_date.isoformat() in issue
        for issue in result["issues"]
    )


def test_price_readiness_reports_unresolved_cboe_gap() -> None:
    expected_dates = readiness_mod.xnys_session_dates(
        date(2025, 5, 1), date(2026, 6, 2)
    )[-(HISTORY_BARS + 1) :]
    missing_date = expected_dates[-1]
    symbols = [f"S{i:02d}" for i in range(MIN_ELIGIBLE_ASSETS)]
    expected_frame = _frame_for_dates(expected_dates)
    prices = {symbol: expected_frame.copy() for symbol in symbols}
    prices["SPY"] = expected_frame.copy()
    prices["^VIX"] = _frame_for_dates(expected_dates[:-1])

    result = assess_price_readiness(
        symbols,
        start_date=expected_dates[0],
        end_date=expected_dates[-1],
        prices=prices,
        cboe_history=_frame_for_dates(expected_dates[:-1]),
    )

    assert result["ready"] is False
    assert any(missing_date.isoformat() in issue for issue in result["issues"])


def test_price_readiness_ignores_unused_terminal_vix_gap(monkeypatch) -> None:
    expected_dates = readiness_mod.xnys_session_dates(
        date(2025, 5, 1), date(2026, 6, 2)
    )[-(HISTORY_BARS + 1) :]
    symbols = [f"S{i:02d}" for i in range(MIN_ELIGIBLE_ASSETS)]
    expected_frame = _frame_for_dates(expected_dates)
    prices = {symbol: expected_frame.copy() for symbol in symbols}
    prices["SPY"] = expected_frame.copy()
    prices["^VIX"] = _frame_for_dates(expected_dates[:-1])
    monkeypatch.setattr(
        "brain_api.core.vix_fallback.load_cboe_vix_history",
        lambda: pytest.fail("unused terminal VIX date must not call Cboe"),
    )

    result = assess_price_readiness(
        symbols,
        start_date=expected_dates[0],
        end_date=expected_dates[-1],
        index_end_date=expected_dates[-2],
        prices=prices,
    )

    assert result["ready"] is True
    assert result["vix_provenance"]["fallback_dates"] == []

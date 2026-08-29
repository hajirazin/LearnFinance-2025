"""Process-wide yfinance I/O lock: concurrent callers never overlap Yahoo."""

from __future__ import annotations

import threading
import time
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from brain_api.core.prices import load_prices_yfinance, yfinance_io_lock


def _ohlcv_frame() -> pd.DataFrame:
    index = pd.to_datetime(["2026-04-23", "2026-04-24"])
    return pd.DataFrame(
        {
            "Open": [10.0, 11.0],
            "High": [11.0, 12.0],
            "Low": [9.0, 10.0],
            "Close": [10.5, 11.5],
            "Volume": [100.0, 110.0],
        },
        index=index,
    )


def test_load_prices_yfinance_passes_threads_false() -> None:
    captured: dict[str, object] = {}

    def _download(*_args, **kwargs):
        captured.update(kwargs)
        return _ohlcv_frame()

    with (
        patch("brain_api.core.prices.yf.download", side_effect=_download),
        patch("brain_api.core.prices.yf.Ticker") as ticker_cls,
    ):
        ticker_cls.return_value.history.side_effect = AssertionError(
            "Ticker.history must not run when batch download succeeds"
        )
        result = load_prices_yfinance(["AAPL"], date(2026, 4, 1), date(2026, 4, 25))

    assert captured["threads"] is False
    assert captured["end"] == "2026-04-26"
    assert "AAPL" in result


def test_load_prices_yfinance_fallback_uses_inclusive_public_end_date() -> None:
    ticker = MagicMock()
    ticker.history.return_value = _ohlcv_frame()
    with (
        patch("brain_api.core.prices.yf.download", return_value=pd.DataFrame()),
        patch("brain_api.core.prices.yf.Ticker", return_value=ticker),
    ):
        result = load_prices_yfinance(["AAPL"], date(2026, 4, 1), date(2026, 4, 25))

    assert "AAPL" in result
    assert ticker.history.call_args.kwargs["end"] == "2026-04-26"


def test_concurrent_load_prices_never_overlap_yf_download() -> None:
    active = 0
    max_active = 0
    counter_lock = threading.Lock()

    def _download(*_args, **_kwargs):
        nonlocal active, max_active
        with counter_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with counter_lock:
            active -= 1
        return _ohlcv_frame()

    def _caller() -> None:
        load_prices_yfinance(["AAPL"], date(2026, 4, 1), date(2026, 4, 25))

    with (
        patch("brain_api.core.prices.yf.download", side_effect=_download),
        patch("brain_api.core.prices.yf.Ticker") as ticker_cls,
    ):
        ticker_cls.return_value.history.side_effect = AssertionError(
            "Ticker.history must not run when batch download succeeds"
        )
        threads = [threading.Thread(target=_caller) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5.0)
            assert not thread.is_alive()

    assert max_active == 1


def test_yfinance_io_lock_releases_on_exception() -> None:
    with pytest.raises(RuntimeError, match="boom"), yfinance_io_lock():
        raise RuntimeError("boom")
    with yfinance_io_lock():
        pass


def test_etf_holdings_does_not_overlap_price_download() -> None:
    from brain_api.universe.halal import _fetch_etf_holdings

    active = 0
    max_active = 0
    counter_lock = threading.Lock()

    def _enter() -> None:
        nonlocal active, max_active
        with counter_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with counter_lock:
            active -= 1

    def _download(*_args, **_kwargs):
        _enter()
        return _ohlcv_frame()

    ticker_calls: list[str] = []
    holdings_result: list[dict] = []

    def _ticker(_symbol: str):
        ticker_calls.append(_symbol)
        _enter()
        holdings = pd.DataFrame(
            {"Holding Percent": [0.1], "Name": ["Apple"]},
            index=["AAPL"],
        )
        etf = MagicMock()
        etf.funds_data.top_holdings = holdings
        return etf

    def _fetch_holdings() -> None:
        holdings_result.extend(_fetch_etf_holdings("SPUS"))

    with (
        patch("brain_api.core.prices.yf.download", side_effect=_download),
        patch("brain_api.core.prices.yf.Ticker") as price_ticker,
        patch("brain_api.universe.halal.yf.Ticker", side_effect=_ticker),
    ):
        price_ticker.return_value.history.side_effect = AssertionError(
            "Ticker.history must not run when batch download succeeds"
        )
        threads = [
            threading.Thread(
                target=lambda: load_prices_yfinance(
                    ["AAPL"], date(2026, 4, 1), date(2026, 4, 25)
                )
            ),
            threading.Thread(target=_fetch_holdings),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5.0)
            assert not thread.is_alive()

    assert ticker_calls == ["SPUS"]
    assert holdings_result[0]["symbol"] == "AAPL"
    assert max_active == 1


def test_load_prices_releases_lock_when_download_raises() -> None:
    ticker = MagicMock()
    ticker.history.return_value = _ohlcv_frame()
    with (
        patch(
            "brain_api.core.prices.yf.download", side_effect=RuntimeError("yahoo down")
        ),
        patch("brain_api.core.prices.yf.Ticker", return_value=ticker),
    ):
        first = load_prices_yfinance(["AAPL"], date(2026, 4, 1), date(2026, 4, 25))
        second = load_prices_yfinance(["AAPL"], date(2026, 4, 1), date(2026, 4, 25))
    assert "AAPL" in first
    assert "AAPL" in second

"""Yahoo OHLCV parse: MultiIndex (yfinance 1.0) and flat single-ticker frames."""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd

from brain_api.core.prices import load_prices_yfinance, repair_ohlc_envelope

_OHLCV_VALUES = {
    "Open": [10.0, 11.0],
    "High": [11.0, 12.0],
    "Low": [9.0, 10.0],
    "Close": [10.5, 11.5],
    "Volume": [100.0, 110.0],
}


def _ohlcv_index() -> pd.DatetimeIndex:
    return pd.to_datetime(["2026-04-23", "2026-04-24"])


def _flat_ohlcv_frame() -> pd.DataFrame:
    return pd.DataFrame(_OHLCV_VALUES, index=_ohlcv_index())


def _ticker_grouped_ohlcv(symbols: list[str]) -> pd.DataFrame:
    """yfinance 1.0 `group_by='ticker'` MultiIndex even for one ticker."""
    columns = pd.MultiIndex.from_tuples(
        [(symbol, price) for symbol in symbols for price in _OHLCV_VALUES],
        names=["Ticker", "Price"],
    )
    data = {
        (symbol, price): values
        for symbol in symbols
        for price, values in _OHLCV_VALUES.items()
    }
    return pd.DataFrame(data, index=_ohlcv_index(), columns=columns)


def _assert_history_must_not_run(ticker_cls) -> None:
    ticker_cls.return_value.history.side_effect = AssertionError(
        "Ticker.history must not run when yahoo download parses"
    )


def test_load_prices_parses_single_ticker_yahoo_multiindex() -> None:
    with (
        patch(
            "brain_api.core.prices.yf.download",
            return_value=_ticker_grouped_ohlcv(["AAPL"]),
        ),
        patch("brain_api.core.prices.yf.Ticker") as ticker_cls,
    ):
        _assert_history_must_not_run(ticker_cls)
        result = load_prices_yfinance(["AAPL"], date(2026, 4, 1), date(2026, 4, 25))

    assert "AAPL" in result
    assert list(result["AAPL"].columns) == ["open", "high", "low", "close", "volume"]
    assert len(result["AAPL"]) == 2


def test_load_prices_parses_multi_ticker_yahoo_multiindex() -> None:
    with (
        patch(
            "brain_api.core.prices.yf.download",
            return_value=_ticker_grouped_ohlcv(["AAPL", "MSFT"]),
        ),
        patch("brain_api.core.prices.yf.Ticker") as ticker_cls,
    ):
        _assert_history_must_not_run(ticker_cls)
        result = load_prices_yfinance(
            ["AAPL", "MSFT"], date(2026, 4, 1), date(2026, 4, 25)
        )

    assert "AAPL" in result
    assert "MSFT" in result
    assert list(result["AAPL"].columns) == ["open", "high", "low", "close", "volume"]
    assert list(result["MSFT"].columns) == ["open", "high", "low", "close", "volume"]


def test_load_prices_still_parses_flat_single_ticker_columns() -> None:
    with (
        patch("brain_api.core.prices.yf.download", return_value=_flat_ohlcv_frame()),
        patch("brain_api.core.prices.yf.Ticker") as ticker_cls,
    ):
        _assert_history_must_not_run(ticker_cls)
        result = load_prices_yfinance(["AAPL"], date(2026, 4, 1), date(2026, 4, 25))

    assert "AAPL" in result
    assert list(result["AAPL"].columns) == ["open", "high", "low", "close", "volume"]
    assert len(result["AAPL"]) == 2


def test_load_prices_repairs_yahoo_ohlc_envelope_without_mutating_download() -> None:
    downloaded = _flat_ohlcv_frame()
    downloaded.loc[downloaded.index[1], "Low"] = 11.25
    downloaded.loc[downloaded.index[1], "High"] = 11.25
    original = downloaded.copy(deep=True)

    with (
        patch("brain_api.core.prices.yf.download", return_value=downloaded),
        patch("brain_api.core.prices.yf.Ticker") as ticker_cls,
    ):
        _assert_history_must_not_run(ticker_cls)
        result = load_prices_yfinance(["AAPL"], date(2026, 4, 1), date(2026, 4, 25))

    repaired = result["AAPL"].iloc[1]
    assert repaired["low"] == 11.0
    assert repaired["high"] == 11.5
    pd.testing.assert_frame_equal(downloaded, original)


def test_repair_ohlc_envelope_does_not_impute_nonfinite_or_nonpositive_rows() -> None:
    frame = pd.DataFrame(
        {
            "open": [10.0, 0.0],
            "high": [np.nan, -1.0],
            "low": [11.0, 10.0],
            "close": [12.0, 10.0],
        }
    )

    repaired = repair_ohlc_envelope(frame)

    assert np.isnan(repaired.loc[0, "high"])
    assert repaired.loc[0, "low"] == 11.0
    assert repaired.loc[1, "high"] == -1.0
    assert repaired.loc[1, "low"] == 10.0

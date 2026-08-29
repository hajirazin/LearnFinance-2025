"""Aligned SPY/VIX session history for causal HMM fitting."""

from datetime import date

import numpy as np
import pandas as pd

from brain_api.core.sac.market_sessions import (
    align_to_xnys_sessions,
    require_exact_session_dates,
    xnys_session_dates,
)


def extract_aligned_market_history(
    prices: dict[str, pd.DataFrame],
    *,
    start_date: date,
    completed_through: date,
) -> tuple[list[date], np.ndarray, np.ndarray]:
    """Return identical-date, finite positive SPY/VIX close arrays."""
    missing = sorted({"SPY", "^VIX"} - set(prices))
    if missing:
        raise ValueError(f"SAC v3 training is missing market histories: {missing}")

    def _series(symbol: str) -> pd.Series:
        frame = prices[symbol]
        if frame is None or frame.empty or "close" not in frame:
            raise ValueError(f"SAC v3 training has no close history for {symbol}")
        series = frame["close"].copy()
        index = (
            series.index.tz_localize(None)
            if series.index.tz is not None
            else series.index
        )
        series.index = index.normalize()
        if series.index.has_duplicates or not series.index.is_monotonic_increasing:
            raise ValueError(
                f"SAC v3 {symbol} market history must be unique and ordered"
            )
        return series

    expected_dates = xnys_session_dates(start_date, completed_through)
    expected_index = pd.DatetimeIndex(expected_dates)

    def _align_to_expected_sessions(symbol: str) -> pd.Series:
        series = _series(symbol)
        aligned, actual_dates = align_to_xnys_sessions(series, expected_dates)
        require_exact_session_dates(
            actual_dates,
            expected_dates,
            context=f"SAC v3 training {symbol} market history",
        )
        return aligned.reindex(expected_index).astype(float)

    spy_values = _align_to_expected_sessions("SPY").to_numpy(dtype=float)
    vix_values = _align_to_expected_sessions("^VIX").to_numpy(dtype=float)
    if (
        len(spy_values) < 21
        or not np.all(np.isfinite(spy_values))
        or not np.all(np.isfinite(vix_values))
        or np.any(spy_values <= 0)
        or np.any(vix_values <= 0)
    ):
        raise ValueError(
            "SAC v3 SPY/VIX histories require at least 21 finite positive rows"
        )
    return expected_dates, spy_values, vix_values

"""Causal MAD clipping of price-input returns; realized labels stay separate."""

import warnings

import numpy as np
import pandas as pd

WINDOW = 252
MINIMUM = 60
MULTIPLIER = 6.0
NORMAL_SCALE = 1.4826


def clip_returns(logs, *, window=WINDOW, minimum=MINIMUM, multiplier=MULTIPLIER):
    """Use exact median absolute deviation within the preceding raw window.

    Missing observations, warmup rows, and zero-MAD windows are unchanged.
    Current and future observations cannot affect a row's clipping bounds.
    """
    if not 1 <= minimum <= window or multiplier <= 0:
        raise ValueError("Invalid rolling MAD policy")
    if not logs.index.is_unique or not logs.index.is_monotonic_increasing:
        raise ValueError("Returns must have unique chronological rows")
    result, records = logs.copy(), []
    for symbol in logs:
        values = logs[symbol].to_numpy(dtype=float)
        previous = np.pad(values, (window, 0), constant_values=np.nan)
        history = np.lib.stride_tricks.sliding_window_view(previous, window)[:-1]
        count = np.isfinite(history).sum(axis=1)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            median = np.nanmedian(history, axis=1)
            mad = np.nanmedian(np.abs(history - median[:, None]), axis=1)
        radius = multiplier * NORMAL_SCALE * mad
        usable = (count >= minimum) & np.isfinite(values) & (radius > 0)
        bounded = np.clip(values, median - radius, median + radius)
        changed = usable & (values != bounded)
        result.loc[changed, symbol] = bounded[changed]
        for row in np.flatnonzero(changed):
            records.append(
                {
                    "session": logs.index[row],
                    "symbol": symbol,
                    "raw_log_return": values[row],
                    "clipped_log_return": bounded[row],
                    "prior_median": median[row],
                    "prior_mad": mad[row],
                    "prior_observations": int(count[row]),
                    "lower": median[row] - radius[row],
                    "upper": median[row] + radius[row],
                }
            )
    return result, pd.DataFrame(
        records,
        columns=[
            "session",
            "symbol",
            "raw_log_return",
            "clipped_log_return",
            "prior_median",
            "prior_mad",
            "prior_observations",
            "lower",
            "upper",
        ],
    )


def reconstruct_close(close, raw_logs, clipped_logs):
    """Accumulate only observed clipping corrections; never fill source prices."""
    if not close.index.equals(raw_logs.index) or not close.columns.equals(
        raw_logs.columns
    ):
        raise ValueError("Price and return axes differ")
    if not raw_logs.index.equals(clipped_logs.index) or not raw_logs.columns.equals(
        clipped_logs.columns
    ):
        raise ValueError("Raw and clipped return axes differ")
    if not raw_logs.isna().equals(clipped_logs.isna()):
        raise ValueError("Clipping must preserve missing returns")
    correction = (clipped_logs - raw_logs).fillna(0.0).cumsum()
    return close * np.exp(correction)


def price_features(features, close, raw_logs, clipped_logs, *, index=None):
    """Replace the four price-derived channels; news and volume stay identical."""
    result = features.copy()
    for slot, values in (
        (0, np.expm1(clipped_logs.rolling(20).sum())),
        (1, np.expm1(clipped_logs.rolling(231).sum().shift(21))),
        (2, clipped_logs.rolling(20).std(ddof=1) * np.sqrt(252)),
        (5, reconstruct_close(close, raw_logs, clipped_logs)),
    ):
        result[..., slot] = values.reindex(index=index).to_numpy(dtype=np.float32)
    return result

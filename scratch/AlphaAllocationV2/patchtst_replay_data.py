"""Assemble common stock-Friday inputs for the three-model replay."""

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd
from build_friday_dataset import (
    ACTION_DATES,
    OUTPUT,
    audit_candidates,
    join_friday_ohlcv,
    load_prices,
    verify_prices,
)
from compare_feature_sets import EVAL_END, EVAL_START, SEVEN, TOP_K, TRAIN_YEARS

YEAR_DAYS = 365
WEEKLY_CONTEXT = 12
DAILY_HORIZON = 5
DAY_PATCH = 10
DAY_STRIDE = 5


def comparable_fridays(sessions, fridays):
    """Keep anchors where the fifth following session is the next Friday."""
    ends = fridays + pd.Timedelta(days=7)
    counts = sessions.searchsorted(ends, side="right") - sessions.searchsorted(
        fridays, side="right"
    )
    return fridays[ends.isin(fridays) & (counts == DAILY_HORIZON)]


def weekly_windows(values, length):
    """Build chronological windows on the actual trading Friday grid."""
    result = np.full(
        (*values.shape[:2], length, values.shape[-1]), np.nan, dtype=np.float32
    )
    windows = np.lib.stride_tricks.sliding_window_view(values, length, axis=0)
    result[length - 1 :] = windows.swapaxes(-1, -2)
    return result


def close_windows(logs, fridays, width):
    """Left pad only calendar-length differences, retaining every observed session."""
    positions = logs.index.get_indexer(fridays)
    if (positions < 0).any() or (positions + DAILY_HORIZON >= len(logs)).any():
        raise ValueError("Close window falls outside the session grid")
    starts = logs.index.searchsorted(
        fridays - pd.Timedelta(days=YEAR_DAYS), side="right"
    )
    counts = positions - starts + 1
    if (counts > width).any() or (counts <= 0).any():
        raise ValueError("365-day window does not fit the configured context")
    daily_x = np.full((len(fridays), logs.shape[1], width), np.nan, dtype=np.float32)
    daily_y = np.empty((len(fridays), logs.shape[1], DAILY_HORIZON), dtype=np.float32)
    mask = np.arange(width) >= (width - counts[:, None])
    for i, (start, end, count) in enumerate(
        zip(starts, positions, counts, strict=True)
    ):
        daily_x[i, :, -count:] = logs.iloc[start : end + 1].to_numpy().T
        daily_y[i] = logs.iloc[end + 1 : end + 1 + DAILY_HORIZON].to_numpy().T
    return daily_x, mask, daily_y


@dataclass(frozen=True)
class ReplayData:
    labels: pd.DataFrame
    weekly_x: np.ndarray
    daily_x: np.ndarray
    daily_mask: np.ndarray
    daily_y: np.ndarray
    coverage: pd.DataFrame
    evaluation_dates: pd.DatetimeIndex


def load_replay():
    with duckdb.connect() as db:
        panel = db.read_parquet(str(OUTPUT / "friday_dataset.parquet")).df()
    panel = join_friday_ohlcv(panel.set_index(["friday", "symbol"]).sort_index())
    calendar, close, _volume = load_prices()
    fridays = pd.DatetimeIndex(panel.index.unique("friday"))
    symbols = pd.Index(close.columns)
    grid = pd.MultiIndex.from_product([fridays, symbols], names=["friday", "symbol"])
    panel = panel.loc[grid]

    # Apply the audited price adjustments used to build the Friday panel.
    _, audit, daily_audit = audit_candidates(
        close, pd.date_range(fridays[0], fridays[-1], freq="W-FRI")
    )
    adjusted, _ = verify_prices(audit, daily_audit, close)
    logs = np.log(adjusted).diff()
    for session, symbol in daily_audit.index[
        daily_audit["status"].str.startswith("unresolved")
    ]:
        logs.loc[session, symbol] = np.nan
    for symbol, day in ACTION_DATES:
        logs.loc[day, symbol] = np.nan

    matched_dates = comparable_fridays(
        calendar.sessions_in_range(close.index[0], close.index[-1]), fridays
    )
    evaluation = matched_dates[
        matched_dates.to_series().between(EVAL_START, EVAL_END).to_numpy()
    ]
    # Data before the first rolling fit only supplies trailing feature context.
    dates = matched_dates[
        matched_dates >= evaluation[0] - pd.DateOffset(years=TRAIN_YEARS)
    ]
    positions = fridays.get_indexer(dates)
    start = positions[0] - WEEKLY_CONTEXT + 1
    values = (
        panel[list(SEVEN)].to_numpy().reshape(len(fridays), len(symbols), len(SEVEN))
    )
    weekly_x = weekly_windows(values[start:], WEEKLY_CONTEXT)[positions - start]

    # The model expects a rectangular input. Up to one patch's worth of left
    # padding accounts for the variable number of sessions in 365 calendar days.
    session_counts = logs.index.searchsorted(
        dates, side="right"
    ) - logs.index.searchsorted(dates - pd.Timedelta(days=YEAR_DAYS), side="right")
    width = (
        DAY_PATCH
        + int(np.ceil((session_counts.max() - DAY_PATCH) / DAY_STRIDE)) * DAY_STRIDE
    )
    if width - session_counts.min() >= DAY_PATCH:
        raise ValueError("A PatchTST context would contain a padding-only patch")
    daily_x, daily_mask, daily_y = close_windows(logs, dates, width)

    labels = panel.loc[
        dates, ["next_friday", "forward_return", "price_usable_at_cutoff"]
    ].copy()
    cutoff_valid = (
        labels["price_usable_at_cutoff"].to_numpy().reshape(len(dates), len(symbols))
    )
    snapshot = np.isfinite(weekly_x[:, :, -1]).all(axis=-1) & cutoff_valid
    weekly_valid = np.isfinite(weekly_x).all(axis=(-1, -2)) & cutoff_valid
    daily_valid = (np.isfinite(daily_x) | ~daily_mask[:, None, :]).all(axis=-1)
    common = weekly_valid & daily_valid
    coverage = pd.DataFrame(
        {
            "latest_seven_features": snapshot.sum(axis=1),
            "twelve_fridays": weekly_valid.sum(axis=1),
            "common_candidates": common.sum(axis=1),
            "daily_context_sessions": session_counts,
        },
        index=dates,
    )
    if coverage.loc[evaluation, "common_candidates"].lt(TOP_K).any():
        raise ValueError("A 2025 Friday has fewer than 15 common candidates")
    date_rows, _ = np.nonzero(common)
    daily_mask = daily_mask[date_rows]
    daily_x = daily_x[common]
    # Mean-filled padding becomes zero after RevIN; the mask keeps these values
    # out of the mean/variance. All actual sessions must have valid returns.
    row_means = np.nansum(daily_x, axis=1) / daily_mask.sum(axis=1)
    daily_x = np.where(daily_mask, daily_x, row_means[:, None]).astype(np.float32)
    labels = labels.loc[common.ravel(), ["next_friday", "forward_return"]]
    daily_y = daily_y[common]
    comparable_labels = labels["forward_return"].notna() & np.isfinite(daily_y).all(
        axis=1
    )
    np.testing.assert_allclose(
        np.expm1(daily_y[comparable_labels].sum(axis=1, dtype=np.float64)),
        labels.loc[comparable_labels, "forward_return"],
        rtol=1e-5,
        atol=1e-7,
    )
    return ReplayData(
        labels=labels,
        weekly_x=weekly_x[common],
        daily_x=daily_x[:, :, None],
        daily_mask=daily_mask[:, :, None],
        daily_y=daily_y,
        coverage=coverage,
        evaluation_dates=evaluation,
    )


def monthly_folds(data):
    """Use identical historical stock-weeks and strict completion cutoffs."""
    fridays = data.labels.index.get_level_values("friday")
    ends = data.labels["next_friday"]
    observed = data.labels["forward_return"].notna() & np.isfinite(data.daily_y).all(
        axis=1
    )
    for month in data.evaluation_dates.to_period("M").unique():
        dates = data.evaluation_dates[data.evaluation_dates.to_period("M") == month]
        cutoff = dates[0]
        start = cutoff - pd.DateOffset(years=TRAIN_YEARS)
        historical = observed & (fridays >= start) & ends.lt(cutoff)
        historical_dates = fridays[historical].unique()
        if len(historical_dates) < 5:
            raise ValueError(f"Insufficient training dates for {cutoff.date()}")
        validation_start = historical_dates[int(len(historical_dates) * 0.8)]
        train = historical & ends.lt(validation_start)
        validation = historical & (fridays >= validation_start)
        yield (
            cutoff,
            historical.to_numpy(),
            train.to_numpy(),
            validation.to_numpy(),
            fridays.isin(dates),
        )

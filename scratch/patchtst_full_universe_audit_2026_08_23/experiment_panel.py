"""Exact-session dynamic panel construction for PatchTST research."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from experiment_spec import CLOSE_INDEX, FEATURE_NAMES


@dataclass(frozen=True)
class SplitWindow:
    start: date
    end: date


def validate_splits(splits: dict[str, SplitWindow]) -> None:
    if list(splits) != ["train", "validation", "test"]:
        raise ValueError("splits must be ordered train, validation, test")
    values = list(splits.values())
    if any(window.start > window.end for window in values):
        raise ValueError("a split starts after it ends")
    if not values[0].end < values[1].start < values[1].end < values[2].start:
        raise ValueError("splits overlap or are not chronological")
    if (values[1].start - values[0].end).days < 14:
        raise ValueError("train/validation embargo is shorter than two weeks")
    if (values[2].start - values[1].end).days < 14:
        raise ValueError("validation/test embargo is shorter than two weeks")


def _split_for(day: date, splits: dict[str, SplitWindow]) -> str | None:
    for name, window in splits.items():
        if window.start <= day <= window.end:
            return name
    return None


def _normalized_prices(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["open", "high", "low", "close", "volume"]
    if any(name not in frame.columns for name in required):
        raise ValueError("OHLCV columns are missing")
    clean = frame[required].astype(float).copy()
    index = pd.DatetimeIndex(pd.to_datetime(clean.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    clean.index = index.normalize()
    clean = clean.sort_index()
    if not clean.index.is_unique:
        raise ValueError("OHLCV session index contains duplicates")
    return clean


def _window_arrays(
    frame: pd.DataFrame,
    required_sessions: pd.DatetimeIndex,
    context_length: int,
    prediction_length: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not required_sessions.isin(frame.index).all():
        return None
    values = frame.loc[required_sessions].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or not (values > 0).all():
        return None
    returns = np.diff(np.log(values), axis=0).astype(np.float32)
    x = returns[:context_length]
    y = returns[context_length : context_length + prediction_length]
    if x.shape != (context_length, 5) or y.shape != (prediction_length, 5):
        raise AssertionError("unexpected PatchTST window shape")
    return x, y


def build_weekly_panel(
    prices: dict[str, pd.DataFrame],
    *,
    sessions: pd.DatetimeIndex,
    splits: dict[str, SplitWindow],
    include_test_labels: bool,
    context_length: int,
    prediction_length: int,
) -> pd.DataFrame:
    """Build an unbalanced panel using exact exchange sessions.

    A symbol-week is admitted only when it has every required raw price bar:
    one pre-context bar, all context sessions, and all target sessions. Test
    target values are not read until ``include_test_labels`` is true.
    """
    validate_splits(splits)
    if not prices:
        raise RuntimeError("price dictionary is empty")
    exact_sessions = (
        pd.DatetimeIndex(pd.to_datetime(sessions)).tz_localize(None).normalize()
    )
    exact_sessions = exact_sessions.sort_values().unique()
    normalized = {symbol: _normalized_prices(frame) for symbol, frame in prices.items()}
    start = min(window.start for window in splits.values())
    end = max(window.end for window in splits.values())
    decisions = pd.date_range(start, end, freq="W-MON")
    rows: list[dict[str, Any]] = []
    exclusions: dict[str, int] = {}
    per_symbol: dict[str, dict[str, int]] = {
        symbol: {"eligible": 0, "excluded": 0} for symbol in sorted(normalized)
    }

    for decision_ts in decisions:
        decision = decision_ts.date()
        split = _split_for(decision, splits)
        if split is None:
            continue
        insertion = int(exact_sessions.searchsorted(decision_ts, side="left"))
        past_start = insertion - context_length - 1
        future_end = insertion + prediction_length
        if past_start < 0 or future_end > len(exact_sessions):
            exclusions["calendar_boundary"] = exclusions.get(
                "calendar_boundary", 0
            ) + len(normalized)
            continue
        required = exact_sessions[past_start:future_end]
        context_end = exact_sessions[insertion - 1]
        target_sessions = exact_sessions[insertion:future_end]
        next_windows = [window for window in splits.values() if window.start > decision]
        if next_windows and target_sessions[-1].date() >= min(
            w.start for w in next_windows
        ):
            exclusions["target_crosses_next_split"] = exclusions.get(
                "target_crosses_next_split", 0
            ) + len(normalized)
            continue

        for symbol, frame in normalized.items():
            missing_exact = not required.isin(frame.index).all()
            arrays = (
                None
                if missing_exact
                else _window_arrays(frame, required, context_length, prediction_length)
            )
            if arrays is None:
                reason = (
                    "missing_exact_session"
                    if missing_exact
                    else "nonpositive_or_nonfinite"
                )
                exclusions[reason] = exclusions.get(reason, 0) + 1
                per_symbol[symbol]["excluded"] += 1
                continue
            x, y_all = arrays
            close = x[:, CLOSE_INDEX]
            locked = split == "test" and not include_test_labels
            y: np.ndarray | None = None if locked else y_all
            actual = np.nan if locked else float(y_all[:, CLOSE_INDEX].sum())
            rows.append(
                {
                    "decision_date": decision,
                    "split": split,
                    "symbol": symbol,
                    "context_end": context_end.date(),
                    "target_end": target_sessions[-1].date(),
                    "x": x,
                    "y": y,
                    "actual_weekly_log_return": actual,
                    "past_week_log_return": float(close[-5:].sum()),
                    "momentum_4w_log_return": float(close[-20:].sum()),
                    "context_log_return": float(close.sum()),
                    "volatility_4w": float(close[-20:].std(ddof=1)),
                    "volume_volatility_4w": float(x[-20:, 4].std(ddof=1)),
                }
            )
            per_symbol[symbol]["eligible"] += 1

    panel = pd.DataFrame(rows)
    if panel.empty:
        raise RuntimeError(f"no panel rows; exclusions={exclusions}")
    panel = panel.sort_values(["decision_date", "symbol"]).reset_index(drop=True)
    panel.attrs["exclusion_counts"] = exclusions
    panel.attrs["per_symbol_counts"] = per_symbol
    panel.attrs["feature_names"] = FEATURE_NAMES
    return panel


def panel_arrays(
    panel: pd.DataFrame, split: str
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rows = panel[panel["split"] == split].copy()
    if rows.empty:
        raise RuntimeError(f"{split} split is empty")
    if rows["y"].isna().any():
        raise RuntimeError(f"{split} labels are locked")
    x = np.stack(rows["x"].to_list()).astype(np.float32)
    y = np.stack(rows["y"].to_list()).astype(np.float32)
    metadata = rows.drop(columns=["x", "y"]).reset_index(drop=True)
    return x, y, metadata

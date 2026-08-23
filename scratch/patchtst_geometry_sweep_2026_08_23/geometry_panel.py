"""Exact-XNYS close-only panel construction with confirmatory label locking."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from geometry_spec import CONTEXT_LENGTH, PREDICTION_LENGTH, EvaluationFold


def validate_fold(fold: EvaluationFold) -> None:
    """Reject overlapping or insufficiently embargoed decision windows."""
    windows = [fold.train, fold.validation, fold.evaluation]
    if any(window.start > window.end for window in windows):
        raise ValueError("a fold window starts after it ends")
    if not (
        fold.train.end < fold.validation.start
        and fold.validation.end < fold.evaluation.start
    ):
        raise ValueError("fold windows overlap or are not chronological")
    if (fold.validation.start - fold.train.end).days < 14:
        raise ValueError("train/validation embargo is shorter than two weeks")
    if (fold.evaluation.start - fold.validation.end).days < 14:
        raise ValueError("validation/evaluation embargo is shorter than two weeks")


def _split_for(day: date, fold: EvaluationFold) -> str | None:
    for name, window in (
        ("train", fold.train),
        ("validation", fold.validation),
        ("evaluation", fold.evaluation),
    ):
        if window.start <= day <= window.end:
            return name
    return None


def _normalized_close(frame: pd.DataFrame) -> pd.Series:
    if "close" not in frame.columns:
        raise ValueError("adjusted close column is missing")
    close = frame["close"].astype(float).copy()
    index = pd.DatetimeIndex(pd.to_datetime(close.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    close.index = index.normalize()
    close = close.sort_index()
    if not close.index.is_unique:
        raise ValueError("close session index contains duplicates")
    return close


def _window_arrays(
    close: pd.Series,
    required_sessions: pd.DatetimeIndex,
    *,
    labels_locked: bool,
) -> tuple[np.ndarray, np.ndarray | None] | None:
    context_sessions = required_sessions[: CONTEXT_LENGTH + 1]
    context_prices = close.loc[context_sessions].to_numpy(dtype=np.float64)
    if not np.isfinite(context_prices).all() or not (context_prices > 0).all():
        return None
    x = np.diff(np.log(context_prices)).astype(np.float32).reshape(-1, 1)
    if x.shape != (CONTEXT_LENGTH, 1):
        raise AssertionError("unexpected PatchTST context shape")
    if labels_locked:
        return x, None
    all_prices = close.loc[required_sessions].to_numpy(dtype=np.float64)
    if not np.isfinite(all_prices).all() or not (all_prices > 0).all():
        return None
    all_returns = np.diff(np.log(all_prices)).astype(np.float32)
    y = all_returns[CONTEXT_LENGTH:].reshape(-1, 1)
    if y.shape != (PREDICTION_LENGTH, 1):
        raise AssertionError("unexpected PatchTST target shape")
    return x, y


def build_fold_panel(
    prices: dict[str, pd.DataFrame],
    *,
    sessions: pd.DatetimeIndex,
    fold: EvaluationFold,
    include_evaluation_labels: bool,
) -> pd.DataFrame:
    """Build one dynamic panel without reading locked evaluation target values."""
    validate_fold(fold)
    if not prices:
        raise RuntimeError("price dictionary is empty")
    exact_sessions = pd.DatetimeIndex(pd.to_datetime(sessions))
    if exact_sessions.tz is not None:
        exact_sessions = exact_sessions.tz_localize(None)
    exact_sessions = exact_sessions.normalize().sort_values().unique()
    normalized = {
        symbol: _normalized_close(frame) for symbol, frame in sorted(prices.items())
    }
    decisions = pd.date_range(fold.train.start, fold.evaluation.end, freq="W-MON")
    rows: list[dict[str, Any]] = []
    exclusions: dict[str, int] = {}
    per_symbol = {symbol: {"eligible": 0, "excluded": 0} for symbol in normalized}

    for decision_ts in decisions:
        decision = decision_ts.date()
        split = _split_for(decision, fold)
        if split is None:
            continue
        insertion = int(exact_sessions.searchsorted(decision_ts, side="left"))
        past_start = insertion - CONTEXT_LENGTH - 1
        future_end = insertion + PREDICTION_LENGTH
        if past_start < 0 or future_end > len(exact_sessions):
            exclusions["calendar_boundary"] = exclusions.get(
                "calendar_boundary", 0
            ) + len(normalized)
            continue
        required = exact_sessions[past_start:future_end]
        target_sessions = exact_sessions[insertion:future_end]
        later_starts = [
            window.start
            for window in (fold.validation, fold.evaluation)
            if window.start > decision
        ]
        if later_starts and target_sessions[-1].date() >= min(later_starts):
            exclusions["target_crosses_next_split"] = exclusions.get(
                "target_crosses_next_split", 0
            ) + len(normalized)
            continue

        for symbol, close in normalized.items():
            if not required.isin(close.index).all():
                exclusions["missing_exact_session"] = (
                    exclusions.get("missing_exact_session", 0) + 1
                )
                per_symbol[symbol]["excluded"] += 1
                continue
            labels_locked = split == "evaluation" and not include_evaluation_labels
            arrays = _window_arrays(close, required, labels_locked=labels_locked)
            if arrays is None:
                exclusions["nonpositive_or_nonfinite_close"] = (
                    exclusions.get("nonpositive_or_nonfinite_close", 0) + 1
                )
                per_symbol[symbol]["excluded"] += 1
                continue
            x, y = arrays
            close_returns = x[:, 0]
            rows.append(
                {
                    "fold": fold.name,
                    "evidence_kind": fold.evidence_kind,
                    "decision_date": decision,
                    "split": split,
                    "symbol": symbol,
                    "context_end": exact_sessions[insertion - 1].date(),
                    "target_end": target_sessions[-1].date(),
                    "x": x,
                    "y": y,
                    "actual_weekly_log_return": (
                        np.nan if y is None else float(y[:, 0].sum())
                    ),
                    "past_week_log_return": float(close_returns[-5:].sum()),
                    "momentum_4w_log_return": float(close_returns[-20:].sum()),
                    "context_log_return": float(close_returns.sum()),
                    "volatility_4w": float(close_returns[-20:].std(ddof=1)),
                }
            )
            per_symbol[symbol]["eligible"] += 1

    panel = pd.DataFrame(rows)
    if panel.empty:
        raise RuntimeError(f"no panel rows; exclusions={exclusions}")
    panel = panel.sort_values(["decision_date", "symbol"]).reset_index(drop=True)
    panel.attrs["exclusion_counts"] = exclusions
    panel.attrs["per_symbol_counts"] = per_symbol
    panel.attrs["feature_names"] = ["close_log_return"]
    panel.attrs["evaluation_labels_included"] = include_evaluation_labels
    return panel


def panel_arrays(
    panel: pd.DataFrame, split: str
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Materialize one split and reject any locked target rows."""
    rows = panel[panel["split"] == split].copy()
    if rows.empty:
        raise RuntimeError(f"{split} split is empty")
    if rows["y"].isna().any():
        raise RuntimeError(f"{split} labels are locked")
    x = np.stack(rows["x"].to_list()).astype(np.float32)
    y = np.stack(rows["y"].to_list()).astype(np.float32)
    metadata = rows.drop(columns=["x", "y"]).reset_index(drop=True)
    return x, y, metadata


def panel_identity(panel: pd.DataFrame, *, include_evaluation: bool) -> str:
    """Hash stable sample keys without target values."""
    from geometry_spec import sha256_json

    rows = panel if include_evaluation else panel[panel["split"] != "evaluation"]
    identity = rows[
        ["fold", "decision_date", "split", "symbol", "context_end", "target_end"]
    ]
    return sha256_json(identity.astype(str).to_dict(orient="records"))

"""Validation weekly rank IC for PatchTST checkpoint selection."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mean_weekly_rank_ic(
    decision_dates: np.ndarray,
    symbols: np.ndarray,
    predicted_weekly_logs: np.ndarray,
    actual_weekly_logs: np.ndarray,
) -> float:
    """Mean per-week Spearman rank IC of predicted vs actual weekly log returns.

    Each row is one symbol-week. Weeks with fewer than three symbols or with
    degenerate predicted ranks (population std < 1e-12) are skipped. Average
    ranks break ties. Raises ``FloatingPointError`` when no finite weekly IC
    exists; never substitutes zero.
    """
    if not (
        len(decision_dates)
        == len(symbols)
        == len(predicted_weekly_logs)
        == len(actual_weekly_logs)
    ):
        raise ValueError("rank IC inputs must have equal length")

    frame = pd.DataFrame(
        {
            "decision_date": decision_dates,
            "symbol": symbols,
            "actual_weekly_log_return": np.asarray(
                actual_weekly_logs, dtype=np.float64
            ),
            "predicted": np.asarray(predicted_weekly_logs, dtype=np.float64),
        }
    )
    values: list[float] = []
    for _, week in frame.groupby("decision_date", sort=True):
        if len(week) < 3 or week["predicted"].std(ddof=0) < 1e-12:
            continue
        values.append(
            float(
                week["predicted"]
                .rank(method="average")
                .corr(week["actual_weekly_log_return"].rank(method="average"))
            )
        )
    if not values or not np.isfinite(values).all():
        raise FloatingPointError("validation weekly rank IC is not finite")
    return float(np.mean(values))


def checkpoint_is_better(
    rank_ic: float,
    val_mse: float,
    best_rank_ic: float,
    best_val_mse: float,
) -> bool:
    """True when this epoch should replace the stored checkpoint.

    Prefer a strictly higher validation weekly rank IC. Equal IC (np.isclose)
    uses lower close-only validation MSE as the tie-break.
    """
    if rank_ic > best_rank_ic + 1e-12:
        return True
    return bool(np.isclose(rank_ic, best_rank_ic) and val_mse < best_val_mse)

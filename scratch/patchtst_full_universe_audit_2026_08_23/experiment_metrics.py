"""Cross-sectional metrics and week-block uncertainty for PatchTST research."""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd


def _rank_ic(predicted: np.ndarray, actual: np.ndarray) -> float:
    if len(predicted) < 3 or np.std(predicted) < 1e-12 or np.std(actual) < 1e-12:
        return math.nan
    pred_rank = pd.Series(predicted).rank(method="average").to_numpy()
    actual_rank = pd.Series(actual).rank(method="average").to_numpy()
    return float(np.corrcoef(pred_rank, actual_rank)[0, 1])


def _has_cross_sectional_ranking(predicted: np.ndarray) -> bool:
    return len(predicted) >= 2 and float(np.std(predicted)) >= 1e-12


def _top_symbols(week: pd.DataFrame, top_k: int) -> set[str] | None:
    if not _has_cross_sectional_ranking(
        week["predicted_weekly_return"].to_numpy(float)
    ):
        return None
    ordered = week.sort_values(
        ["predicted_weekly_return", "symbol"], ascending=[False, True]
    )
    return set(ordered.head(top_k)["symbol"])


def weekly_metrics(frame: pd.DataFrame, top_k: int = 15) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for decision, week in frame.groupby("decision_date", sort=True):
        week = week.sort_values("symbol")
        if len(week) < 2 * top_k:
            raise ValueError(
                f"{decision} has {len(week)} symbols; need at least {2 * top_k}"
            )
        predicted = week["predicted_weekly_return"].to_numpy(float)
        actual = week["actual_weekly_return"].to_numpy(float)
        has_ranking = _has_cross_sectional_ranking(predicted)
        if has_ranking:
            order = np.lexsort((week["symbol"].to_numpy(), -predicted))
            top = actual[order[:top_k]]
            bottom = actual[order[-top_k:]]
        rows.append(
            {
                "decision_date": decision,
                "n_symbols": len(week),
                "rank_ic": _rank_ic(predicted, actual),
                "top15_excess": (
                    float(top.mean() - actual.mean()) if has_ranking else math.nan
                ),
                "top15_bottom15_spread": (
                    float(top.mean() - bottom.mean()) if has_ranking else math.nan
                ),
                "mae": float(np.mean(np.abs(predicted - actual))),
                "mse": float(np.mean((predicted - actual) ** 2)),
            }
        )
    return pd.DataFrame(rows)


def _selection_stability(frame: pd.DataFrame, top_k: int) -> tuple[float, float]:
    selections = [
        _top_symbols(week, top_k)
        for _, week in frame.groupby("decision_date", sort=True)
    ]
    valid_pairs = [
        (previous, current)
        for previous, current in pairwise(selections)
        if previous is not None and current is not None
    ]
    if not valid_pairs:
        return math.nan, math.nan
    overlaps = [len(previous & current) / top_k for previous, current in valid_pairs]
    turnovers = [len(current - previous) / top_k for previous, current in valid_pairs]
    return float(np.mean(overlaps)), float(np.mean(turnovers))


def aggregate_metrics(frame: pd.DataFrame, top_k: int = 15) -> dict[str, float | int]:
    predicted = frame["predicted_weekly_return"].to_numpy(float)
    actual = frame["actual_weekly_return"].to_numpy(float)
    weekly = weekly_metrics(frame, top_k=top_k)
    pred_up = predicted > 0
    actual_up = actual > 0
    positive = actual_up
    negative = ~actual_up
    balanced = (
        float(
            (
                (pred_up[positive] == actual_up[positive]).mean()
                + (pred_up[negative] == actual_up[negative]).mean()
            )
            / 2
        )
        if positive.any() and negative.any()
        else math.nan
    )
    overlap, turnover = _selection_stability(frame, top_k)
    return {
        "n_rows": len(frame),
        "n_weeks": int(frame["decision_date"].nunique()),
        "min_symbols_per_week": int(weekly["n_symbols"].min()),
        "median_symbols_per_week": float(weekly["n_symbols"].median()),
        "max_symbols_per_week": int(weekly["n_symbols"].max()),
        "mae": float(np.mean(np.abs(predicted - actual))),
        "rmse": float(np.sqrt(np.mean((predicted - actual) ** 2))),
        "direction_accuracy": float(np.mean(pred_up == actual_up)),
        "balanced_accuracy": balanced,
        "positive_prevalence": float(np.mean(actual_up)),
        "weekly_rank_ic": float(weekly["rank_ic"].mean()),
        "rank_ic_information_ratio": float(
            weekly["rank_ic"].mean() / weekly["rank_ic"].std(ddof=1)
        ),
        "top15_excess": float(weekly["top15_excess"].mean()),
        "top15_bottom15_spread": float(weekly["top15_bottom15_spread"].mean()),
        "top15_overlap": overlap,
        "top15_turnover": turnover,
    }


def _moving_block_indices(
    n_weeks: int, rng: np.random.Generator, block_weeks: int
) -> np.ndarray:
    if n_weeks < block_weeks:
        return rng.integers(0, n_weeks, size=n_weeks)
    starts = rng.integers(
        0, n_weeks - block_weeks + 1, size=math.ceil(n_weeks / block_weeks)
    )
    return np.concatenate([np.arange(start, start + block_weeks) for start in starts])[
        :n_weeks
    ]


def paired_block_bootstrap(
    challenger: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    seed: int,
    repetitions: int,
    block_weeks: int,
    top_k: int,
) -> dict[str, dict[str, float | list[float]]]:
    keys = ["decision_date", "symbol"]
    joined = challenger.merge(
        reference,
        on=keys,
        suffixes=("_challenger", "_reference"),
        validate="one_to_one",
    )
    if len(joined) != len(challenger) or len(joined) != len(reference):
        raise ValueError("paired comparison row keys differ")
    if not np.allclose(
        joined["actual_weekly_return_challenger"],
        joined["actual_weekly_return_reference"],
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("paired actual labels differ")
    left_weekly = weekly_metrics(challenger, top_k=top_k).set_index("decision_date")
    right_weekly = weekly_metrics(reference, top_k=top_k).set_index("decision_date")
    dates = left_weekly.index.intersection(right_weekly.index).sort_values()
    if len(dates) != len(left_weekly) or len(dates) != len(right_weekly):
        raise ValueError("paired comparison week keys differ")
    rng = np.random.default_rng(seed)
    indices = np.stack(
        [
            _moving_block_indices(len(dates), rng, block_weeks)
            for _ in range(repetitions)
        ]
    )
    metric_columns = ["rank_ic", "top15_excess", "top15_bottom15_spread", "mae"]
    output: dict[str, dict[str, float | list[float]]] = {}
    for column in metric_columns:
        differences = (
            left_weekly.loc[dates, column].to_numpy()
            - right_weekly.loc[dates, column].to_numpy()
        )
        sampled = differences[indices]
        counts = np.isfinite(sampled).sum(axis=1)
        samples = np.full(repetitions, np.nan, dtype=float)
        np.divide(
            np.nansum(sampled, axis=1),
            counts,
            out=samples,
            where=counts > 0,
        )
        finite_differences = differences[np.isfinite(differences)]
        finite_samples = samples[np.isfinite(samples)]
        delta = (
            float(finite_differences.mean()) if len(finite_differences) else math.nan
        )
        ci95 = (
            [
                float(np.quantile(finite_samples, 0.025)),
                float(np.quantile(finite_samples, 0.975)),
            ]
            if len(finite_samples)
            else [math.nan, math.nan]
        )
        name = "weekly_rank_ic" if column == "rank_ic" else column
        output[name] = {
            "delta": delta,
            "ci95": ci95,
        }
    return output

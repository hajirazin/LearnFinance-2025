#!/usr/bin/env python3
"""Metrics and week-block uncertainty for the corrected PatchTST suite."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return math.nan
    return float(np.corrcoef(pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy())[0, 1])


def _fractional_extreme_weights(values: np.ndarray, k: int, *, largest: bool) -> np.ndarray:
    """Return tie-neutral membership weights whose sum is exactly ``k``."""
    if not 0 < k <= len(values):
        raise ValueError("k must be within the cross-section")
    ordered = np.sort(values)
    threshold = ordered[-k] if largest else ordered[k - 1]
    strict = values > threshold if largest else values < threshold
    tied = values == threshold
    weights = strict.astype(float)
    remaining = k - int(strict.sum())
    if remaining:
        weights[tied] = remaining / int(tied.sum())
    return weights


def _expected_tie_aware_dcg(scores: np.ndarray, relevance: np.ndarray, k: int) -> float:
    """Expected DCG under uniform ordering within equal-score groups."""
    total = 0.0
    position = 0
    for score in np.sort(np.unique(scores))[::-1]:
        group = relevance[scores == score]
        slots = min(len(group), k - position)
        if slots <= 0:
            break
        discounts = 1.0 / np.log2(np.arange(position + 2, position + slots + 2))
        total += float(group.mean() * discounts.sum())
        position += slots
    return total


def _balanced_accuracy(pred: np.ndarray, actual: np.ndarray) -> float:
    mask = (pred != 0) & (actual != 0)
    if not mask.any():
        return math.nan
    pred_up, actual_up = pred[mask] > 0, actual[mask] > 0
    positive = actual_up
    negative = ~actual_up
    if not positive.any() or not negative.any():
        return math.nan
    return float(((pred_up[positive] == actual_up[positive]).mean() + (pred_up[negative] == actual_up[negative]).mean()) / 2)


def pesaran_timmermann(pred: np.ndarray, actual: np.ndarray) -> float:
    mask = (pred != 0) & (actual != 0)
    pred_up, actual_up = pred[mask] > 0, actual[mask] > 0
    n = len(pred_up)
    if n < 3:
        return math.nan
    py = float(actual_up.mean())
    px = float(pred_up.mean())
    p = float((pred_up == actual_up).mean())
    p_star = py * px + (1 - py) * (1 - px)
    variance_p = (p_star * (1 - p_star)) / n
    variance_star = (
        ((2 * py - 1) ** 2 * px * (1 - px) + (2 * px - 1) ** 2 * py * (1 - py)) / n
        + 4 * px * py * (1 - px) * (1 - py) / (n * n)
    )
    denominator = math.sqrt(max(variance_p - variance_star, 1e-15))
    return float((p - p_star) / denominator)


def weekly_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for decision, week in frame.groupby("decision_date", sort=True):
        pred = week["predicted_weekly_return"].to_numpy(float)
        actual = week["actual_weekly_return"].to_numpy(float)
        if np.std(pred) < 1e-12:
            rows.append(
                {
                    "decision_date": decision,
                    "rank_ic": math.nan,
                    "top3_excess": math.nan,
                    "top3_bottom3_spread": math.nan,
                    "ndcg_at_3": math.nan,
                    "prediction_dispersion": 0.0,
                    "mae": float(np.mean(np.abs(pred - actual))),
                    "mse": float(np.mean((pred - actual) ** 2)),
                }
            )
            continue
        top_weights = _fractional_extreme_weights(pred, 3, largest=True)
        bottom_weights = _fractional_extreme_weights(pred, 3, largest=False)
        relevance = pd.Series(actual).rank(method="average").to_numpy(float)
        dcg = _expected_tie_aware_dcg(pred, relevance, 3)
        ideal_dcg = _expected_tie_aware_dcg(relevance, relevance, 3)
        ndcg = float(dcg / ideal_dcg)
        rows.append(
            {
                "decision_date": decision,
                "rank_ic": _safe_corr(pred, actual),
                "top3_excess": float(np.dot(top_weights, actual) / 3 - actual.mean()),
                "top3_bottom3_spread": float(np.dot(top_weights, actual) / 3 - np.dot(bottom_weights, actual) / 3),
                "ndcg_at_3": ndcg,
                "prediction_dispersion": float(np.std(pred, ddof=1)),
                "mae": float(np.mean(np.abs(pred - actual))),
                "mse": float(np.mean((pred - actual) ** 2)),
            }
        )
    return pd.DataFrame(rows)


def aggregate_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    pred = frame["predicted_weekly_return"].to_numpy(float)
    actual = frame["actual_weekly_return"].to_numpy(float)
    neutral = (pred == 0) | (actual == 0)
    directional = ~neutral
    weekly = weekly_statistics(frame)
    return {
        "n_rows": int(len(frame)),
        "n_weeks": int(frame["decision_date"].nunique()),
        "mae": float(np.mean(np.abs(pred - actual))),
        "rmse": float(np.sqrt(np.mean((pred - actual) ** 2))),
        "direction_accuracy": float(np.mean((pred[directional] > 0) == (actual[directional] > 0))) if directional.any() else math.nan,
        "balanced_accuracy": _balanced_accuracy(pred, actual),
        "positive_prevalence": float(np.mean(actual[actual != 0] > 0)) if (actual != 0).any() else math.nan,
        "neutral_count": int(neutral.sum()),
        "pesaran_timmermann": pesaran_timmermann(pred, actual),
        "weekly_rank_ic": float(weekly["rank_ic"].mean()),
        "top3_excess": float(weekly["top3_excess"].mean()),
        "top3_bottom3_spread": float(weekly["top3_bottom3_spread"].mean()),
        "ndcg_at_3": float(weekly["ndcg_at_3"].mean()),
        "prediction_dispersion": float(weekly["prediction_dispersion"].mean()),
    }


def _moving_block_indices(n_weeks: int, rng: np.random.Generator, block: int = 4) -> np.ndarray:
    if n_weeks < block:
        return rng.integers(0, n_weeks, size=n_weeks)
    starts = rng.integers(0, n_weeks - block + 1, size=math.ceil(n_weeks / block))
    return np.concatenate([np.arange(start, start + block) for start in starts])[:n_weeks]


def bootstrap_intervals(frame: pd.DataFrame, *, seed: int = 20260823, repetitions: int = 2000) -> dict[str, list[float]]:
    dates = np.array(sorted(frame["decision_date"].unique()))
    by_week = [frame[frame["decision_date"] == day].sort_values("symbol") for day in dates]
    pred = np.stack([week["predicted_weekly_return"].to_numpy(float) for week in by_week])
    actual = np.stack([week["actual_weekly_return"].to_numpy(float) for week in by_week])
    weekly = weekly_statistics(frame).set_index("decision_date").loc[dates]
    rng = np.random.default_rng(seed)
    indices = np.stack([_moving_block_indices(len(dates), rng) for _ in range(repetitions)])
    sampled_pred, sampled_actual = pred[indices], actual[indices]
    neutral = (sampled_pred == 0) | (sampled_actual == 0)
    correct = (sampled_pred > 0) == (sampled_actual > 0)
    valid_count = (~neutral).sum(axis=(1, 2)).clip(min=1)
    positive = (sampled_actual > 0) & ~neutral
    negative = (sampled_actual < 0) & ~neutral
    positive_count = positive.sum(axis=(1, 2))
    negative_count = negative.sum(axis=(1, 2))
    tpr = np.where(positive_count > 0, (correct & positive).sum(axis=(1, 2)) / positive_count.clip(min=1), np.nan)
    tnr = np.where(negative_count > 0, (correct & negative).sum(axis=(1, 2)) / negative_count.clip(min=1), np.nan)
    flat_pred = sampled_pred.reshape(repetitions, -1)
    flat_actual = sampled_actual.reshape(repetitions, -1)
    values = {
        "mae": np.abs(sampled_pred - sampled_actual).mean(axis=(1, 2)),
        "rmse": np.sqrt(((sampled_pred - sampled_actual) ** 2).mean(axis=(1, 2))),
        "direction_accuracy": (correct & ~neutral).sum(axis=(1, 2)) / valid_count,
        "balanced_accuracy": (tpr + tnr) / 2,
        "positive_prevalence": ((sampled_actual > 0) & (sampled_actual != 0)).sum(axis=(1, 2)) / (sampled_actual != 0).sum(axis=(1, 2)).clip(min=1),
        "pesaran_timmermann": np.array([pesaran_timmermann(flat_pred[i], flat_actual[i]) for i in range(repetitions)]),
        "weekly_rank_ic": weekly["rank_ic"].to_numpy()[indices].mean(axis=1),
        "top3_excess": weekly["top3_excess"].to_numpy()[indices].mean(axis=1),
        "top3_bottom3_spread": weekly["top3_bottom3_spread"].to_numpy()[indices].mean(axis=1),
        "ndcg_at_3": weekly["ndcg_at_3"].to_numpy()[indices].mean(axis=1),
        "prediction_dispersion": weekly["prediction_dispersion"].to_numpy()[indices].mean(axis=1),
    }
    return {
        name: ([math.nan, math.nan] if np.isnan(series).all() else [float(np.nanquantile(series, 0.025)), float(np.nanquantile(series, 0.975))])
        for name, series in values.items()
    }


def paired_bootstrap_delta(
    challenger: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    seed: int = 20260823,
    repetitions: int = 2000,
) -> dict[str, dict[str, float | list[float]]]:
    key = ["decision_date", "symbol"]
    left_keys = challenger[key].astype(str).to_records(index=False).tolist()
    right_keys = reference[key].astype(str).to_records(index=False).tolist()
    if sorted(left_keys) != sorted(right_keys):
        raise ValueError("paired comparison row keys differ")
    joined = challenger.merge(reference, on=key, suffixes=("_challenger", "_reference"), validate="one_to_one")
    if not np.allclose(joined["actual_weekly_return_challenger"], joined["actual_weekly_return_reference"], rtol=0, atol=1e-12):
        raise ValueError("paired comparison actual labels differ")
    dates = np.array(sorted(joined["decision_date"].unique()))
    left = challenger.sort_values(["decision_date", "symbol"])
    right = reference.sort_values(["decision_date", "symbol"])
    left_weekly = weekly_statistics(left).set_index("decision_date").loc[dates]
    right_weekly = weekly_statistics(right).set_index("decision_date").loc[dates]
    left_mae = left.assign(error=lambda x: np.abs(x["predicted_weekly_return"] - x["actual_weekly_return"])).groupby("decision_date")["error"].mean().loc[dates].to_numpy()
    right_mae = right.assign(error=lambda x: np.abs(x["predicted_weekly_return"] - x["actual_weekly_return"])).groupby("decision_date")["error"].mean().loc[dates].to_numpy()
    rng = np.random.default_rng(seed)
    indices = np.stack([_moving_block_indices(len(dates), rng) for _ in range(repetitions)])
    samples = {
        "weekly_rank_ic": (left_weekly["rank_ic"].to_numpy() - right_weekly["rank_ic"].to_numpy())[indices].mean(axis=1),
        "top3_excess": (left_weekly["top3_excess"].to_numpy() - right_weekly["top3_excess"].to_numpy())[indices].mean(axis=1),
        "mae": (left_mae - right_mae)[indices].mean(axis=1),
    }
    output: dict[str, dict[str, float | list[float]]] = {}
    challenger_metrics, reference_metrics = aggregate_metrics(challenger), aggregate_metrics(reference)
    for name, values in samples.items():
        if np.isnan(values).all():
            raise RuntimeError(f"paired comparison metric is undefined for every week: {name}")
        output[name] = {
            "delta": float(challenger_metrics[name]) - float(reference_metrics[name]),
            "ci95": [float(np.nanquantile(values, 0.025)), float(np.nanquantile(values, 0.975))],
        }
    return output

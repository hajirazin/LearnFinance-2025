"""Causal gates, cross-sectional metrics, stability, and paired uncertainty."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

RIDGE_FEATURE_COLUMNS = [
    "past_week_log_return",
    "momentum_4w_log_return",
    "context_log_return",
    "volatility_4w",
]


def prediction_frame(
    metadata: pd.DataFrame,
    predicted_weekly_logs: np.ndarray,
    *,
    model: str,
    seed: str,
) -> pd.DataFrame:
    """Build one auditable arithmetic-return prediction frame."""
    if len(metadata) != len(predicted_weekly_logs):
        raise ValueError("prediction length does not match metadata")
    frame = metadata[
        ["fold", "decision_date", "symbol", "actual_weekly_log_return"]
    ].copy()
    frame["predicted_weekly_log_return"] = predicted_weekly_logs
    frame["actual_weekly_return"] = np.expm1(
        frame["actual_weekly_log_return"].to_numpy(float)
    )
    frame["predicted_weekly_return"] = np.expm1(predicted_weekly_logs)
    frame["model"] = model
    frame["seed"] = seed
    return frame


def build_causal_control_frames(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Fit close-only causal mean and ridge gates without evaluation leakage."""
    fit = panel[panel["split"].isin(["train", "validation"])].copy()
    evaluation = panel[panel["split"] == "evaluation"].copy()
    if fit.empty or evaluation.empty:
        raise RuntimeError("control fit or evaluation rows are empty")
    if (
        fit["actual_weekly_log_return"].isna().any()
        or evaluation["actual_weekly_log_return"].isna().any()
    ):
        raise RuntimeError("control labels are locked")
    ridge = Ridge(alpha=1.0)
    ridge.fit(fit[RIDGE_FEATURE_COLUMNS], fit["actual_weekly_log_return"])
    ridge_logs = ridge.predict(evaluation[RIDGE_FEATURE_COLUMNS])

    symbol_history = {
        symbol: values["actual_weekly_log_return"].to_list()
        for symbol, values in fit.groupby("symbol")
    }
    causal_logs = np.empty(len(evaluation), dtype=float)
    position = {index: offset for offset, index in enumerate(evaluation.index)}
    for _, indices in evaluation.groupby("decision_date", sort=True).groups.items():
        for index in indices:
            history = symbol_history.get(str(evaluation.at[index, "symbol"]), [])
            causal_logs[position[index]] = float(np.mean(history)) if history else 0.0
        for index in indices:
            symbol_history.setdefault(str(evaluation.at[index, "symbol"]), []).append(
                float(evaluation.at[index, "actual_weekly_log_return"])
            )

    metadata = evaluation.reset_index(drop=True)
    frames = {
        "causal_historical_mean": prediction_frame(
            metadata,
            causal_logs,
            model="causal_historical_mean",
            seed="control",
        ),
        "ridge": prediction_frame(
            metadata, np.asarray(ridge_logs), model="ridge", seed="control"
        ),
    }
    frames["ridge"].attrs["ridge_feature_columns"] = list(RIDGE_FEATURE_COLUMNS)
    frames["ridge"].attrs["ridge_alpha"] = 1.0
    return frames


def _rank_ic(predicted: np.ndarray, actual: np.ndarray) -> float:
    if len(predicted) < 3 or np.std(predicted) < 1e-12 or np.std(actual) < 1e-12:
        return math.nan
    predicted_rank = pd.Series(predicted).rank(method="average").to_numpy()
    actual_rank = pd.Series(actual).rank(method="average").to_numpy()
    return float(np.corrcoef(predicted_rank, actual_rank)[0, 1])


def weekly_metrics(frame: pd.DataFrame, *, top_k: int) -> pd.DataFrame:
    """Calculate weekly signal and adjacent-week stability metrics."""
    rows: list[dict[str, Any]] = []
    previous_by_fold: dict[str, set[str]] = {}
    group_columns = ["fold", "decision_date"]
    for (fold, decision), week in frame.groupby(group_columns, sort=True):
        week = week.sort_values("symbol")
        if len(week) < 2 * top_k:
            raise ValueError(f"{fold}/{decision} has fewer than {2 * top_k} symbols")
        predicted = week["predicted_weekly_return"].to_numpy(float)
        actual = week["actual_weekly_return"].to_numpy(float)
        has_ranking = float(np.std(predicted)) >= 1e-12
        top: np.ndarray | None = None
        bottom: np.ndarray | None = None
        overlap = math.nan
        turnover = math.nan
        if has_ranking:
            order = np.lexsort((week["symbol"].to_numpy(), -predicted))
            top = actual[order[:top_k]]
            bottom = actual[order[-top_k:]]
            selected = set(week.iloc[order[:top_k]]["symbol"])
            previous = previous_by_fold.get(str(fold))
            if previous is not None:
                overlap = len(previous & selected) / top_k
                turnover = len(selected - previous) / top_k
            previous_by_fold[str(fold)] = selected
        rows.append(
            {
                "fold": fold,
                "decision_date": decision,
                "n_symbols": len(week),
                "rank_ic": _rank_ic(predicted, actual),
                "top15_excess": (
                    float(top.mean() - actual.mean()) if top is not None else math.nan
                ),
                "top15_bottom15_spread": (
                    float(top.mean() - bottom.mean())
                    if top is not None and bottom is not None
                    else math.nan
                ),
                "mae": float(np.mean(np.abs(predicted - actual))),
                "mse": float(np.mean((predicted - actual) ** 2)),
                "top15_overlap": overlap,
                "top15_turnover": turnover,
            }
        )
    return pd.DataFrame(rows)


def _finite_mean(values: pd.Series) -> float:
    finite = values[np.isfinite(values.to_numpy(float))]
    return float(finite.mean()) if len(finite) else math.nan


def aggregate_metrics(frame: pd.DataFrame, *, top_k: int = 15) -> dict[str, Any]:
    """Aggregate point, direction, rank, portfolio, and stability diagnostics."""
    predicted = frame["predicted_weekly_return"].to_numpy(float)
    actual = frame["actual_weekly_return"].to_numpy(float)
    weekly = weekly_metrics(frame, top_k=top_k)
    predicted_up = predicted > 0
    actual_up = actual > 0
    positive = actual_up
    negative = ~actual_up
    balanced = (
        float(
            (
                (predicted_up[positive] == actual_up[positive]).mean()
                + (predicted_up[negative] == actual_up[negative]).mean()
            )
            / 2
        )
        if positive.any() and negative.any()
        else math.nan
    )
    rank_std = weekly["rank_ic"].std(ddof=1)
    return {
        "n_rows": len(frame),
        "n_weeks": len(weekly),
        "min_symbols_per_week": int(weekly["n_symbols"].min()),
        "median_symbols_per_week": float(weekly["n_symbols"].median()),
        "max_symbols_per_week": int(weekly["n_symbols"].max()),
        "mae": float(np.mean(np.abs(predicted - actual))),
        "rmse": float(np.sqrt(np.mean((predicted - actual) ** 2))),
        "direction_accuracy": float(np.mean(predicted_up == actual_up)),
        "balanced_accuracy": balanced,
        "positive_prevalence": float(np.mean(actual_up)),
        "weekly_rank_ic": _finite_mean(weekly["rank_ic"]),
        "rank_ic_information_ratio": (
            float(weekly["rank_ic"].mean() / rank_std)
            if np.isfinite(rank_std) and rank_std > 0
            else math.nan
        ),
        "top15_excess": _finite_mean(weekly["top15_excess"]),
        "top15_bottom15_spread": _finite_mean(weekly["top15_bottom15_spread"]),
        "top15_overlap": _finite_mean(weekly["top15_overlap"]),
        "top15_turnover": _finite_mean(weekly["top15_turnover"]),
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
    """Bootstrap paired weekly differences, including selection transitions."""
    keys = ["fold", "decision_date", "symbol"]
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
    left = weekly_metrics(challenger, top_k=top_k)
    right = weekly_metrics(reference, top_k=top_k)
    weekly_keys = ["fold", "decision_date"]
    paired = left.merge(
        right,
        on=weekly_keys,
        suffixes=("_challenger", "_reference"),
        validate="one_to_one",
    )
    if len(paired) != len(left) or len(paired) != len(right):
        raise ValueError("paired comparison week keys differ")
    rng = np.random.default_rng(seed)
    indices = np.stack(
        [
            _moving_block_indices(len(paired), rng, block_weeks)
            for _ in range(repetitions)
        ]
    )
    columns = [
        ("weekly_rank_ic", "rank_ic"),
        ("top15_excess", "top15_excess"),
        ("top15_bottom15_spread", "top15_bottom15_spread"),
        ("mae", "mae"),
        ("top15_overlap", "top15_overlap"),
        ("top15_turnover", "top15_turnover"),
    ]
    output: dict[str, dict[str, float | list[float]]] = {}
    for name, column in columns:
        differences = paired[f"{column}_challenger"].to_numpy(float) - paired[
            f"{column}_reference"
        ].to_numpy(float)
        sampled = differences[indices]
        counts = np.isfinite(sampled).sum(axis=1)
        samples = np.full(repetitions, np.nan, dtype=float)
        np.divide(np.nansum(sampled, axis=1), counts, out=samples, where=counts > 0)
        finite_differences = differences[np.isfinite(differences)]
        finite_samples = samples[np.isfinite(samples)]
        output[name] = {
            "delta": (
                float(finite_differences.mean())
                if len(finite_differences)
                else math.nan
            ),
            "ci95": (
                [
                    float(np.quantile(finite_samples, 0.025)),
                    float(np.quantile(finite_samples, 0.975)),
                ]
                if len(finite_samples)
                else [math.nan, math.nan]
            ),
        }
    return output


def research_clearance(
    arm_metrics: dict[str, dict[str, float]],
    comparisons: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Apply the preregistered causal-gate policy without production effects."""
    reasons: list[str] = []
    for fold_name in ("development_2024", "development_2025", "confirmatory_2026"):
        if arm_metrics[fold_name]["weekly_rank_ic"] <= 0:
            reasons.append(f"{fold_name} weekly rank IC is not positive")
    for evidence_kind in ("development", "confirmatory"):
        for gate in ("causal_historical_mean", "ridge"):
            comparison = comparisons[evidence_kind][gate]
            rank = comparison["weekly_rank_ic"]
            if rank["delta"] <= 0:
                reasons.append(f"{evidence_kind} rank IC did not beat {gate}")
            if evidence_kind == "confirmatory" and rank["ci95"][0] <= 0:
                reasons.append(f"confirmatory rank IC lower bound did not clear {gate}")
            for metric in ("top15_excess", "top15_bottom15_spread"):
                if evidence_kind == "confirmatory" and comparison[metric]["delta"] < 0:
                    reasons.append(f"confirmatory {metric} did not beat {gate}")
    return {
        "passed": not reasons,
        "failure_reasons": reasons,
        "production_action": "none",
    }

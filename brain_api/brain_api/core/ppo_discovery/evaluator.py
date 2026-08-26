"""Locked evaluation, comparators, and ablations for ppo_discovery."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from brain_api.core.ppo_discovery.config import (
    CASH_FLOOR,
    MAX_SELECTED,
    REQUIRED_ABLATIONS,
    PPODiscoveryConfig,
)
from brain_api.core.ppo_discovery.schemas import PPODiscoveryError


def weekly_net_cagr(weekly_net_log: Sequence[float]) -> float:
    """Convert additive weekly log returns into annualized CAGR."""
    if len(weekly_net_log) == 0:
        raise PPODiscoveryError("CAGR requires at least one week")
    total = float(np.sum(weekly_net_log))
    n = len(weekly_net_log)
    return float(np.expm1(total * (52.0 / n)))


def max_drawdown(weekly_net_log: Sequence[float]) -> float:
    """Maximum peak-to-trough drawdown of compounded wealth starting at 1.0."""
    logs = np.asarray(weekly_net_log, dtype=np.float64)
    wealth = np.concatenate(([1.0], np.exp(np.cumsum(logs))))
    peaks = np.maximum.accumulate(wealth)
    dd = 1.0 - wealth / np.maximum(peaks, 1e-12)
    return float(np.max(dd)) if len(dd) else 0.0


def block_bootstrap_mean_ci(
    paired_diffs: Sequence[float],
    *,
    block_size: int = 4,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return ``(point, lo, hi)`` 95% CI from a moving-block bootstrap."""
    series = np.asarray(paired_diffs, dtype=np.float64)
    if len(series) == 0:
        raise PPODiscoveryError("bootstrap requires paired weekly differences")
    rng = np.random.default_rng(seed)
    n = len(series)
    if n < 4:
        raise PPODiscoveryError("bootstrap requires at least 4 paired weeks")
    n_blocks = int(np.ceil(n / block_size))
    max_start = n - block_size
    means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        sample = np.concatenate(
            [series[start : start + block_size] for start in starts]
        )[:n]
        means[i] = sample.mean()
    point = float(series.mean())
    lo, hi = np.quantile(means, [0.025, 0.975])
    return point, float(lo), float(hi)


def cash_only_weights() -> dict[str, float]:
    return {"CASH": 1.0}


def equal_weight_news_rank(
    news_rank: Mapping[str, float], eligible: Sequence[str]
) -> dict[str, float]:
    """Top 15 by news rank, equal stock weights, 2% cash."""
    ordered = sorted(eligible, key=lambda symbol: (-float(news_rank[symbol]), symbol))
    selected = ordered[:MAX_SELECTED]
    if not selected:
        return cash_only_weights()
    stock_mass = 1.0 - CASH_FLOOR
    weight = stock_mass / len(selected)
    weights = dict.fromkeys(selected, weight)
    weights["CASH"] = CASH_FLOOR
    return weights


def reject_current_patchtst_on_old_weeks(use_current_patchtst: bool) -> None:
    """Alpha-HRP replica must not score historical weeks with today's current."""
    if use_current_patchtst:
        raise PPODiscoveryError(
            "Alpha-HRP replica cannot score old weeks with today's PatchTST current"
        )


def mark_ablations(
    available: Mapping[str, Any],
) -> dict[str, Any]:
    """Every required ablation is present or explicitly marked unavailable."""
    report: dict[str, Any] = {}
    for name in REQUIRED_ABLATIONS:
        if name in available:
            report[name] = available[name]
        else:
            report[name] = {"status": "unavailable"}
    return report


def aggregate_seed_metrics(seed_cagrs: Mapping[int, float]) -> dict[str, float]:
    values = np.asarray(list(seed_cagrs.values()), dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def select_candidate_seed(
    seed_val_cagrs: Mapping[int, float], seed_val_sharpes: Mapping[int, float]
) -> int:
    """Median validation net CAGR; Sharpe then lower seed as ties."""
    values = np.asarray(list(seed_val_cagrs.values()), dtype=np.float64)
    median = float(np.median(values))
    ranked = sorted(
        seed_val_cagrs,
        key=lambda seed: (
            abs(seed_val_cagrs[seed] - median),
            -seed_val_sharpes.get(seed, 0.0),
            seed,
        ),
    )
    return int(ranked[0])


def evaluate_policy_weeks(
    weekly_net_log: Sequence[float],
    *,
    config: PPODiscoveryConfig | None = None,
) -> dict[str, float]:
    del config
    logs = np.asarray(weekly_net_log, dtype=np.float64)
    cagr = weekly_net_cagr(logs)
    vol = float(logs.std(ddof=0) * np.sqrt(52))
    sharpe = float((logs.mean() * 52) / vol) if vol > 0 else 0.0
    return {
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(logs),
        "n_weeks": float(len(logs)),
    }


__all__ = [
    "aggregate_seed_metrics",
    "block_bootstrap_mean_ci",
    "cash_only_weights",
    "equal_weight_news_rank",
    "evaluate_policy_weeks",
    "mark_ablations",
    "max_drawdown",
    "reject_current_patchtst_on_old_weeks",
    "select_candidate_seed",
    "weekly_net_cagr",
]

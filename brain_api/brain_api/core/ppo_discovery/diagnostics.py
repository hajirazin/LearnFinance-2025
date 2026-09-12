"""Turnover, membership, and allocation-head diagnostics for ppo_discovery."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np

from brain_api.core.portfolio_rl.constraints import compute_turnover_from_allocations
from brain_api.core.ppo_discovery.schemas import PPODiscoveryError

_MEMBERSHIP_WEIGHT = 1e-12
_SIMPLEX_TOLERANCE = 1e-6


@dataclass(frozen=True)
class PortfolioTransitionDiagnostics:
    """One prior-to-target book transition. Cost is None on live inference."""

    turnover: float
    transaction_cost_fraction: float | None
    retained_count: int
    entered_count: int
    exited_count: int
    replacement_fraction: float
    stock_weight_cv: float
    equal_weight_l1_distance: float

    def to_dict(self) -> dict[str, Any]:
        cost = self.transaction_cost_fraction
        return {
            "turnover": float(self.turnover),
            "transaction_cost_fraction": None if cost is None else float(cost),
            "retained_count": int(self.retained_count),
            "entered_count": int(self.entered_count),
            "exited_count": int(self.exited_count),
            "replacement_fraction": float(self.replacement_fraction),
            "stock_weight_cv": float(self.stock_weight_cv),
            "equal_weight_l1_distance": float(self.equal_weight_l1_distance),
        }


def compute_portfolio_transition_diagnostics(
    prior: Mapping[str, float],
    target: Mapping[str, float],
    *,
    transaction_cost_fraction: float | None,
) -> PortfolioTransitionDiagnostics:
    """Membership, turnover, CV, and equal-weight L1 for one rebalance."""
    _require_allocation(prior, "prior")
    _require_allocation(target, "target")
    if transaction_cost_fraction is not None:
        cost = float(transaction_cost_fraction)
        if not isfinite(cost) or cost < 0.0:
            raise PPODiscoveryError(
                "transaction_cost_fraction must be finite and >= 0, "
                f"got {transaction_cost_fraction!r}"
            )
        transaction_cost_fraction = cost
    turnover = float(compute_turnover_from_allocations(dict(prior), dict(target)))
    prior_stocks = _stock_membership(prior)
    target_stocks = _stock_membership(target)
    retained = prior_stocks & target_stocks
    entered = target_stocks - prior_stocks
    exited = prior_stocks - target_stocks
    target_count = len(target_stocks)
    replacement = (
        0.0 if target_count == 0 else float(len(entered)) / float(target_count)
    )
    weights = np.asarray(
        [float(target[symbol]) for symbol in sorted(target_stocks)],
        dtype=np.float64,
    )
    return PortfolioTransitionDiagnostics(
        turnover=turnover,
        transaction_cost_fraction=transaction_cost_fraction,
        retained_count=len(retained),
        entered_count=len(entered),
        exited_count=len(exited),
        replacement_fraction=replacement,
        stock_weight_cv=_stock_weight_cv(weights),
        equal_weight_l1_distance=_equal_weight_l1(weights),
    )


def summarize_portfolio_transition_diagnostics(
    rows: Sequence[PortfolioTransitionDiagnostics],
) -> dict[str, float]:
    """Mean/median/max closed-loop diagnostics. Null cost is not allowed."""
    if not rows:
        raise PPODiscoveryError(
            "portfolio diagnostics summary is empty; at least one week is required"
        )
    if any(row.transaction_cost_fraction is None for row in rows):
        raise PPODiscoveryError(
            "closed-loop portfolio diagnostics require a finite "
            "transaction_cost_fraction on every week"
        )
    turnovers = np.asarray([row.turnover for row in rows], dtype=np.float64)
    costs = np.asarray(
        [float(row.transaction_cost_fraction) for row in rows], dtype=np.float64
    )
    mean_cost = float(costs.mean())
    return {
        "n_weeks": float(len(rows)),
        "mean_turnover": float(turnovers.mean()),
        "median_turnover": float(np.median(turnovers)),
        "max_turnover": float(turnovers.max()),
        "mean_transaction_cost_fraction": mean_cost,
        "mean_transaction_cost_bps": mean_cost * 10_000.0,
        "mean_replacement_fraction": float(
            np.mean([row.replacement_fraction for row in rows])
        ),
        "mean_retained_count": float(np.mean([row.retained_count for row in rows])),
        "mean_entered_count": float(np.mean([row.entered_count for row in rows])),
        "mean_exited_count": float(np.mean([row.exited_count for row in rows])),
        "mean_stock_weight_cv": float(np.mean([row.stock_weight_cv for row in rows])),
        "mean_equal_weight_l1_distance": float(
            np.mean([row.equal_weight_l1_distance for row in rows])
        ),
    }


def build_allocation_head_diagnostics(
    ablations: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare full PPO test CAGR to equal-weight-selected. No fabricated flags."""
    full = ablations.get("full_ppo")
    equal = ablations.get("equal_weight_selected")
    if _ok_finite_cagr(full) and _ok_finite_cagr(equal):
        full_cagr = float(full["cagr"])
        equal_cagr = float(equal["cagr"])
        delta = full_cagr - equal_cagr
        return {
            "full_ppo_cagr": full_cagr,
            "equal_weight_selected_cagr": equal_cagr,
            "cagr_delta": delta,
            "ppo_outperformed_equal_weight": delta > 0.0,
        }
    return {
        "status": "unavailable",
        "full_ppo": full if isinstance(full, dict) else {"status": "missing"},
        "equal_weight_selected": (
            equal if isinstance(equal, dict) else {"status": "missing"}
        ),
    }


def build_transaction_cost_training_diagnostics(
    ablations: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare cost-trained vs no-cost-trained net CAGR and mean turnover."""
    cost_trained = ablations.get("full_ppo")
    no_cost = ablations.get("no_transaction_cost_term")
    if _ok_with_mean_turnover(cost_trained) and _ok_with_mean_turnover(no_cost):
        cost_cagr = float(cost_trained["cagr"])
        no_cost_cagr = float(no_cost["cagr"])
        cost_turnover = float(
            cost_trained["portfolio_diagnostics"]["summary"]["mean_turnover"]
        )
        no_cost_turnover = float(
            no_cost["portfolio_diagnostics"]["summary"]["mean_turnover"]
        )
        cagr_delta = cost_cagr - no_cost_cagr
        turnover_delta = cost_turnover - no_cost_turnover
        return {
            "cost_trained_net_cagr": cost_cagr,
            "no_cost_trained_net_cagr": no_cost_cagr,
            "net_cagr_delta": cagr_delta,
            "cost_trained_mean_turnover": cost_turnover,
            "no_cost_trained_mean_turnover": no_cost_turnover,
            "mean_turnover_delta": turnover_delta,
            "cost_training_improved_net_cagr": cagr_delta > 0.0,
            "cost_training_reduced_turnover": turnover_delta < 0.0,
        }
    return {
        "status": "unavailable",
        "full_ppo": (
            cost_trained if isinstance(cost_trained, dict) else {"status": "missing"}
        ),
        "no_transaction_cost_term": (
            no_cost if isinstance(no_cost, dict) else {"status": "missing"}
        ),
    }


def _require_allocation(weights: Mapping[str, float], label: str) -> None:
    if "CASH" not in weights:
        raise PPODiscoveryError(f"{label} allocation is missing CASH")
    total = 0.0
    for symbol, raw in weights.items():
        try:
            weight = float(raw)
        except (TypeError, ValueError) as exc:
            raise PPODiscoveryError(
                f"{label} weight for {symbol!r} is not finite"
            ) from exc
        if not isfinite(weight):
            raise PPODiscoveryError(f"{label} weight for {symbol!r} is not finite")
        if weight < 0.0:
            raise PPODiscoveryError(f"{label} weight for {symbol!r} is negative")
        total += weight
    if abs(total - 1.0) > _SIMPLEX_TOLERANCE:
        raise PPODiscoveryError(
            f"{label} allocation is not a simplex (sum={total}, tol={_SIMPLEX_TOLERANCE})"
        )


def _stock_membership(weights: Mapping[str, float]) -> set[str]:
    return {
        symbol
        for symbol, weight in weights.items()
        if symbol != "CASH" and float(weight) > _MEMBERSHIP_WEIGHT
    }


def _stock_weight_cv(weights: np.ndarray) -> float:
    if weights.size < 2:
        return 0.0
    mean = float(weights.mean())
    if mean == 0.0:
        return 0.0
    return float(weights.std(ddof=0) / mean)


def _equal_weight_l1(weights: np.ndarray) -> float:
    k = int(weights.size)
    if k == 0:
        return 0.0
    invested = float(weights.sum())
    equal = invested / float(k)
    return float(0.5 * np.sum(np.abs(weights - equal)))


def _finite_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return isfinite(number)


def _ok_finite_cagr(row: Any) -> bool:
    return (
        isinstance(row, dict)
        and row.get("status") == "ok"
        and _finite_number(row.get("cagr"))
    )


def _ok_with_mean_turnover(row: Any) -> bool:
    if not _ok_finite_cagr(row):
        return False
    summary = (row.get("portfolio_diagnostics") or {}).get("summary") or {}
    return _finite_number(summary.get("mean_turnover"))


__all__ = [
    "PortfolioTransitionDiagnostics",
    "build_allocation_head_diagnostics",
    "build_transaction_cost_training_diagnostics",
    "compute_portfolio_transition_diagnostics",
    "summarize_portfolio_transition_diagnostics",
]

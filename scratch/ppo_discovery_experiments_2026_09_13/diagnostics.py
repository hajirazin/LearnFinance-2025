"""Research-only ablation comparison diagnostics. Not a production promote gate."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any


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
    "build_allocation_head_diagnostics",
    "build_transaction_cost_training_diagnostics",
]

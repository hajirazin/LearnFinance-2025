"""Portfolio transition diagnostics: membership, turnover, CV, equal-weight L1."""

from __future__ import annotations

import math

import pytest

from brain_api.core.ppo_discovery.diagnostics import (
    PortfolioTransitionDiagnostics,
    build_allocation_head_diagnostics,
    build_transaction_cost_training_diagnostics,
    compute_portfolio_transition_diagnostics,
    summarize_portfolio_transition_diagnostics,
)
from brain_api.core.ppo_discovery.schemas import PPODiscoveryError


def test_entries_exits_retention_and_replacement() -> None:
    row = compute_portfolio_transition_diagnostics(
        {"AAPL": 0.3, "MSFT": 0.3, "CASH": 0.4},
        {"MSFT": 0.3, "GOOG": 0.3, "CASH": 0.4},
        transaction_cost_fraction=0.001,
    )
    assert row.retained_count == 1
    assert row.entered_count == 1
    assert row.exited_count == 1
    assert row.replacement_fraction == pytest.approx(0.5)


def test_turnover_includes_forced_exit() -> None:
    row = compute_portfolio_transition_diagnostics(
        {"AAPL": 0.4, "CASH": 0.6},
        {"CASH": 1.0},
        transaction_cost_fraction=0.0,
    )
    assert row.turnover == pytest.approx(0.4)
    assert row.exited_count == 1
    assert row.entered_count == 0
    assert row.retained_count == 0


def test_cash_only_replacement_and_cv_are_zero() -> None:
    row = compute_portfolio_transition_diagnostics(
        {"CASH": 1.0},
        {"CASH": 1.0},
        transaction_cost_fraction=0.0,
    )
    assert row.replacement_fraction == 0.0
    assert row.stock_weight_cv == 0.0
    assert row.equal_weight_l1_distance == 0.0
    assert row.retained_count == 0


def test_equal_weight_l1_is_zero_for_equal_book() -> None:
    equal = compute_portfolio_transition_diagnostics(
        {"AAPL": 0.3, "MSFT": 0.3, "CASH": 0.4},
        {"AAPL": 0.3, "MSFT": 0.3, "CASH": 0.4},
        transaction_cost_fraction=0.0,
    )
    overweight = compute_portfolio_transition_diagnostics(
        {"AAPL": 0.3, "MSFT": 0.3, "CASH": 0.4},
        {"AAPL": 0.5, "MSFT": 0.1, "CASH": 0.4},
        transaction_cost_fraction=0.0,
    )
    assert equal.equal_weight_l1_distance == pytest.approx(0.0)
    assert overweight.equal_weight_l1_distance == pytest.approx(0.2)


def test_cv_zero_for_single_stock() -> None:
    row = compute_portfolio_transition_diagnostics(
        {"CASH": 1.0},
        {"AAPL": 0.98, "CASH": 0.02},
        transaction_cost_fraction=0.001,
    )
    assert row.stock_weight_cv == 0.0


def test_rejects_missing_cash() -> None:
    with pytest.raises(PPODiscoveryError, match="CASH"):
        compute_portfolio_transition_diagnostics(
            {"AAPL": 1.0},
            {"AAPL": 0.5, "CASH": 0.5},
            transaction_cost_fraction=0.0,
        )


def test_rejects_negative() -> None:
    with pytest.raises(PPODiscoveryError, match="negative"):
        compute_portfolio_transition_diagnostics(
            {"AAPL": -0.1, "CASH": 1.1},
            {"CASH": 1.0},
            transaction_cost_fraction=0.0,
        )


def test_rejects_nonfinite() -> None:
    with pytest.raises(PPODiscoveryError, match="finite"):
        compute_portfolio_transition_diagnostics(
            {"AAPL": math.nan, "CASH": 1.0},
            {"CASH": 1.0},
            transaction_cost_fraction=0.0,
        )


def test_rejects_nonsimplex() -> None:
    with pytest.raises(PPODiscoveryError, match="simplex"):
        compute_portfolio_transition_diagnostics(
            {"AAPL": 0.5, "CASH": 0.6},
            {"CASH": 1.0},
            transaction_cost_fraction=0.0,
        )


def test_summarize_keys_and_cost_bps() -> None:
    rows = [
        compute_portfolio_transition_diagnostics(
            {"AAPL": 0.4, "CASH": 0.6},
            {"AAPL": 0.5, "CASH": 0.5},
            transaction_cost_fraction=0.001,
        ),
        compute_portfolio_transition_diagnostics(
            {"AAPL": 0.5, "CASH": 0.5},
            {"MSFT": 0.5, "CASH": 0.5},
            transaction_cost_fraction=0.003,
        ),
    ]
    summary = summarize_portfolio_transition_diagnostics(rows)
    assert summary["n_weeks"] == 2.0
    assert set(summary) == {
        "n_weeks",
        "mean_turnover",
        "median_turnover",
        "max_turnover",
        "mean_transaction_cost_fraction",
        "mean_transaction_cost_bps",
        "mean_replacement_fraction",
        "mean_retained_count",
        "mean_entered_count",
        "mean_exited_count",
        "mean_stock_weight_cv",
        "mean_equal_weight_l1_distance",
    }
    assert summary["mean_transaction_cost_fraction"] == pytest.approx(0.002)
    assert summary["mean_transaction_cost_bps"] == pytest.approx(20.0)
    assert summary["mean_turnover"] == pytest.approx(
        (rows[0].turnover + rows[1].turnover) / 2.0
    )


def test_summarize_rejects_empty() -> None:
    with pytest.raises(PPODiscoveryError, match="empty"):
        summarize_portfolio_transition_diagnostics([])


def test_summarize_rejects_null_cost_on_closed_loop_rows() -> None:
    live = compute_portfolio_transition_diagnostics(
        {"CASH": 1.0},
        {"AAPL": 0.5, "CASH": 0.5},
        transaction_cost_fraction=None,
    )
    with pytest.raises(PPODiscoveryError, match="transaction_cost_fraction"):
        summarize_portfolio_transition_diagnostics([live])


def test_allocation_head_diagnostics_boolean_only_when_both_ok() -> None:
    ok = build_allocation_head_diagnostics(
        {
            "full_ppo": {"status": "ok", "cagr": 0.20},
            "equal_weight_selected": {"status": "ok", "cagr": 0.15},
        }
    )
    assert ok["ppo_outperformed_equal_weight"] is True
    assert ok["cagr_delta"] == pytest.approx(0.05)

    missing = build_allocation_head_diagnostics(
        {"full_ppo": {"status": "failed", "error": "boom"}}
    )
    assert missing["status"] == "unavailable"
    assert "ppo_outperformed_equal_weight" not in missing


def test_cost_training_diagnostics_require_turnover() -> None:
    ablations = {
        "full_ppo": {
            "status": "ok",
            "cagr": 0.18,
            "portfolio_diagnostics": {"summary": {"mean_turnover": 0.10}},
        },
        "no_transaction_cost_term": {
            "status": "ok",
            "cagr": 0.20,
            "portfolio_diagnostics": {"summary": {"mean_turnover": 0.40}},
        },
    }
    payload = build_transaction_cost_training_diagnostics(ablations)
    assert payload["cost_training_improved_net_cagr"] is False
    assert payload["cost_training_reduced_turnover"] is True
    assert payload["net_cagr_delta"] == pytest.approx(-0.02)
    assert payload["mean_turnover_delta"] == pytest.approx(-0.30)

    unavailable = build_transaction_cost_training_diagnostics(
        {"full_ppo": {"status": "ok", "cagr": 0.18}}
    )
    assert unavailable["status"] == "unavailable"
    assert "cost_training_improved_net_cagr" not in unavailable


def test_diagnostics_to_dict_preserves_null_cost() -> None:
    row = compute_portfolio_transition_diagnostics(
        {"CASH": 1.0},
        {"CASH": 1.0},
        transaction_cost_fraction=None,
    )
    assert isinstance(row, PortfolioTransitionDiagnostics)
    assert row.to_dict()["transaction_cost_fraction"] is None

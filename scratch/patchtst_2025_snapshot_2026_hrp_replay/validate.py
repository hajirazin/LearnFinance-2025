"""Reconcile saved HRP positions, cash flows, original curves and causal targets."""

import json

import numpy as np
import pandas as pd
from hrp_replay import INITIAL, LABEL, hrp_target
from run import ROOT, SOURCE, load_evidence, sha256, source_hashes, write_json

from brain_api.core.hrp import compute_hrp_allocation
from brain_api.core.patchtst.calendar import load_patchtst_calendar


def main():
    summary = json.loads((ROOT / "summary.json").read_text())
    _, _, prices, plan, original = load_evidence(SOURCE)
    nav = pd.read_csv(ROOT / "weekly_nav.csv", index_col="date", parse_dates=True)
    holdings = pd.read_csv(ROOT / "hrp_weekly_holdings.csv", parse_dates=["monday"])
    trades = pd.read_csv(ROOT / "hrp_trades.csv", parse_dates=["monday"])
    ledger = json.loads((ROOT / "hrp_weekly_ledger.json").read_text())
    sessions = load_patchtst_calendar(
        "us", pd.Timestamp("2023-11-01").date(), pd.Timestamp("2026-12-10").date()
    ).sessions
    np.testing.assert_allclose(nav[original.columns], original, atol=1e-10, rtol=0)
    assert nav.index.equals(original.index)
    assert len(ledger) == len(plan) == summary["weeks"]
    previous, residuals, facade_diffs = {}, [], []
    for index, (week, book) in enumerate(zip(plan, ledger, strict=True)):
        held = holdings.loc[holdings.monday.eq(week["monday"])].set_index("symbol")
        transaction = trades.loc[trades.monday.eq(week["monday"])]
        assert len(held) == held.index.nunique() == 15
        assert set(held.index) == set(week["selected"])
        assert set(book["target_weights"]) == set(held.index)
        assert (
            pd.Timestamp(book["history"]["history_end"])
            == week["anchor"]
            < week["monday"]
        )
        assert book["history"]["daily_returns"] == 252
        expected_dates = sessions[sessions <= week["anchor"]][-253:]
        assert pd.Timestamp(book["history"]["history_start"]) == expected_dates[0]
        # Exercise the production high-level facade on explicitly truncated prices.
        subset = {s: prices[s].reindex(expected_dates) for s in week["selected"]}
        production = compute_hrp_allocation(
            subset,
            lookback_days=252,
            min_data_days=253,
            as_of_date=week["monday"].date(),
        )
        assert not production.symbols_excluded
        for s, percent in production.percentage_weights.items():
            facade_diffs.append(abs(held.at[s, "target_weight"] - percent / 100))
        opening_prices = transaction.set_index("symbol").price.to_dict()
        opening_nav = (
            sum(q * opening_prices[s] for s, q in previous.items())
            if previous
            else INITIAL
        )
        np.testing.assert_allclose(
            held.opening_price,
            [prices[s].at[week["execution"], "open"] for s in held.index],
            atol=1e-10,
            rtol=0,
        )
        np.testing.assert_allclose(
            held.closing_price,
            [prices[s].at[week["valuation"], "close"] for s in held.index],
            atol=1e-10,
            rtol=0,
        )
        np.testing.assert_allclose(
            held.opening_value, opening_nav * held.target_weight, atol=1e-8, rtol=0
        )
        np.testing.assert_allclose(
            held.closing_value,
            held.quantity_adjusted_units * held.closing_price,
            atol=1e-8,
            rtol=0,
        )
        assert np.isclose(
            held.closing_value.sum(),
            nav.loc[week["valuation"], LABEL],
            atol=1e-8,
            rtol=0,
        )
        assert np.isclose(held.target_weight.sum(), 1, atol=1e-12, rtol=0)
        assert np.isclose(
            book["rebalance"]["nav_before"], opening_nav, atol=1e-8, rtol=0
        )
        assert book["rebalance"]["cash_after"] == book["rebalance"]["fees"] == 0
        before_quantities = transaction.set_index("symbol").quantity_before.to_dict()
        after_quantities = transaction.set_index("symbol").quantity_after.to_dict()
        assert set(before_quantities) == set(previous) | set(held.index)
        for s in before_quantities:
            assert np.isclose(
                before_quantities[s], previous.get(s, 0), atol=1e-9, rtol=0
            )
            assert np.isclose(
                after_quantities[s],
                held.quantity_adjusted_units.get(s, 0),
                atol=1e-9,
                rtol=0,
            )
        np.testing.assert_allclose(
            transaction.quantity_delta,
            transaction.quantity_after - transaction.quantity_before,
            atol=1e-9,
            rtol=0,
        )
        residuals.append(
            abs(
                (transaction.quantity_delta * transaction.price).sum()
                - (INITIAL if index == 0 else 0)
            )
        )
        previous = held.quantity_adjusted_units.to_dict()
    assert max(residuals) < 1e-7
    assert max(facade_diffs) <= 0.000000500001
    causal_checks = 0
    for week in [plan[0], plan[len(plan) // 2], plan[-1]]:
        before, _ = hrp_target(prices, week["selected"], sessions, week["anchor"])
        changed = {s: prices[s].copy() for s in week["selected"]}
        for frame in changed.values():
            frame.loc[frame.index > week["anchor"], "close"] = -999999
        after, _ = hrp_target(changed, week["selected"], sessions, week["anchor"])
        assert before == after
        causal_checks += 1
    assert source_hashes(SOURCE) == summary["identity"]["source_sha256"]
    for name, digest in summary["outputs_sha256"].items():
        assert sha256(ROOT / name) == digest
    for label in nav:
        assert np.isclose(
            nav[label].iloc[-1],
            summary["metrics"][label]["ending_value"],
            atol=1e-8,
            rtol=0,
        )
    report = {
        "run_id": summary["run_id"],
        "attempt": 1,
        "as_of": summary["as_of"],
        "weekly_books_reconciled": len(ledger),
        "holdings_rows": len(holdings),
        "unchanged_original_curves": list(original.columns),
        "selected_stock_sets_identical": True,
        "original_experiment_all_file_hashes_unchanged": True,
        "exact_prior_session_daily_returns": 252,
        "production_hrp_facade_comparisons": len(facade_diffs),
        "max_production_rounding_difference_fraction": max(facade_diffs),
        "future_price_perturbation_checks": causal_checks,
        "maximum_self_financing_residual_usd": max(residuals),
        "price_fetches": 0,
        "model_inference_runs": 0,
        "training_runs": 0,
    }
    write_json(ROOT / "validation.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

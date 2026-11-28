"""Replay saved equal-weight and HRP targets with repository IBKR economics."""

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from fee_accounting import fee_rebalance
from hrp_replay import INITIAL, LABEL, mark_to_market, quotes
from run import REPO, ROOT, SOURCE, load_evidence, sha256, source_hashes, write_json
from scratch.ppo_2025_model_2026_simulation_2026_11_07.book import buy_and_hold_book

from brain_api.core.portfolio_rl.broker_costs import (
    IBKRSingaporeCostConfig,
    compute_ibkr_leg_cost,
)

OUTPUT = ROOT / "ibkr_fees"
CONVENTIONS = {"repo_default": 0.005, "every_trade": 1e-12}


def simulate_fees(prices, plan, targets, epsilon):
    shares, records, fees = {}, [], 0.0
    values, dates = [INITIAL], [plan[0]["execution"]]
    max_residual = 0.0
    for week, weights in zip(plan, targets, strict=True):
        opening = quotes(prices, set(shares) | set(weights), week["execution"], "open")
        before = shares
        shares, record = fee_rebalance(
            shares, weights, opening, INITIAL if not records else 0, epsilon
        )
        closing = quotes(prices, shares, week["valuation"], "close")
        closing_nav = mark_to_market(shares, closing)
        # Independently reconcile the exported trade ledger against actual
        # quantities and the per-leg production fee function.
        actual_nav = (
            sum(q * opening[s] for s, q in before.items()) if records else INITIAL
        )
        assert np.isclose(actual_nav, record["nav_before"], rtol=0, atol=1e-8)
        for trade in record["trades"]:
            delta = shares.get(trade["symbol"], 0) - before.get(trade["symbol"], 0)
            assert np.isclose(trade["quantity_delta"], delta, rtol=0, atol=1e-9)
            exempt = trade["notional_usd"] / actual_nav < epsilon
            assert exempt == trade["fee_exempt"]
            expected = (
                0
                if exempt
                else compute_ibkr_leg_cost(
                    trade["symbol"],
                    abs(delta) * trade["price"],
                    abs(delta),
                    trade["side"],
                    IBKRSingaporeCostConfig.default().with_nav(actual_nav),
                ).total
            )
            assert np.isclose(trade["total"], expected, rtol=0, atol=1e-9)
        assert np.isclose(
            sum(t["total"] for t in record["trades"]),
            record["fees_usd"],
            rtol=0,
            atol=1e-8,
        )
        assert set(shares) == set(week["selected"])
        np.testing.assert_allclose(
            [shares[s] * opening[s] / record["nav_after"] for s in weights],
            list(weights.values()),
            rtol=0,
            atol=1e-12,
        )
        max_residual = max(
            max_residual,
            abs(record["cash_after"]),
            abs(record["nav_after"] + record["fees_usd"] - actual_nav),
        )
        fees += record["fees_usd"]
        records.append(
            {
                **week,
                **record,
                "closing_nav": closing_nav,
                "closing_prices": closing,
                "shares_after": shares,
                "target_weights": weights,
            }
        )
        values.append(closing_nav)
        dates.append(week["valuation"])
    return (
        pd.Series(values, index=pd.DatetimeIndex(dates, name="date")),
        records,
        fees,
        max_residual,
    )


def main():
    OUTPUT.mkdir(exist_ok=True)
    sources = source_hashes(SOURCE)
    zero_cost = {
        name: sha256(ROOT / name)
        for name in ["summary.json", "weekly_nav.csv", "hrp_weekly_ledger.json"]
    }
    identity = {
        "original_source_sha256": sources,
        "zero_fee_result_sha256": zero_cost,
        "cost_config": IBKRSingaporeCostConfig.default().to_dict(),
        "nav_policy": "actual current opening NAV, starting at USD 10000",
        "conventions": CONVENTIONS,
        "code_sha256": {
            str(path.relative_to(REPO)): sha256(path)
            for path in [
                ROOT / "fee_accounting.py",
                ROOT / "fee_run.py",
                REPO / "brain_api/brain_api/core/portfolio_rl/broker_costs.py",
                REPO / "scratch/ppo_2025_model_2026_simulation_2026_11_07/book.py",
            ]
        },
    }
    if (OUTPUT / "summary.json").exists():
        saved = json.loads((OUTPUT / "summary.json").read_text())
        if saved["identity"] != identity:
            raise ValueError(
                "Existing fee result has a different identity; refusing overwrite"
            )
        assert all(
            sha256(OUTPUT / name) == digest
            for name, digest in saved["output_sha256"].items()
        )
        print("IBKR comparison already complete; skipped.")
        return
    _, manifest, prices, plan, _ = load_evidence(SOURCE)
    combined = pd.read_csv(ROOT / "weekly_nav.csv", index_col="date", parse_dates=True)
    hrp_ledger = json.loads((ROOT / "hrp_weekly_ledger.json").read_text())
    targets = {
        "PatchTST": [
            dict.fromkeys(w["selected"], 1 / len(w["selected"])) for w in plan
        ],
        LABEL: [w["target_weights"] for w in hrp_ledger],
    }
    metrics, trades, weekly, all_records = {}, [], [], {}
    for convention, epsilon in CONVENTIONS.items():
        for strategy, weights in targets.items():
            nav, ledger, fees, residual = simulate_fees(prices, plan, weights, epsilon)
            assert nav.index.equals(combined.index)
            name = f"{strategy} / IBKR {convention}"
            combined[name] = nav
            gross_end = float(combined[strategy].iloc[-1])
            metrics[name] = {
                "strategy": strategy,
                "convention": convention,
                "ending_value": float(nav.iloc[-1]),
                "total_return": float(nav.iloc[-1] / INITIAL - 1),
                "max_weekly_drawdown": float((nav / nav.cummax() - 1).min()),
                "total_fees_usd": fees,
                "ending_value_reduction_usd": gross_end - float(nav.iloc[-1]),
                "return_reduction_pp": (gross_end - float(nav.iloc[-1]))
                / INITIAL
                * 100,
                "maximum_accounting_residual_usd": residual,
                "trades": sum(len(w["trades"]) for w in ledger),
                "fee_exempt_trades": sum(
                    t["fee_exempt"] for w in ledger for t in w["trades"]
                ),
            }
            all_records[name] = ledger
            for w in ledger:
                weekly.append(
                    {
                        "portfolio": name,
                        "monday": w["monday"],
                        "execution": w["execution"],
                        "valuation": w["valuation"],
                        "opening_nav": w["nav_before"],
                        "invested_after_fee": w["nav_after"],
                        "fees_usd": w["fees_usd"],
                        "closing_nav": w["closing_nav"],
                        **w["fee_breakdown"],
                    }
                )
                trades.extend(
                    {"portfolio": name, "monday": w["monday"], **t} for t in w["trades"]
                )
    for label in ["SPY", "QQQ"]:
        entry = quotes(prices, [label], plan[0]["execution"], "open")[label]
        book, fee = buy_and_hold_book(label, entry, INITIAL)
        nav = [INITIAL] + [
            book.value(quotes(prices, [label], w["valuation"], "close")) for w in plan
        ]
        name = f"{label} / IBKR entry fee"
        combined[name] = nav
        values = np.array(nav)
        metrics[name] = {
            "ending_value": nav[-1],
            "total_return": nav[-1] / INITIAL - 1,
            "max_weekly_drawdown": float(
                (values / np.maximum.accumulate(values) - 1).min()
            ),
            "total_fees_usd": fee,
        }
    assert sources == source_hashes(SOURCE)
    assert zero_cost == {name: sha256(ROOT / name) for name in zero_cost}
    combined.to_csv(OUTPUT / "weekly_nav.csv")
    pd.DataFrame(trades).to_csv(OUTPUT / "trades.csv", index=False)
    pd.DataFrame(weekly).to_csv(OUTPUT / "weekly_fees.csv", index=False)
    write_json(OUTPUT / "weekly_ledger.json", all_records)
    report = {
        "run_id": "scratch:patchtst-2025-snapshot-2026-ibkr-fees",
        "attempt": 1,
        "as_of": manifest["retrieved_at"],
        "run_at": datetime.now(UTC),
        "identity": identity,
        "first_execution": plan[0]["execution"],
        "last_valuation": plan[-1]["valuation"],
        "weeks": len(plan),
        "initial_capital": INITIAL,
        "metrics": metrics,
        "terminal_liquidation": False,
        "validation": {
            "weekly_books_reconciled": len(weekly),
            "trade_legs_verified": len(trades),
            "source_and_zero_fee_results_unchanged": True,
            "new_forecasts_or_hrp_fits": 0,
        },
        "limitations": [
            "Same retrospective current-roster biases and November 25 incomplete-week cutoff as original experiment.",
            "Repository fee rates held fixed across 2026; not verified historical or current broker quotations.",
            "Actual evolving NAV used for each rebalance; not the fixed USD 10000 training-time cost anchor.",
            "Adjusted-price units approximate historical per-share charges; FX, spread, slippage, tax and dividend withholding omitted.",
            "repo_default retains fee-free execution for weight changes below 0.005; every_trade charges all non-dust executions.",
            "Fees deducted before reinvestment; invested capital solves capital plus actual-leg fees equals pre-fee NAV. No new cash or terminal liquidation.",
        ],
        "output_sha256": {
            p.name: sha256(p) for p in OUTPUT.iterdir() if p.suffix in {".csv", ".json"}
        },
    }
    write_json(OUTPUT / "summary.json", report)
    print(
        json.dumps({"metrics": metrics, "validation": report["validation"]}, indent=2)
    )


if __name__ == "__main__":
    main()

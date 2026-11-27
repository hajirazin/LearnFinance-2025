"""Independently reconcile the real run's cash flows, rankings, and cutoffs."""

import json
import pickle

import numpy as np
import pandas as pd
from prepare import ROOT, sha256, write_json


def main():
    holdings = pd.read_csv(ROOT / "weekly_holdings.csv")
    trades = pd.read_csv(ROOT / "trades.csv")
    forecasts = pd.read_csv(ROOT / "forecasts.csv")
    nav = pd.read_csv(ROOT / "weekly_nav.csv", index_col="date")
    ledger = json.loads((ROOT / "weekly_ledger.json").read_text())
    summary = json.loads((ROOT / "summary.json").read_text())
    previous = {}
    max_residual = 0
    for week in ledger:
        monday = week["monday"][:10]
        held = holdings.loc[holdings["monday"].eq(monday)]
        transaction = trades.loc[trades["monday"].eq(monday)]
        scores = forecasts.loc[forecasts["monday"].eq(monday)].set_index("symbol")
        assert len(held) == held["symbol"].nunique() == 15
        assert (
            pd.to_datetime(scores["anchor"]) < pd.Timestamp(week["execution"])
        ).all()
        ranks = scores["rank"].to_dict()
        ranked = scores.sort_values(
            ["predicted_return", "symbol"], ascending=[False, True]
        ).index.tolist()
        assert all(ranks[s] == i + 1 for i, s in enumerate(ranked))
        for symbol in held["symbol"]:
            assert ranks[symbol] <= (20 if symbol in previous else 15)
        assert all(
            s in held["symbol"].tolist() for s in previous if ranks.get(s, 10**9) <= 20
        )
        opening = held["opening_value"].sum()
        ending = held["closing_value"].sum()
        assert np.allclose(held["opening_value"], opening / 15, atol=1e-8, rtol=0)
        assert np.allclose(
            held["quantity_adjusted_units"] * held["closing_price"],
            held["closing_value"],
            atol=1e-8,
            rtol=0,
        )
        assert np.isclose(
            ending, nav.loc[week["valuation"][:10], "PatchTST"], atol=1e-8, rtol=0
        )
        prices = transaction.set_index("symbol")["price"].to_dict()
        nav_before = (
            sum(quantity * prices[s] for s, quantity in previous.items())
            if previous
            else 10000
        )
        assert np.isclose(opening, nav_before, rtol=0, atol=1e-7)
        dollars = (transaction["quantity_delta"] * transaction["price"]).sum()
        residual = abs(dollars - (10000 if not previous else 0))
        assert residual < 1e-7
        max_residual = max(max_residual, residual)
        previous = held.set_index("symbol")["quantity_adjusted_units"].to_dict()
    manifest = json.loads((ROOT / "price_manifest.json").read_text())
    for label, symbol in {"SPY": "SPY", "QQQ": "QQQ", "S&P 500": "^GSPC"}.items():
        prices = pd.read_csv(ROOT / "data/prices" / f"{symbol}.csv", index_col="date")
        expected = (
            10000
            * prices.loc[nav.index[-1], "close"]
            / prices.loc[nav.index[0], "open"]
        )
        assert np.isclose(expected, nav[label].iloc[-1], atol=1e-8, rtol=0)
    with (ROOT / "data/inputs.pkl").open("rb") as handle:
        inputs = pickle.load(handle)
    causal_checks = 0
    for week in [ledger[0], ledger[len(ledger) // 2], ledger[-1]]:
        anchor = pd.Timestamp(week["anchor"]).date()
        symbol = week["selected"][0]
        before, _ = inputs.window(symbol, anchor)
        assert before is not None
        _, right = inputs.calendar.window(anchor, inputs.config.context_calendar_days)
        future = inputs.features[symbol][right:].copy()
        inputs.features[symbol][right:] = 999999
        after, _ = inputs.window(symbol, anchor)
        assert np.array_equal(before, after)
        inputs.features[symbol][right:] = future
        causal_checks += 1
    result = {
        "weekly_books_reconciled": len(ledger),
        "holdings_rows": len(holdings),
        "forecast_rows": len(forecasts),
        "future_input_perturbation_checks": causal_checks,
        "maximum_self_financing_residual_usd": max_residual,
        "benchmarks_reconciled": 3,
        "unfinished_daily_bar_excluded": summary["last_valuation"][:10] == "2026-11-25",
        "frozen_prices_verified": all(
            sha256(ROOT / "data/prices" / name) == digest
            for name, digest in manifest["prices_sha256"].items()
        ),
        "output_sha256": {
            p.name: sha256(p)
            for p in ROOT.iterdir()
            if p.suffix in {".csv", ".json"} and p.name != "validation.json"
        },
    }
    assert result["frozen_prices_verified"]
    assert result["unfinished_daily_bar_excluded"]
    write_json(ROOT / "validation.json", result)
    print(
        json.dumps({k: v for k, v in result.items() if k != "output_sha256"}, indent=2)
    )


if __name__ == "__main__":
    main()

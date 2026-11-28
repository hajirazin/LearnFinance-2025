"""Fee payments must reduce invested capital without creating new cash."""

import numpy as np
import pytest
from fee_accounting import fee_rebalance
from scratch.ppo_2025_model_2026_simulation_2026_11_07.book import buy_and_hold_book


def test_full_investment_pays_entry_fees_from_capital():
    shares, book = fee_rebalance({}, {"A": 1.0}, {"A": 100}, 10000, 1e-12)
    # $0.35 commission + commission pass-through + per-share clearing/CAT.
    expected_notional = (10000 - 0.35 * 1.00074) / (1 + 0.000233 / 100)
    assert shares["A"] == pytest.approx(expected_notional / 100)
    assert book["nav_after"] + book["fees_usd"] == pytest.approx(10000)
    assert book["cash_after"] == pytest.approx(0, abs=1e-7)
    reference, fee = buy_and_hold_book("A", 100, 10000)
    assert shares == pytest.approx(reference.shares)
    assert book["fees_usd"] == pytest.approx(fee)


def test_rotating_positions_includes_sell_fees_and_preserves_weekend_gap():
    shares, book = fee_rebalance(
        {"OLD": 100}, {"NEW": 1.0}, {"OLD": 120, "NEW": 40}, 0, 1e-12
    )
    assert book["nav_before"] == 12000
    assert shares["NEW"] * 40 + book["fees_usd"] == pytest.approx(12000)
    sells = [t for t in book["trades"] if t["side"] == "sell"]
    assert len(sells) == 1
    assert sells[0]["notional_usd"] == 12000
    assert sells[0]["sec_fee"] == pytest.approx(12000 * 0.0000206)
    assert sells[0]["finra_taf"] == pytest.approx(100 * 0.000195)
    assert sum(t["quantity_delta"] * t["price"] for t in book["trades"]) + book[
        "fees_usd"
    ] == pytest.approx(0, abs=1e-7)


def test_unchanged_weights_do_not_create_trades_or_fees():
    shares, book = fee_rebalance({"A": 100}, {"A": 1.0}, {"A": 120}, 0, 1e-12)
    assert shares == {"A": 100}
    assert book["fees_usd"] == 0
    assert not book["trades"]


def test_repo_threshold_exempts_small_trades_but_still_records_their_execution():
    prior = {"A": 50, "B": 50}
    weights, prices = {"A": 0.501, "B": 0.499}, {"A": 100, "B": 100}
    exempted, default = fee_rebalance(prior, weights, prices, 0, 0.005)
    charged, strict = fee_rebalance(prior, weights, prices, 0, 1e-12)
    assert default["fees_usd"] == 0
    assert len(default["trades"]) == 2
    assert all(t["total"] == 0 for t in default["trades"])
    assert strict["fees_usd"] > 0
    assert sum(exempted[s] * prices[s] for s in exempted) == pytest.approx(10000)
    assert sum(charged[s] * prices[s] for s in charged) < 10000


def test_fee_impact_scales_with_the_actual_account_value():
    _, small = fee_rebalance({}, {"A": 1.0}, {"A": 100}, 10000, 1e-12)
    _, large = fee_rebalance({}, {"A": 1.0}, {"A": 100}, 100000, 1e-12)
    assert large["fees_usd"] > small["fees_usd"]
    assert large["fees_usd"] / 100000 < small["fees_usd"] / 10000


@pytest.mark.parametrize("invalid_price", [0, np.nan, np.inf])
def test_missing_held_price_aborts_even_for_exiting_stock(invalid_price):
    with pytest.raises(ValueError, match="price"):
        fee_rebalance({"OLD": 3}, {"A": 1.0}, {"OLD": invalid_price, "A": 20}, 0, 1e-12)


def test_two_week_fee_replay_carries_fee_reduced_capital_through_weekend_gap():
    import pandas as pd
    from fee_run import simulate_fees

    days = pd.to_datetime(["2026-01-05", "2026-01-09", "2026-01-12", "2026-01-16"])
    prices = {
        s: pd.DataFrame(
            {"open": [100, 110, 220, 220], "close": [100, 110, 220, 220]}, index=days
        )
        for s in ["A", "B"]
    }
    plan = [
        {
            "monday": days[0],
            "execution": days[0],
            "valuation": days[1],
            "selected": ["A", "B"],
        },
        {
            "monday": days[2],
            "execution": days[2],
            "valuation": days[3],
            "selected": ["A", "B"],
        },
    ]
    nav, ledger, fees, residual = simulate_fees(
        prices, plan, [{"A": 0.5, "B": 0.5}] * 2, 1e-12
    )
    assert nav.tolist() == pytest.approx(
        [10000, (10000 - fees) * 1.1, (10000 - fees) * 2.2]
    )
    assert ledger[1]["fees_usd"] == 0
    assert ledger[1]["nav_before"] == pytest.approx(nav.iloc[-1])
    assert residual < 1e-7

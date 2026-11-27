"""Offline checks of money conservation, timing, and selection."""

from datetime import date

import numpy as np
import pandas as pd
import pytest
from accounting import (
    completed_weekly_schedule,
    equal_weight_rebalance,
    mark_to_market,
    weekly_schedule,
)

from brain_api.core.sticky_selection import select_with_rank_band


def test_weekend_gap_is_kept_and_rebalance_is_self_financing():
    friday_value = mark_to_market({"A": 10, "B": 20}, {"A": 110, "B": 50})
    assert friday_value == 2100
    opening = {"A": 121, "B": 45, "C": 10}
    shares, ledger = equal_weight_rebalance({"A": 10, "B": 20}, ["A", "C"], opening, 0)
    assert ledger["nav_before"] == 2110
    assert shares == {"A": 1055 / 121, "C": 105.5}
    assert mark_to_market(shares, opening) == pytest.approx(2110)
    assert sum(
        t["quantity_delta"] * t["price"] for t in ledger["trades"]
    ) == pytest.approx(0)


def test_first_buy_uses_exactly_starting_capital_and_fractional_shares():
    shares, ledger = equal_weight_rebalance({}, ["A", "B"], {"A": 30, "B": 40}, 10000)
    assert shares == {"A": 5000 / 30, "B": 125}
    assert ledger["nav_after"] == pytest.approx(10000)
    assert ledger["cash_after"] == 0


@pytest.mark.parametrize("price", [None, np.nan, np.inf, 0, -1])
def test_missing_or_invalid_held_mark_aborts(price):
    with pytest.raises(ValueError, match="price"):
        equal_weight_rebalance({"OLD": 1}, ["NEW"], {"OLD": price, "NEW": 20}, 0)


def test_rank_band_keeps_rank_20_evicts_rank_21_and_fills_to_15():
    scores = {f"S{i:02}": -float(i) for i in range(1, 26)}
    previous = {f"S{i:02}" for i in range(7, 22)}
    selection = select_with_rank_band(scores, previous, 15, 20)
    assert len(selection.selected) == 15
    assert "S20" in selection.selected
    assert "S21" not in selection.selected
    assert "S01" in selection.selected
    assert selection.kept_count == 14


def test_calendar_handles_monday_friday_holidays_and_final_complete_week():
    sessions = pd.bdate_range("2026-01-01", "2026-11-30").difference(
        pd.to_datetime(["2026-01-01", "2026-01-19", "2026-04-03", "2026-11-26"])
    )
    rows = weekly_schedule(sessions, date(2026, 1, 5), date(2026, 11, 28))
    assert len(rows) == 47
    assert rows[0]["anchor"] == pd.Timestamp("2026-01-02")
    holiday = next(row for row in rows if row["monday"] == pd.Timestamp("2026-01-19"))
    assert holiday["execution"] == pd.Timestamp("2026-01-20")
    assert holiday["anchor"] == pd.Timestamp("2026-01-16")
    good_friday = next(
        row for row in rows if row["monday"] == pd.Timestamp("2026-03-30")
    )
    assert good_friday["valuation"] == pd.Timestamp("2026-04-02")
    assert rows[-1]["valuation"] == pd.Timestamp("2026-11-27")


def test_unfinished_early_close_bar_is_not_a_friday_closing_price():
    sessions = pd.to_datetime(
        ["2026-11-20", "2026-11-23", "2026-11-24", "2026-11-25", "2026-11-27"]
    )
    closes = pd.Series(
        pd.to_datetime(
            [
                "2026-11-20T21:00Z",
                "2026-11-23T21:00Z",
                "2026-11-24T21:00Z",
                "2026-11-25T21:00Z",
                "2026-11-27T18:00Z",
            ]
        ),
        index=sessions,
    )
    rows = completed_weekly_schedule(
        closes, date(2026, 11, 23), date(2026, 11, 28), "2026-11-27T17:52Z"
    )
    assert rows[-1]["valuation"] == pd.Timestamp("2026-11-25")
    assert not rows[-1]["week_complete"]
    finished = completed_weekly_schedule(
        closes, date(2026, 11, 23), date(2026, 11, 28), "2026-11-27T18:01Z"
    )
    assert finished[-1]["valuation"] == pd.Timestamp("2026-11-27")
    assert finished[-1]["week_complete"]


def test_two_week_simulation_carries_actual_positions_across_weekend(
    monkeypatch, tmp_path
):
    import replay

    monkeypatch.setattr(replay, "ROOT", tmp_path)
    sessions = pd.bdate_range("2026-01-02", "2026-01-16")
    plan = weekly_schedule(sessions, date(2026, 1, 5), date(2026, 1, 16))
    symbols = [f"S{i:02}" for i in range(15)]
    prices = {
        s: pd.DataFrame(
            {"open": [10, 22], "close": [11, 22]},
            index=pd.to_datetime(["2026-01-05", "2026-01-12"]),
        )
        for s in symbols
    }
    for frame in prices.values():
        frame.loc[pd.Timestamp("2026-01-09")] = [11, 11]
        frame.loc[pd.Timestamp("2026-01-16")] = [22, 22]
    for s in ("SPY", "QQQ", "^GSPC"):
        prices[s] = pd.DataFrame({"open": 100, "close": 100}, index=sessions)
    forecasts = pd.DataFrame(
        [
            {"monday": week["monday"], "symbol": s, "predicted_return": 0.1}
            for week in plan
            for s in symbols
        ]
    )
    nav, ledger = replay.simulate(prices, forecasts, plan)
    assert nav["PatchTST"].tolist() == pytest.approx([10000, 11000, 22000])
    assert ledger[1]["monday_open_nav"] == pytest.approx(22000)
    assert ledger[1]["weekend_return"] == pytest.approx(1)
    assert nav["SPY"].tolist() == [10000, 10000, 10000]

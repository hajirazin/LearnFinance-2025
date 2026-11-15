"""Test cash, fees, and the holding period in the V1 comparison."""

import unittest

import pandas as pd
from v1_holdout_portfolio import (
    INITIAL_NAV,
    allocation_path,
    rebalance,
    simulate,
    transaction_fees,
)


class V1HoldoutPortfolioTest(unittest.TestCase):
    def test_equal_weight_keeps_sticky_names_and_assigns_one_fifteenth(self):
        days = pd.to_datetime(["2026-10-02", "2026-10-09"])
        symbols = [f"S{i:02}" for i in range(21)]
        index = pd.MultiIndex.from_product([days, symbols], names=["friday", "symbol"])
        second_week = [*range(21, 7, -1), 2, *range(7, 2, -1), 1]
        scores = pd.Series([*range(21, 0, -1), *second_week], index=index)
        first, second = allocation_path(scores, None, equal_weight=True)
        self.assertEqual(set(first[1].index), set(symbols[:15]))
        self.assertEqual(set(second[1].index), set(symbols[:15]))
        self.assertEqual(second[2].reasons[symbols[14]], "sticky")
        self.assertTrue(second[1].eq(1 / 15).all())

    def test_both_sides_pay_slippage_and_cat_and_only_sell_pays_sec_and_taf(self):
        delta = pd.Series({"AAA": -5000.0, "BBB": 5000.0})
        opens = pd.Series({"AAA": 10.0, "BBB": 20.0})
        slippage, regulatory = transaction_fees(delta, opens, 10)
        self.assertAlmostEqual(slippage, 10.0)
        self.assertAlmostEqual(regulatory, 0.22)

    def test_initial_buy_is_self_financing(self):
        weights = pd.Series({"AAA": 0.7, "BBB": 0.3})
        invested, delta, slippage, regulatory, cash = rebalance(
            pd.Series(dtype=float),
            weights,
            pd.Series({"AAA": 10, "BBB": 20}),
            INITIAL_NAV,
            10,
        )
        self.assertAlmostEqual(invested.sum(), (INITIAL_NAV - 0.01) / 1.001)
        self.assertAlmostEqual(
            invested.sum() + slippage + regulatory + cash, INITIAL_NAV
        )
        self.assertTrue(delta.gt(0).all())
        self.assertAlmostEqual(invested["AAA"] / invested.sum(), 0.7)

    def test_first_trade_occurs_after_friday_close_and_marks_terminal_open(self):
        dates = pd.to_datetime(["2026-10-02", "2026-10-05", "2026-10-06", "2026-10-12"])
        prices = pd.DataFrame({"AAA": [10.0, 20.0, 22.0, 24.0]}, index=dates)
        metrics, ledger, nav, orders = simulate(
            [(dates[0], pd.Series({"AAA": 1.0}), None)],
            prices,
            prices,
            prices,
            dates[-1],
            0,
        )
        self.assertEqual(ledger.loc[0, "fill"], dates[1])
        self.assertAlmostEqual(metrics["net_return"], 0.1999988)
        self.assertEqual(nav.iloc[-1]["mark"], "terminal_open")
        self.assertAlmostEqual(orders.iloc[0]["absolute_shares"], 499.9995)

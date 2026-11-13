"""Check the historical carry and outcome rules for sticky ranks."""

import unittest
from datetime import date, timedelta

from compare_sticky_rank import MODELS, replay


class StickyRankReplayTest(unittest.TestCase):
    def test_missing_outcome_does_not_interrupt_carry(self):
        start = date(2025, 1, 3)
        fridays = [(start + timedelta(days=7 * i)).isoformat() for i in range(3)]
        forecasts = {}
        for friday, rank_of_previous in zip(fridays, (1, 2, 2), strict=True):
            forecasts[friday] = {
                "A": {**dict.fromkeys(MODELS, 3 - rank_of_previous), "actual": None},
                "B": {**dict.fromkeys(MODELS, rank_of_previous), "actual": 0.01},
            }
        weekly, picks = replay(forecasts, top_n=1, hold_threshold=2)
        strict = [
            r for r in weekly if r["model"] == "ridge" and r["selector"] == "strict"
        ]
        sticky = [
            r for r in weekly if r["model"] == "ridge" and r["selector"] == "sticky"
        ]
        self.assertEqual([r["actual_return"] for r in strict], [None, 0.01, 0.01])
        self.assertEqual([r["actual_return"] for r in sticky], [None, None, None])
        self.assertEqual([r["fraction_replaced"] for r in sticky], [None, 0.0, 0.0])
        self.assertTrue(all(not r["matched"] for r in weekly))
        held = [r for r in picks if r["model"] == "ridge" and r["selector"] == "sticky"]
        self.assertEqual([r["symbol"] for r in held], ["A", "A", "A"])
        self.assertEqual([r["predicted_rank"] for r in held], [1, 2, 2])

    def test_previously_selected_stock_is_evicted_outside_hold_band(self):
        forecasts = {}
        for friday, values in (("2025-01-03", (3, 2, 1)), ("2025-01-17", (1, 2, 3))):
            forecasts[friday] = {
                symbol: {**dict.fromkeys(MODELS, value), "actual": value / 100}
                for symbol, value in zip("ABC", values, strict=True)
            }
        weekly, picks = replay(forecasts, top_n=1, hold_threshold=2)
        self.assertTrue(all(r["matched"] for r in weekly))
        held = [r for r in picks if r["model"] == "ridge" and r["selector"] == "sticky"]
        self.assertEqual([r["symbol"] for r in held], ["A", "C"])

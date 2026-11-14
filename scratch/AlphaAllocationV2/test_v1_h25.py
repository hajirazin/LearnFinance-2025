"""Check that ranking and hindsight baskets use the intended information."""

import unittest

import numpy as np
import pandas as pd
from run_v1_h25 import oracle_scores, rank_scores


class H25ReplayTest(unittest.TestCase):
    def test_previous_membership_comes_from_the_model_own_top15(self):
        dates = pd.to_datetime(["2026-01-09", "2026-01-16"])
        symbols = [f"S{i:02}" for i in range(20)]
        index = pd.MultiIndex.from_product([dates, symbols], names=["friday", "symbol"])
        x = np.tile(np.arange(20, dtype=np.float32), 2)[:, None, None]
        flags = []

        def model(inputs, _mask, previous):
            flags.append(previous.tolist())
            return inputs[:, 0, 0]

        scores = rank_scores(index, x, np.ones_like(x, dtype=bool), model, "cpu")
        self.assertEqual(flags, [[0] * 20, [0] * 5 + [1] * 15])
        self.assertEqual(scores.loc[dates[1]].idxmax(), "S19")

    def test_hindsight_returns_use_the_next_execution_open(self):
        decisions = pd.to_datetime(["2026-10-30"])
        dates = pd.to_datetime(["2026-10-30", "2026-11-02", "2026-11-09"])
        symbols = [f"S{i:02}" for i in range(16)]
        index = pd.MultiIndex.from_product(
            [decisions, symbols], names=["friday", "symbol"]
        )
        opens = pd.DataFrame(
            [np.repeat(50, 16), np.repeat(100, 16), np.arange(100, 116)],
            index=dates,
            columns=symbols,
        )
        returns = oracle_scores(
            pd.DataFrame(index=index),
            decisions,
            dict.fromkeys(("open", "raw_open", "close"), opens),
        )
        self.assertAlmostEqual(returns.loc[(decisions[0], "S15")], 0.15)
        self.assertEqual(returns.loc[(decisions[0], "S00")], 0)

"""Test one-at-a-time sentiment counterfactual selection."""

import unittest

import numpy as np
import pandas as pd
from run_v1_h28 import compare_selection, select
from train_v1_h28 import replace_price_with_return
from v1_hypothesis_data import SEVEN


class H28SelectionTest(unittest.TestCase):
    def test_other_candidates_stay_in_the_universe(self):
        symbols = [f"S{i:02d}" for i in range(25)]
        baseline = pd.Series(np.arange(25)[::-1], index=symbols)
        previous = {*symbols[:14], symbols[19]}
        index = pd.MultiIndex.from_product(
            [[pd.Timestamp("2026-01-09")], [symbols[19]]],
            names=["friday", "symbol"],
        )
        cases = pd.DataFrame(
            {"neutral_score": [5], "negative_score": [-1]}, index=index
        )
        compared = compare_selection(cases, baseline, previous).iloc[0]
        self.assertEqual(compared.neutral_rank, 20)
        self.assertEqual(compared.negative_rank, 25)
        self.assertTrue(compared.original_selected)
        self.assertTrue(compared.removed)

    def test_matches_full_selector_for_each_candidate_with_ties(self):
        symbols = [f"S{i:02d}" for i in range(28)]
        rng = np.random.default_rng(42)
        baseline = pd.Series(rng.integers(0, 7, len(symbols)), index=symbols)
        group = symbols[::2]
        previous = set(symbols[7:22])
        index = pd.MultiIndex.from_product(
            [[pd.Timestamp("2026-01-09")], group], names=["friday", "symbol"]
        )
        cases = pd.DataFrame(
            {
                "neutral_score": rng.integers(0, 7, len(group)),
                "negative_score": rng.integers(0, 7, len(group)),
            },
            index=index,
        )
        compared = compare_selection(cases, baseline, previous)
        for (_, symbol), row in compared.iterrows():
            for case in ("neutral", "negative"):
                counterfactual = baseline.copy()
                counterfactual.loc[symbol] = row[f"{case}_score"]
                ranked = counterfactual.sort_values(ascending=False, kind="stable")
                picked = select(counterfactual.to_numpy(), symbols, previous)
                self.assertEqual(row[f"{case}_rank"], ranked.index.get_loc(symbol) + 1)
                self.assertEqual(row[f"{case}_selected"], symbol in picked)


class H28CloseInputTest(unittest.TestCase):
    def test_replaces_only_encoder_price_and_preserves_source(self):
        x = np.arange(48, dtype=np.float32).reshape(2, 3, 8)
        original = x.copy()
        mask = np.ones_like(x, dtype=bool)
        changed = replace_price_with_return(x, mask)
        price = SEVEN.index("close")
        self.assertEqual(changed.dtype, x.dtype)
        np.testing.assert_array_equal(changed[..., price], x[..., -1])
        np.testing.assert_array_equal(
            np.delete(changed, price, axis=-1), np.delete(x, price, axis=-1)
        )
        np.testing.assert_array_equal(x, original)

    def test_rejects_different_price_and_return_coverage(self):
        x = np.zeros((1, 2, 8), dtype=np.float32)
        mask = np.ones_like(x, dtype=bool)
        mask[0, 0, -1] = False
        with self.assertRaisesRegex(ValueError, "same coverage"):
            replace_price_with_return(x, mask)


if __name__ == "__main__":
    unittest.main()

"""Test the realized top-15 comparison against known rankings."""

import unittest

import pandas as pd
from compare_v1_rank_market import ranking_week


class RankingWeekTest(unittest.TestCase):
    def test_oracle_hits_and_perfect_rank_correlation(self):
        actual = pd.Series({f"S{i:02}": -i / 100 for i in range(30)})
        sticky = list(actual.index[:14]) + ["S20"]
        result = ranking_week(actual, actual, sticky)
        self.assertAlmostEqual(result["rank_ic"], 1.0)
        self.assertEqual(result["strict_top15_hits"], 15)
        self.assertEqual(result["sticky_top15_hits"], 14)
        self.assertEqual(result["random_expected_hits"], 7.5)

    def test_missing_selected_outcome_excludes_week(self):
        scores = pd.Series({f"S{i:02}": -i / 100 for i in range(30)})
        actual = scores.drop("S00")
        self.assertIsNone(ranking_week(scores, actual, list(scores.index[:15])))

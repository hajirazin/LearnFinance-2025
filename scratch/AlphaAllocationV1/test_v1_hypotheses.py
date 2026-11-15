"""Check daily context, masked inputs and multiple-test decisions."""

import unittest

import numpy as np
import pandas as pd
import torch
from patchtst_replay_models import model_output
from run_v1_hypotheses import holm, one_sided_test
from train_v1_hypotheses import model_factory
from v1_hypothesis_data import windows


class V1HypothesisTest(unittest.TestCase):
    def test_holm_adjusts_four_tests_and_preserves_monotonicity(self):
        result = holm({"a": 0.01, "b": 0.02, "c": 0.021, "d": 0.4})
        self.assertEqual(result, {"a": 0.04, "b": 0.06, "c": 0.06, "d": 0.4})

    def test_bootstrap_p_value_is_one_sided(self):
        x = np.linspace(0.01, 0.03, 40)
        self.assertLess(one_sided_test(x)[0], 0.05)
        self.assertGreater(one_sided_test(-x)[0], 0.95)

    def test_features_are_masked_and_daily_targets_are_unchanged(self):
        dates = pd.bdate_range("2024-01-02", periods=100)
        log_returns = pd.DataFrame(
            {"AAA": np.arange(100, dtype=float) / 1000}, index=dates
        )
        features = np.ones((100, 1, 7), dtype=np.float32)
        features[50, 0, 2] = np.nan
        anchor = dates[[89]]
        _, x, mask, y = windows(
            log_returns, features, anchor, use_features=True, year=False
        )
        _, close_x, _, close_y = windows(
            log_returns, None, anchor, use_features=False, year=False
        )
        np.testing.assert_array_equal(y, close_y)
        self.assertEqual(x.shape, (1, 60, 8))
        np.testing.assert_array_equal(x[:, :, 7:], close_x)
        self.assertFalse(mask[0, 20, 2])
        self.assertEqual(x[0, 20, 2], 1.0)
        self.assertAlmostEqual(float(y[0, 0]), 0.09)
        model = model_factory(x, daily=True).eval()
        with torch.inference_mode():
            output = model_output(
                model, torch.from_numpy(x), torch.from_numpy(mask), daily=True
            )
        self.assertEqual(output.shape, (1, 5))
        self.assertTrue(torch.isfinite(output).all())

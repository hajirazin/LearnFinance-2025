"""Protect shared inputs, causal folds, and equal-week checkpoint selection."""

import unittest

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import torch
from report_v1_xgboost_comparison import align_predictions, selected_baskets
from train_v1_xgboost_comparison import (
    make_patchtst,
    purged_dates,
    shared_inputs,
    weekly_weights,
)
from v1_hypothesis_data import windows


class XGBoostComparisonTest(unittest.TestCase):
    def test_split_labels_stay_inside_their_years(self):
        sessions = pd.bdate_range("2020-01-01", "2026-01-20")
        train, validation = purged_dates(sessions)
        train_ends = sessions[sessions.get_indexer(train) + 5]
        val_ends = sessions[sessions.get_indexer(validation) + 5]
        self.assertTrue((train_ends < pd.Timestamp("2025-01-01")).all())
        self.assertTrue((validation >= pd.Timestamp("2025-01-01")).all())
        self.assertTrue((val_ends < pd.Timestamp("2026-01-01")).all())

    def test_masked_and_auxiliary_values_do_not_enter_shared_inputs(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(2, 255, 8)).astype(np.float32)
        mask = np.ones_like(x, dtype=bool)
        mask[:, :10] = False
        before = shared_inputs(x, mask)
        x[:, :10] = 1e6
        x[..., 7] = 1e4
        after = shared_inputs(x, mask)
        self.assertEqual(before.shape, (2, 255, 7))
        np.testing.assert_array_equal(before, after)

    def test_validation_weights_give_each_week_equal_influence(self):
        index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2025-01-03"), "A"),
                (pd.Timestamp("2025-01-10"), "A"),
                (pd.Timestamp("2025-01-10"), "B"),
            ],
            names=["friday", "symbol"],
        )
        weights = weekly_weights(index)
        errors_squared = np.array([4.0, 1.0, 1.0])
        self.assertAlmostEqual(np.average(errors_squared, weights=weights), 2.5)
        self.assertAlmostEqual(weights.mean(), 1)

    def test_patchtst_accepts_only_seven_channels_and_returns_one_target(self):
        model = make_patchtst().eval()
        self.assertEqual(model.config.num_input_channels, 7)
        self.assertEqual(model.config.num_targets, 1)
        self.assertIsNone(model.config.scaling)
        with torch.inference_mode():
            prediction = model(past_values=torch.zeros(2, 255, 7)).regression_outputs
        self.assertEqual(prediction.shape, (2, 1))
        self.assertTrue(torch.isfinite(prediction).all())

    def test_future_observations_cannot_change_earlier_inputs(self):
        sessions = xcals.get_calendar("XNYS").sessions_in_range(
            "2022-01-03", "2023-12-29"
        )
        logs = pd.DataFrame({"A": np.full(len(sessions), 0.001)}, index=sessions)
        features = np.ones((len(sessions), 1, 7), dtype=np.float32)
        anchor = sessions[[300]]
        _, x, mask, _ = windows(logs, features, anchor, use_features=True, year=True)
        before = shared_inputs(x, mask)
        logs.iloc[301:] = -1
        features[301:] = -100
        _, x, mask, _ = windows(logs, features, anchor, use_features=True, year=True)
        np.testing.assert_array_equal(before, shared_inputs(x, mask))

    def test_prediction_pairing_is_keyed_and_missing_forecasts_fail(self):
        index = pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2026-01-09"), "A"), (pd.Timestamp("2026-01-09"), "B")],
            names=["friday", "symbol"],
        )
        actual = pd.DataFrame({"actual": [0.1, -0.1]}, index=index)
        scores = pd.DataFrame({"XGBoost": [0.02, 0.03]}, index=index)
        pd.testing.assert_frame_equal(
            align_predictions(actual, scores),
            align_predictions(actual, scores.iloc[::-1]),
        )
        with self.assertRaises(ValueError):
            align_predictions(actual, scores.iloc[:1])

    def test_selection_uses_forecasts_without_realized_outcomes(self):
        index = pd.MultiIndex.from_product(
            [[pd.Timestamp("2026-01-09")], [f"S{i:02}" for i in range(25)]],
            names=["friday", "symbol"],
        )
        scores = pd.DataFrame(
            {
                arm: np.arange(25, dtype=float)
                for arm in ("Ridge", "XGBoost", "PatchTST")
            },
            index=index,
        )
        for names in selected_baskets(scores).values():
            self.assertEqual(set(names), {f"S{i:02}" for i in range(10, 25)})


if __name__ == "__main__":
    unittest.main()

"""Check how stock forecasts are carried into the top-15 basket evaluation."""

import unittest

import numpy as np
import pandas as pd
from compare_feature_sets import TOP_K, evaluate, forecast_accuracy


class ForecastEvaluationTest(unittest.TestCase):
    def test_basket_uses_predicted_top_15_even_if_a_label_is_missing(self):
        fridays = pd.date_range("2025-01-03", periods=2, freq="W-FRI")
        index = pd.MultiIndex.from_product(
            [fridays, [f"S{i:02}" for i in range(TOP_K + 1)]],
            names=["friday", "symbol"],
        )
        predicted = np.arange(TOP_K + 1, 0, -1) / 1000
        actual = np.arange(TOP_K + 1) / 1000
        forecasts = pd.DataFrame(
            {
                "six": np.tile(predicted, len(fridays)),
                "eight": np.tile(predicted[::-1], len(fridays)),
                "forward_return": np.tile(actual, len(fridays)),
                "next_friday": index.get_level_values("friday") + pd.Timedelta(days=7),
            },
            index=index,
        )
        forecasts.loc[(fridays[1], "S00"), "forward_return"] = np.nan

        weekly, picks, stocks = evaluate(forecasts, {"six": (), "eight": ()})

        self.assertEqual(len(stocks), len(forecasts) * 2)
        self.assertEqual(len(picks), len(fridays) * TOP_K * 2)
        np.testing.assert_allclose(
            weekly["six_top15_predicted_return"], predicted[:TOP_K].mean()
        )
        self.assertAlmostEqual(
            weekly.loc[fridays[0], "six_top15_return"], actual[:TOP_K].mean()
        )
        self.assertAlmostEqual(
            weekly.loc[fridays[0], "six_top15_prediction_error"],
            (predicted[:TOP_K] - actual[:TOP_K]).mean(),
        )
        self.assertTrue(pd.isna(weekly.loc[fridays[1], "six_top15_return"]))
        self.assertTrue(pd.isna(weekly.loc[fridays[1], "six_top15_prediction_error"]))
        self.assertTrue(pd.notna(weekly.loc[fridays[1], "eight_top15_return"]))
        self.assertEqual(weekly["paired"].tolist(), [True, False])
        missing = stocks.loc[stocks["forward_return"].isna()]
        self.assertTrue(missing["predicted_return"].notna().all())
        self.assertTrue(missing["prediction_error"].isna().all())

    def test_forecast_accuracy_uses_observed_pairs(self):
        result = forecast_accuracy(
            pd.Series([0.01, 0.02, 0.99]), pd.Series([0.04, 0.01, np.nan])
        )

        self.assertEqual(result["observations"], 2)
        self.assertAlmostEqual(result["mean_predicted_return"], 0.015)
        self.assertAlmostEqual(result["mean_actual_return"], 0.025)
        self.assertAlmostEqual(result["mean_error"], -0.01)
        self.assertAlmostEqual(result["mae"], 0.02)
        self.assertAlmostEqual(result["rmse"], np.sqrt(0.0005))
        self.assertAlmostEqual(result["zero_forecast_mae"], 0.025)

"""Check target alignment and the seven-target forecast used for ranking."""

import unittest

import numpy as np
import pandas as pd
import torch
from compare_feature_sets import SEVEN
from train_v1_multitarget import MultiTargetPatchTST, close_forecast, future_targets


class MultiTargetTest(unittest.TestCase):
    def test_weekly_target_uses_friday_close_in_a_short_week(self):
        sessions = pd.to_datetime(
            [
                "2025-01-17",
                "2025-01-21",
                "2025-01-22",
                "2025-01-23",
                "2025-01-24",
                "2025-01-27",
            ]
        )
        index = pd.MultiIndex.from_tuples(
            [(sessions[0], "AAA")], names=["friday", "symbol"]
        )
        logs = pd.DataFrame(
            {"AAA": [0.0, 0.01, 0.02, -0.01, 0.02, 0.20]}, index=sessions
        )
        features = np.arange(len(sessions) * len(SEVEN), dtype=np.float32).reshape(
            len(sessions), 1, len(SEVEN)
        )
        daily, weekly, exact_five = future_targets(index, logs, features)
        self.assertFalse(exact_five[0])
        self.assertAlmostEqual(weekly[0, 0, 0].item(), 0.04)
        self.assertAlmostEqual(daily[0, -1, 0].item(), 0.20)
        self.assertEqual(weekly[0, 0, 1], features[4, 0, 0])
        self.assertAlmostEqual(
            weekly[0, 0, -1].item(), np.log1p(features[4, 0, -1]), places=6
        )

    def test_ranking_uses_only_the_close_channel(self):
        output = torch.full((1, 5, 7), 100.0)
        output[..., 0] = 1
        score = close_forecast(output, daily=True, mean=0.005, scale=0.01)
        torch.testing.assert_close(score, torch.expm1(torch.tensor([0.075])))

    def test_model_forecasts_seven_series_at_each_horizon(self):
        x = torch.zeros(2, 60, 8)
        mask = torch.ones_like(x, dtype=torch.bool)
        for horizon in (1, 5):
            with self.subTest(horizon=horizon):
                model = MultiTargetPatchTST(60, horizon).eval()
                with torch.inference_mode():
                    self.assertEqual(model(x, mask).shape, (2, horizon, 7))

"""Regression tests for the matched Friday replay."""

import unittest

import numpy as np
import pandas as pd
import torch
from patchtst_replay_data import (
    ReplayData,
    close_windows,
    comparable_fridays,
    monthly_folds,
    weekly_windows,
)
from patchtst_replay_models import build_model, model_output, simple_returns


class MatchedFridayReplayTest(unittest.TestCase):
    def test_year_uses_calendar_days_and_ends_at_the_decision(self):
        sessions = pd.bdate_range("2023-12-01", "2025-01-17")
        friday = pd.Timestamp("2025-01-10")
        logs = pd.DataFrame(
            {"AAA": np.arange(len(sessions), dtype=float)}, index=sessions
        )
        x, mask, y = close_windows(logs, pd.DatetimeIndex([friday]), 264)
        expected = logs.loc[
            (sessions > friday - pd.Timedelta(days=365)) & (sessions <= friday), "AAA"
        ]
        np.testing.assert_array_equal(x[0, 0, mask[0]], expected)
        np.testing.assert_array_equal(y[0, 0], logs.loc[friday:].iloc[1:6, 0])
        self.assertTrue(np.isnan(x[0, 0, ~mask[0]]).all())

    def test_only_fridays_with_a_matching_five_session_horizon(self):
        sessions = pd.bdate_range("2025-01-10", "2025-01-31").drop("2025-01-20")
        fridays = sessions[sessions.day_of_week == 4]
        self.assertEqual(
            comparable_fridays(sessions, fridays).tolist(), [fridays[0], fridays[2]]
        )

    def test_weekly_context_ends_at_cutoff_and_keeps_missing_values(self):
        x = np.arange(28, dtype=float).reshape(14, 2, 1)
        x[3, 1, 0] = np.nan
        windows = weekly_windows(x, 12)
        self.assertTrue(np.isnan(windows[:11]).all())
        np.testing.assert_array_equal(windows[11, 0, :, 0], x[:12, 0, 0])
        self.assertTrue(np.isnan(windows[11, 1, 3, 0]))
        self.assertEqual(windows[12, 0, -1, 0], x[12, 0, 0])

    def test_daily_logs_compound_and_weekly_scalar_is_already_simple(self):
        daily = torch.log1p(torch.tensor([[0.10, -0.10], [0.01, 0.02]]))
        np.testing.assert_allclose(
            simple_returns(daily, daily=True), [-0.01, 0.0302], atol=1e-7
        )
        weekly = torch.tensor([[0.02], [-0.03]])
        np.testing.assert_allclose(simple_returns(weekly, daily=False), [0.02, -0.03])

    def test_monthly_fit_excludes_labels_ending_at_the_refit(self):
        fridays = pd.date_range("2024-10-04", periods=17, freq="W-FRI")
        index = pd.MultiIndex.from_product(
            [fridays, ["AAA"]], names=["friday", "symbol"]
        )
        labels = pd.DataFrame(
            {"next_friday": fridays + pd.Timedelta(days=7), "forward_return": 0.02},
            index=index,
        )
        data = ReplayData(
            labels=labels,
            weekly_x=np.empty((0, 12, 7)),
            daily_x=np.empty((0, 255, 1)),
            daily_mask=np.empty((0, 255, 1), dtype=bool),
            daily_y=np.zeros((len(fridays), 5)),
            coverage=pd.DataFrame(),
            evaluation_dates=fridays[-2:],
        )
        cutoff, historical, train, validation, score = next(monthly_folds(data))
        self.assertEqual(cutoff, fridays[-2])
        self.assertTrue(labels.loc[historical, "next_friday"].lt(cutoff).all())
        self.assertFalse(historical[-3])
        self.assertTrue(score[-2:].all())
        first_validation = labels.index[validation].get_level_values("friday").min()
        self.assertTrue(labels.loc[train, "next_friday"].lt(first_validation).all())

    def test_patchtst_output_matches_each_loss_target(self):
        daily = torch.zeros((2, 255, 1))
        daily_mask = torch.ones_like(daily, dtype=torch.bool)
        daily_mask[:, :5] = False
        daily_model = build_model(daily, daily=True).eval()
        weekly = torch.zeros((2, 12, 7))
        weekly_model = build_model(weekly, daily=False).eval()
        with torch.inference_mode():
            daily_output = model_output(daily_model, daily, daily_mask, daily=True)
            weekly_output = model_output(weekly_model, weekly, None, daily=False)
        self.assertEqual(daily_output.shape, (2, 5))
        self.assertEqual(weekly_output.shape, (2, 1))
        self.assertTrue(torch.isfinite(daily_output).all())
        self.assertTrue(torch.isfinite(weekly_output).all())

"""Protect temporal separation and the week-level comparison rules."""

from datetime import date

import numpy as np
import pandas as pd
import pytest
from dataset import split_panel
from reporting import block_interval, decision, weekly_metrics

from brain_api.core.patchtst.weekly_data import WeeklyPatchTSTData


def test_outer_evaluation_cannot_enter_model_selection():
    anchors = np.asarray([date(2024, 12, 20), date(2024, 12, 27), date(2025, 1, 3)])
    ends = np.asarray([date(2024, 12, 30), date(2025, 1, 6), date(2025, 1, 10)])
    panel = WeeklyPatchTSTData(
        np.zeros((3, 1, 1)), np.zeros(3), anchors, ends, np.asarray(["A"] * 3)
    )
    development, evaluation = split_panel(panel)
    assert development.anchor_dates.tolist() == [date(2024, 12, 20)]
    assert evaluation.anchor_dates.tolist() == [date(2025, 1, 3)]


def test_paired_bootstrap_preserves_constant_effect_and_decision():
    interval = block_interval(np.full(52, 0.01), 4)
    assert interval["low"] == pytest.approx(0.01)
    assert interval["high"] == pytest.approx(0.01)
    assert decision(interval, 1.0).startswith("Promising")
    assert decision(interval, 1.03).startswith("Inconclusive")
    assert decision(block_interval(np.full(52, -0.01), 4), 1.0).startswith("Does not")


def test_weekly_ranking_and_equal_weight_basket_metrics():
    frame = pd.DataFrame(
        {
            "anchor": ["2025-01-03"] * 30,
            "symbol": [f"S{i:02d}" for i in range(30)],
            "actual": np.arange(30) / 100,
        }
    )
    result = weekly_metrics(frame, np.arange(30) / 100).iloc[0]
    assert result.rank_ic == pytest.approx(1.0)
    assert result.mse == 0
    assert result.top15_excess == pytest.approx(0.075)
    assert result.top_bottom_spread == pytest.approx(0.15)
    assert result.selected.split(",")[0] == "S29"

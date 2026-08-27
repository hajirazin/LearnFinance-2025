"""Training-time price masking for ppo_discovery."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from brain_api.core.ppo_discovery.pipeline import _ohlcv_for_training
from brain_api.core.ppo_discovery.schemas import PPODiscoveryError
from brain_api.core.ppo_discovery.weeks import news_window_starts_at_or_after_archive


def test_missing_stock_is_omitted_but_spy_and_vix_are_required() -> None:
    spy = pd.DataFrame({"close": [1.0]})
    vix = pd.DataFrame({"close": [2.0]})
    aapl = pd.DataFrame({"close": [3.0]})
    ohlcv = _ohlcv_for_training(
        {"SPY": spy, "^VIX": vix, "AAPL": aapl},
        ["AAPL", "MSFT"],
    )
    assert set(ohlcv) == {"AAPL"}
    with pytest.raises(PPODiscoveryError, match="missing yfinance frames"):
        _ohlcv_for_training({"AAPL": aapl, "^VIX": vix}, ["AAPL"])
    with pytest.raises(PPODiscoveryError, match=r"\^VIX"):
        _ohlcv_for_training({"AAPL": aapl, "SPY": spy}, ["AAPL"])


def test_cutoffs_before_news_archive_are_skipped() -> None:
    assert (
        news_window_starts_at_or_after_archive(
            datetime(2014, 12, 26, 20, 0, tzinfo=UTC)
        )
        is False
    )
    assert (
        news_window_starts_at_or_after_archive(datetime(2015, 1, 2, 20, 0, tzinfo=UTC))
        is False
    )
    assert (
        news_window_starts_at_or_after_archive(datetime(2015, 1, 9, 20, 0, tzinfo=UTC))
        is True
    )

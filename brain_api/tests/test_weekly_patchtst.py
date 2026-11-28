"""The active US and India recipes emit one five-session return."""

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest
import torch

from brain_api.core.patchtst.calendar import (
    inference_calendar,
    load_patchtst_calendar,
    training_calendar,
)
from brain_api.core.patchtst.config import (
    INDIA_WEEKLY_CONFIG,
    US_WEEKLY_CONFIG,
    create_patchtst_model,
)
from brain_api.core.patchtst.india_prices import (
    NSE_RECOVERED_BAR,
    restore_nse_cash_bar_gaps,
)
from brain_api.core.patchtst.weekly_data import WeeklyPatchTSTInputs
from brain_api.core.patchtst.weekly_training import (
    PatchTSTTargetScale,
    fit_weekly_patchtst,
    predict_weekly_returns,
    weekly_fold_indices,
)
from brain_api.routes.training.dependencies import (
    get_india_patchtst_config,
    get_patchtst_config,
)
from brain_api.storage.patchtst.local import PatchTSTHalalNewModelStorage


def test_production_recipe_builds_one_scalar_and_reloads(tmp_path):
    assert get_patchtst_config() == US_WEEKLY_CONFIG
    assert get_india_patchtst_config() == INDIA_WEEKLY_CONFIG
    model = create_patchtst_model(US_WEEKLY_CONFIG)
    model.eval()
    values = np.zeros((2, 255, 7), dtype=np.float32)
    with torch.no_grad():
        assert model(past_values=torch.from_numpy(values)).regression_outputs.shape == (
            2,
            1,
        )
    scale = PatchTSTTargetScale(0.043)
    storage = PatchTSTHalalNewModelStorage(tmp_path)
    storage.write_artifacts("v-test", model, scale, US_WEEKLY_CONFIG, {})
    storage.promote_version("v-test")
    artifact = storage.load_current_artifacts()
    assert artifact.feature_scaler == scale
    np.testing.assert_allclose(
        predict_weekly_returns(artifact.model, values, artifact.feature_scaler),
        predict_weekly_returns(model, values, scale),
    )


def test_session_horizon_is_distinct_from_holiday_week():
    calendar = load_patchtst_calendar("us", date(2026, 11, 18), date(2026, 12, 4))
    cutoff = date(2026, 11, 20)
    assert calendar.anchor(cutoff) == cutoff
    assert calendar.week(cutoff).target_week_end == date(2026, 11, 27)
    assert calendar.forecast_end(cutoff) == date(2026, 11, 30)


def test_six_channel_data_train_uses_a_purged_scalar_target(monkeypatch):
    start, end = date(2023, 1, 1), date(2024, 11, 29)
    calendar = training_calendar("us", start, end)
    prices = 100.0 + np.arange(len(calendar.sessions)) / 10.0
    bars = pd.DataFrame({"close": prices, "volume": 21_000}, index=calendar.sessions)
    config = replace(INDIA_WEEKLY_CONFIG, epochs=1, batch_size=16)

    def no_news(*args):
        pytest.fail("India must never read sentiment")

    monkeypatch.setattr(
        "brain_api.core.patchtst.weekly_data.load_session_sentiment", no_news
    )
    inputs = WeeklyPatchTSTInputs(config, calendar, {"AAA.NS": bars})
    dataset = inputs.labeled(start, end)
    assert dataset.X.shape[1:] == (255, 6)
    assert dataset.y.ndim == 1
    idx = calendar.sessions.get_loc(pd.Timestamp(dataset.anchor_dates[0]))
    assert dataset.y[0] == pytest.approx(prices[idx + 5] / prices[idx] - 1)
    train, val = weekly_fold_indices(dataset, config.validation_split)
    assert max(dataset.target_dates[train]) < min(dataset.anchor_dates[val])
    assert max(dataset.target_dates) <= end
    monkeypatch.setattr(
        "brain_api.core.patchtst.weekly_training.get_device",
        lambda: torch.device("cpu"),
    )
    result = fit_weekly_patchtst(dataset, config)
    assert result.evidence["weekly_rmse_pp"] == pytest.approx(
        100 * np.sqrt(result.val_loss)
    )
    assert result.feature_scaler.target_std == pytest.approx(
        np.std(dataset.y[train], dtype=np.float64)
    )


def test_recent_market_wide_price_gap_blocks_training():
    start, end = date(2023, 1, 1), date(2024, 11, 29)
    calendar = training_calendar("us", start, end)
    close = 100 + np.arange(len(calendar.sessions)) / 10
    bars = pd.DataFrame({"close": close, "volume": 1000}, index=calendar.sessions)
    bars.loc["2024-10-01":, "close"] = np.nan
    inputs = WeeklyPatchTSTInputs(INDIA_WEEKLY_CONFIG, calendar, {"AAA.NS": bars})
    with pytest.raises(ValueError, match="Stale PatchTST inputs"):
        inputs.labeled(start, end)


def test_india_calendar_uses_holidays_and_ignores_special_weekends(monkeypatch):
    monkeypatch.delenv("NSE_CASH_CALENDAR_PATH", raising=False)
    calendar = load_patchtst_calendar("india", date(2026, 1, 28), date(2026, 12, 31))
    assert pd.Timestamp("2026-02-01") not in calendar.sessions
    assert pd.Timestamp("2026-11-08") not in calendar.sessions
    assert pd.Timestamp("2026-12-25") not in calendar.sessions
    assert calendar.forecast_end(date(2026, 11, 20)) == date(2026, 11, 30)


def test_india_inference_accepts_a_five_session_horizon_at_year_end(monkeypatch):
    monkeypatch.delenv("NSE_CASH_CALENDAR_PATH", raising=False)

    def unexpected_fetch(year):
        pytest.fail(f"Unexpected request for NSE holidays in {year}")

    monkeypatch.setattr(
        "brain_api.storage.nse_calendar._fetch_nse_cash_holidays", unexpected_fetch
    )
    cutoff = date(2026, 12, 18)
    calendar = inference_calendar("india", cutoff)
    assert calendar.forecast_end(calendar.anchor(cutoff)) == date(2026, 12, 28)


def test_new_nse_year_is_cached(monkeypatch, tmp_path):
    monkeypatch.delenv("NSE_CASH_CALENDAR_PATH", raising=False)
    monkeypatch.setattr("brain_api.storage.nse_calendar.DEFAULT_DATA_PATH", tmp_path)
    fetched = []

    def fetch(year):
        fetched.append(year)
        return frozenset({date(2027, 1, 1)})

    monkeypatch.setattr(
        "brain_api.storage.nse_calendar._fetch_nse_cash_holidays", fetch
    )
    cutoff = date(2026, 12, 25)
    for _ in range(2):
        calendar = inference_calendar("india", cutoff)
        assert calendar.anchor(cutoff) == date(2026, 12, 24)
        assert calendar.forecast_end(date(2026, 12, 24)) == date(2027, 1, 4)
    assert fetched == [2027]


def test_recover_nse_gap_uses_matching_adjusted_close_factors(monkeypatch):
    sessions = pd.to_datetime(["2026-01-30", "2026-02-01", "2026-02-02"])
    prices = {
        "MATCH.NS": pd.DataFrame(
            {"close": [50.0, 55.0], "volume": [100, 200]}, index=sessions[[0, 2]]
        ),
        "MISMATCH.NS": pd.DataFrame(
            {"close": [75.0, 55.0], "volume": [100, 200]}, index=sessions[[0, 2]]
        ),
    }
    official = {
        day.date(): pd.DataFrame(
            {"close": [raw, raw], "volume": [500, 500]}, index=list(prices)
        )
        for day, raw in zip(sessions, [100.0, 105.0, 110.0], strict=True)
    }
    monkeypatch.setattr(
        "brain_api.core.patchtst.india_prices._cash_report", official.__getitem__
    )
    restored = restore_nse_cash_bar_gaps(prices, sessions)
    assert restored["MATCH.NS"].loc[sessions[1], ["close", "volume"]].tolist() == [
        52.5,
        500,
    ]
    assert bool(restored["MATCH.NS"].at[sessions[1], NSE_RECOVERED_BAR])
    assert restored["MISMATCH.NS"].loc[sessions[1], ["close", "volume"]].isna().all()


def test_weekly_snapshot_reloads_regression_head(tmp_path):
    from brain_api.storage.forecaster_snapshots import SnapshotLocalStorage

    store = SnapshotLocalStorage("patchtst_nifty_shariah_500", base_path=tmp_path)
    cutoff = date(2025, 12, 31)
    scale = PatchTSTTargetScale(0.02)
    model = create_patchtst_model(INDIA_WEEKLY_CONFIG)
    store.write_snapshot(
        cutoff_date=cutoff,
        snapshot_digest="123456789abc",
        model=model,
        feature_scaler=scale,
        config=INDIA_WEEKLY_CONFIG,
        metadata={},
    )
    loaded = store.load_snapshot(cutoff)
    assert loaded.feature_scaler == scale
    inputs = np.ones((1, 255, 6), dtype=np.float32)
    assert predict_weekly_returns(
        loaded.model, inputs, loaded.feature_scaler
    ).shape == (1,)


def test_compatible_sac_reuse_requires_forecast_recipe_and_window():
    from brain_api.news.models import NEWS_SCHEMA_VERSION, NEWS_SENTIMENT_REVISION
    from brain_api.routes.training.sac._shared import sac_current_is_reusable
    from brain_api.routes.training.sac.forecast_dependency import SAC_PATCHTST_RECIPE

    incumbent = {
        "sac_schema_version": 3,
        "architecture": "masked_attention",
        "news_schema_version": NEWS_SCHEMA_VERSION,
        "finbert_revision": NEWS_SENTIMENT_REVISION,
        "symbols": ["AAA"],
        "data_window": {"start": "2018-01-01"},
    }

    def reusable(metadata):
        return sac_current_is_reusable(
            metadata,
            ["AAA"],
            forecast_recipe=SAC_PATCHTST_RECIPE,
            start_date="2018-01-01",
        )

    assert not reusable(incumbent)
    migrated = dict(
        incumbent,
        patchtst_recipe=SAC_PATCHTST_RECIPE,
        patchtst_news_revision_policy="allow_later_revisions",
    )
    assert reusable(migrated)
    assert not reusable(dict(migrated, data_window={"start": "2019-01-01"}))

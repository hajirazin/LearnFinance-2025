"""Use the real saved snapshot/input path without changing the SAC learning rule."""

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from brain_api.core.patchtst.calendar import training_calendar
from brain_api.core.patchtst.config import (
    US_WEEKLY_CONFIG,
    create_patchtst_model,
    patchtst_recipe,
)
from brain_api.core.patchtst.weekly_training import PatchTSTTargetScale
from brain_api.core.portfolio_rl.weekly_patchtst import score_sac_forecasts
from brain_api.news.models import (
    NEWS_PROVIDER,
    NEWS_SCHEMA_VERSION,
    NEWS_SENTIMENT_MODEL,
    NEWS_SENTIMENT_REVISION,
    NewsCoverage,
    NewsEvent,
)
from brain_api.news.store import NewsStore
from brain_api.storage.forecaster_snapshots import SnapshotLocalStorage


@pytest.fixture
def historical_forecast_case(tmp_path, monkeypatch):
    from brain_api.core.forecaster_snapshot_identity import (
        expected_dec31_walkforward_snapshot_hash,
    )
    from brain_api.core.portfolio_rl import weekly_patchtst

    monkeypatch.setenv("STORAGE_BACKEND", "local_first")
    anchor = date(2022, 10, 28)
    calendar = training_calendar("us", anchor, date(2022, 10, 31))
    sessions = calendar.sessions[calendar.sessions <= pd.Timestamp(anchor)]
    symbols = [f"AAA{i:02}" for i in range(13)]
    prices = {
        s: pd.DataFrame(
            {"close": 100 + np.arange(len(sessions)) * 0.1, "volume": 10000.0},
            index=sessions,
        )
        for s in symbols
    }
    news = NewsStore(tmp_path)
    close = calendar.closes.loc[str(anchor)].to_pydatetime()
    for index, symbol in enumerate(symbols):
        events = (
            []
            if index >= 4
            else [
                NewsEvent(
                    NEWS_PROVIDER,
                    f"article-{index}",
                    symbol,
                    close - timedelta(hours=1),
                    close + timedelta(days=2),
                    "benzinga",
                    0.8,
                    0.9,
                    0.05,
                    0.05,
                    0.9,
                    f"test-hash-{index}",
                    NEWS_SENTIMENT_MODEL,
                    NEWS_SENTIMENT_REVISION,
                    NEWS_SCHEMA_VERSION,
                    datetime(2022, 11, 1, tzinfo=UTC),
                )
            ]
        )
        news.commit_window(
            events=events,
            coverage=NewsCoverage(
                NEWS_PROVIDER,
                symbol,
                calendar.closes.iloc[0].to_pydatetime() - timedelta(days=7),
                close,
                NEWS_SCHEMA_VERSION,
                NEWS_SENTIMENT_MODEL,
                NEWS_SENTIMENT_REVISION,
                "verified_empty",
                1,
                0,
                0,
                datetime(2022, 11, 1, tzinfo=UTC),
                "test-proof",
            ),
            cache_rows=[],
        )
    snapshots = SnapshotLocalStorage("patchtst_halal_new", base_path=tmp_path)
    digest = expected_dec31_walkforward_snapshot_hash(
        forecaster_bucket="patchtst_halal_new",
        cutoff_date=date(2021, 12, 31),
        config_dict=US_WEEKLY_CONFIG.to_dict(),
    )
    snapshots.write_snapshot(
        cutoff_date=date(2021, 12, 31),
        snapshot_digest=digest,
        model=create_patchtst_model(US_WEEKLY_CONFIG),
        feature_scaler=PatchTSTTargetScale(0.03),
        config=US_WEEKLY_CONFIG,
        metadata={
            "config_symbols_hash": digest,
            "training_evidence": {"latest_label_end": "2021-12-27"},
        },
    )
    monkeypatch.setattr(weekly_patchtst, "SnapshotLocalStorage", lambda *a: snapshots)
    monkeypatch.setattr(
        weekly_patchtst, "load_prices_yfinance", lambda *a, **kw: prices
    )
    monkeypatch.setattr("brain_api.storage.base.DEFAULT_DATA_PATH", tmp_path)
    return symbols, prices


def forecast(symbols, evidence):
    return score_sac_forecasts(
        symbols,
        pd.to_datetime(["2022-10-28"]),
        pd.to_datetime(["2022-10-31"]),
        start=date(2022, 1, 1),
        end=date(2022, 10, 31),
        evidence=evidence,
    )


def test_approved_policy_restores_the_previously_blocked_forecast(
    historical_forecast_case,
):
    symbols, _ = historical_forecast_case
    evidence = {}
    result = forecast(symbols, evidence)
    assert set(result) == set(symbols)
    assert all(
        value.shape == (1,) and np.isfinite(value).all() for value in result.values()
    )
    assert evidence["patchtst_news_revision_policy"] == "allow_later_revisions"
    assert evidence["patchtst_news_later_revision_cells"] == 4
    assert evidence["patchtst_news_lookahead_bias"] is True


def test_approved_news_does_not_bypass_ten_stock_minimum(historical_forecast_case):
    symbols, prices = historical_forecast_case
    for symbol in symbols[9:]:
        del prices[symbol]
    with pytest.raises(ValueError, match="Only 9 PatchTST inputs"):
        forecast(symbols, {})


def test_new_input_policy_does_not_reuse_point_in_time_sac_artifacts():
    from brain_api.routes.training.sac._shared import sac_current_is_reusable
    from brain_api.routes.training.sac.forecast_dependency import SAC_PATCHTST_RECIPE

    old_recipe = patchtst_recipe(US_WEEKLY_CONFIG)
    # The live forecast contract is unchanged; training data has a separate policy.
    assert old_recipe == SAC_PATCHTST_RECIPE
    metadata = {
        "sac_schema_version": 3,
        "architecture": "masked_attention",
        "news_schema_version": NEWS_SCHEMA_VERSION,
        "finbert_revision": NEWS_SENTIMENT_REVISION,
        "symbols": ["AAA"],
        "patchtst_recipe": old_recipe,
    }
    assert not sac_current_is_reusable(
        metadata, ["AAA"], forecast_recipe=SAC_PATCHTST_RECIPE
    )


def test_historical_revision_policy_changes_training_version(monkeypatch):
    from brain_api.core.sac import DEFAULT_SAC_CONFIG, version

    def identity():
        return version.compute_version(
            date(2018, 1, 1),
            date(2026, 11, 27),
            ["AAA"],
            DEFAULT_SAC_CONFIG,
            patchtst_recipe=patchtst_recipe(US_WEEKLY_CONFIG),
        )

    approved = identity()
    monkeypatch.setattr(
        version, "HISTORICAL_PATCHTST_NEWS_REVISION_POLICY", "point_in_time"
    )
    assert identity() != approved

"""Exercise the real seven-feature screening dependency before SAC training."""

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from brain_api.core.patchtst.calendar import inference_calendar
from brain_api.core.patchtst.config import US_WEEKLY_CONFIG, create_patchtst_model
from brain_api.core.patchtst.inference import BatchInferenceResult, run_batch_inference
from brain_api.core.patchtst.weekly_training import PatchTSTTargetScale
from brain_api.main import app
from brain_api.news.models import (
    NEWS_PROVIDER,
    NEWS_SCHEMA_VERSION,
    NEWS_SENTIMENT_MODEL,
    NEWS_SENTIMENT_REVISION,
    NewsCoverage,
)
from brain_api.news.store import NewsStore
from brain_api.routes.training import job_registry
from brain_api.storage.patchtst.local import PatchTSTHalalNewModelStorage


@pytest.fixture
def screening_case(monkeypatch, tmp_path):
    from brain_api.core.patchtst import weekly_inference
    from brain_api.routes.training.sac import full, preflight
    from brain_api.universe import halal_filtered

    cutoff = date(2026, 11, 27)
    symbols = [f"AAA{i:02}" for i in range(20)]
    calendar = inference_calendar("us", cutoff)
    anchor = calendar.anchor(cutoff)
    sessions = calendar.sessions[calendar.sessions <= pd.Timestamp(anchor)]
    prices = {
        s: pd.DataFrame(
            {"close": 100 + np.arange(len(sessions)) * 0.1, "volume": 10000.0},
            index=sessions,
        )
        for s in symbols
    }
    config = replace(
        US_WEEKLY_CONFIG,
        d_model=8,
        num_attention_heads=2,
        num_hidden_layers=1,
        ffn_dim=16,
    )
    storage = PatchTSTHalalNewModelStorage(tmp_path)
    storage.write_artifacts(
        "v-test",
        create_patchtst_model(config),
        PatchTSTTargetScale(0.03),
        config,
        {"symbols": symbols},
    )
    storage.promote_version("v-test")
    news = NewsStore(tmp_path)
    monkeypatch.setattr("brain_api.storage.base_local.DEFAULT_DATA_PATH", tmp_path)
    monkeypatch.setattr(weekly_inference, "NewsStore", lambda base_path=None: news)
    monkeypatch.setattr(
        weekly_inference, "load_prices_yfinance", lambda *a, **kw: prices
    )
    monkeypatch.setattr(halal_filtered, "resolve_cutoff_date", lambda: cutoff)
    monkeypatch.setattr(
        halal_filtered, "compute_min_walkforward_days", lambda as_of: 2000
    )
    monkeypatch.setattr(
        halal_filtered, "filter_symbols_by_min_history", lambda *a: (symbols, [])
    )
    monkeypatch.setattr(
        halal_filtered,
        "get_halal_new_universe",
        lambda: {"stocks": [{"symbol": s} for s in symbols]},
    )
    monkeypatch.setattr(full, "sync_news", lambda: "test-news-commit")
    monkeypatch.setattr(preflight, "sync_news", lambda: "test-news-commit")
    return SimpleNamespace(
        cutoff=cutoff,
        symbols=symbols,
        calendar=calendar,
        storage=storage,
        news=news,
        prices=prices,
        tmp_path=tmp_path,
        client=TestClient(app, raise_server_exceptions=False),
    )


def prove_empty_news(case, *, end=None, symbols=None):
    """A zero sentiment requires complete coverage; no news events are necessary."""
    for symbol in case.symbols if symbols is None else symbols:
        case.news.commit_window(
            events=[],
            cache_rows=[],
            coverage=NewsCoverage(
                NEWS_PROVIDER,
                symbol,
                case.calendar.closes.iloc[0].to_pydatetime() - timedelta(days=7),
                case.calendar.closes.loc[str(end or case.cutoff)].to_pydatetime(),
                NEWS_SCHEMA_VERSION,
                NEWS_SENTIMENT_MODEL,
                NEWS_SENTIMENT_REVISION,
                "verified_empty",
                1,
                0,
                0,
                datetime(2026, 11, 28, tzinfo=UTC),
                "test-proof",
            ),
        )


def test_weekly_batch_identifies_missing_latest_news(screening_case):
    case = screening_case
    prove_empty_news(case, end=date(2026, 11, 20))
    with pytest.raises(ValueError, match=r"news sentiment.*2026-11-27"):
        run_batch_inference(case.symbols, case.cutoff, storage=case.storage)


def test_full_sac_returns_actionable_503_without_starting_job(screening_case):
    case = screening_case
    response = case.client.post("/train/sac/full", json={"universe": "halal_filtered"})
    assert response.status_code == 503, response.text
    assert "news sentiment" in response.json()["detail"]
    assert "2026-11-27" in response.json()["detail"]
    assert "/etl/news/patchtst" in response.json()["detail"]
    assert not job_registry._training_jobs
    assert not (case.tmp_path / "sticky_history.db").exists()


def test_preflight_reports_screening_dependency_instead_of_crashing(screening_case):
    response = screening_case.client.post(
        "/train/sac/preflight", json={"universe": "halal_filtered"}
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert not result["ready"]
    assert result["symbols"] == []
    assert result["errors"][0]["source"] == "universe"
    assert "news sentiment" in result["errors"][0]["detail"]
    assert result["errors"][0]["retryable"] is False
    assert not job_registry._training_jobs


def test_verified_empty_news_restores_real_screening(screening_case):
    case = screening_case
    failed = case.client.get("/universe/halal_filtered")
    assert failed.status_code == 503
    prove_empty_news(case)
    batch = run_batch_inference(case.symbols, case.cutoff, storage=case.storage)
    assert len(batch.predictions) == 20
    assert all(np.isfinite(p.predicted_weekly_return_pct) for p in batch.predictions)
    response = case.client.get("/universe/halal_filtered")
    assert response.status_code == 200, response.text
    assert len(response.json()["stocks"]) == 15
    assert response.json()["total_candidates"] == 20
    assert response.json()["partition"] == "halal_filtered_alpha"


def test_partial_news_coverage_keeps_per_symbol_exclusions(screening_case):
    case = screening_case
    prove_empty_news(case, symbols=case.symbols[:1])
    batch = run_batch_inference(case.symbols, case.cutoff, storage=case.storage)
    valid = [p for p in batch.predictions if p.predicted_weekly_return_pct is not None]
    assert [p.symbol for p in valid] == case.symbols[:1]
    assert len(batch.predictions) == 20


def test_price_gap_with_complete_news_has_no_false_news_diagnosis(screening_case):
    case = screening_case
    prove_empty_news(case)
    for prices in case.prices.values():
        prices.iloc[-50, prices.columns.get_loc("close")] = np.nan
    response = case.client.post("/train/sac/full", json={"universe": "halal_filtered"})
    assert response.status_code == 503, response.text
    assert "no usable input windows" in response.json()["detail"]
    assert "Missing compatible news sentiment" not in response.json()["detail"]
    assert not job_registry._training_jobs


def test_no_prices_is_a_dependency_503_not_an_empty_slate(screening_case):
    case = screening_case
    case.prices.clear()
    response = case.client.post("/train/sac/full", json={"universe": "halal_filtered"})
    assert response.status_code == 503, response.text
    assert "price evidence" in response.json()["detail"]
    assert not job_registry._training_jobs


def test_empty_batch_is_rejected_before_sticky_history(screening_case, monkeypatch):
    from brain_api.universe import halal_filtered

    monkeypatch.setattr(
        halal_filtered,
        "run_batch_inference",
        lambda *a: BatchInferenceResult([], "v-test"),
    )
    response = screening_case.client.post(
        "/train/sac/full", json={"universe": "halal_filtered"}
    )
    assert response.status_code == 503, response.text
    detail = response.json()["detail"]
    assert "0 valid predictions" in detail
    assert "20 requested" in detail
    assert "v-test" in detail
    assert not (screening_case.tmp_path / "sticky_history.db").exists()
    assert not job_registry._training_jobs

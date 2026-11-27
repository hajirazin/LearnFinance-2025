"""Synthetic application inputs for the actual US and India weekly recipes."""

from dataclasses import replace
from datetime import UTC, date, datetime
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from brain_api.core import model_buckets
from brain_api.core.model_buckets import ModelType, get_bucket
from brain_api.core.patchtst.config import create_patchtst_model
from brain_api.core.patchtst.progress import check_cancelled
from brain_api.core.patchtst.weekly_training import (
    PatchTSTTargetScale,
    WeeklyPatchTSTResult,
)
from brain_api.main import app, shutdown_event
from brain_api.news.models import (
    NEWS_PROVIDER,
    NEWS_SCHEMA_VERSION,
    NEWS_SENTIMENT_MODEL,
    NEWS_SENTIMENT_REVISION,
    NewsCoverage,
)
from brain_api.news.store import NewsStore
from brain_api.routes.training.dependencies import (
    get_patchtst_price_loader,
    get_weekly_patchtst_trainer,
)


def make_training_case(monkeypatch, tmp_path, market):
    universe = "halal_new" if market == "us" else "nifty_shariah_500"
    original = get_bucket(ModelType.PATCHTST, universe)
    storage = original.local_storage_class(tmp_path)
    symbols = [f"{s}.NS" if market == "india" else s for s in ("AAA", "BBB", "CCC")]
    case = SimpleNamespace(
        market=market,
        universe=universe,
        storage=storage,
        symbols=symbols,
        endpoint="/train/patchtst" + ("/india" if market == "india" else ""),
        repo=None,
        upload=Mock(return_value=SimpleNamespace(repo_id="test/model")),
        fit_calls=[],
        price_calls=[],
        loss=0.01,
    )
    bucket = replace(
        original,
        local_storage_class=lambda: storage,
        symbols_resolver=lambda: case.symbols,
        hf_repo_getter=lambda: case.repo,
        hf_storage_class=lambda **kwargs: SimpleNamespace(
            upload_model=case.upload, token=None
        ),
    )
    monkeypatch.setitem(model_buckets._BUCKETS, (ModelType.PATCHTST, universe), bucket)
    case.bucket = bucket
    case.artifact_root = tmp_path / "models" / bucket.bucket_name
    monkeypatch.setattr(
        "brain_api.storage.policy._fetch_hf_main_metadata", lambda **kwargs: None
    )
    module = "patchtst" if market == "us" else "patchtst_india"
    monkeypatch.setattr(
        f"brain_api.routes.training.{module}.resolve_training_window",
        lambda: (date(2023, 1, 1), date(2024, 11, 29)),
    )

    def prices(requested, start, end):
        case.price_calls.append((requested, start, end))
        index = pd.bdate_range(start, end)
        t = np.arange(len(index))
        return {
            symbol: pd.DataFrame(
                {
                    "close": 100 * np.exp(0.0002 * t + 0.01 * np.sin(t / 15 + i)),
                    "volume": 10000 + (t % 20) * 10,
                },
                index=index,
            )
            for i, symbol in enumerate(requested)
        }

    def fit(data, config, *, shutdown_event=None, on_progress=None):
        check_cancelled(shutdown_event)
        assert data.X.shape[1:] == (255, 7 if market == "us" else 6)
        case.fit_calls.append((data, config))
        if on_progress:
            on_progress(
                {
                    "phase": "training_weekly_regression",
                    "epoch": 1,
                    "total_epochs": config.epochs,
                    "val_loss": case.loss,
                }
            )
        return WeeklyPatchTSTResult(
            create_patchtst_model(config),
            PatchTSTTargetScale(0.03),
            case.loss,
            case.loss,
            0.03,
            1,
            1,
            0.1,
            {
                "weekly_rmse_pp": 100 * np.sqrt(case.loss),
                "latest_label_end": max(data.target_dates).isoformat(),
                "latest_label_symbols": len(symbols),
            },
        )

    case.prices, case.fit = prices, fit
    monkeypatch.setitem(
        app.dependency_overrides, get_patchtst_price_loader, lambda: prices
    )
    monkeypatch.setitem(
        app.dependency_overrides, get_weekly_patchtst_trainer, lambda: fit
    )
    shutdown_event.clear()
    if market == "us":
        news = NewsStore(tmp_path)
        for symbol in symbols:
            news.commit_window(
                events=[],
                cache_rows=[],
                coverage=NewsCoverage(
                    NEWS_PROVIDER,
                    symbol,
                    datetime(2010, 1, 1, tzinfo=UTC),
                    datetime(2025, 1, 1, tzinfo=UTC),
                    NEWS_SCHEMA_VERSION,
                    NEWS_SENTIMENT_MODEL,
                    NEWS_SENTIMENT_REVISION,
                    "verified_empty",
                    1,
                    0,
                    0,
                    datetime(2025, 1, 1, tzinfo=UTC),
                    "synthetic-empty-proof",
                ),
            )
    case.client = TestClient(app)
    return case


def submit(case, *, snapshots=False, body=None):
    response = case.client.post(
        case.endpoint + ("" if snapshots else "?skip_snapshot=true"),
        json={"universe": case.universe} if body is None else body,
    )
    assert response.status_code == 202, response.text
    status = case.client.get("/train/status/" + response.json()["job_id"]).json()
    return response, status

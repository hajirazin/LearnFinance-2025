"""Historical e2e: mock yfinance.download only at the price edge.

Alpaca ``fetch_news_page`` is the news equivalent of that download edge
(verified-zero complete pages). PPO internals are not mocked.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from brain_api.core.ppo_discovery.config import (
    HISTORY_BARS,
    MIN_ELIGIBLE_ASSETS,
    REQUIRED_ABLATIONS,
    PPODiscoveryConfig,
)
from brain_api.core.ppo_discovery.inference import run_ppo_discovery_inference
from brain_api.core.ppo_discovery.news_adapter import (
    PPOSymbolNewsFeatures,
    features_to_schema,
)
from brain_api.core.ppo_discovery.pipeline import run_ppo_discovery_training
from brain_api.core.ppo_discovery.promotion import evaluate_ppo_discovery_promotion
from brain_api.core.ppo_discovery.schemas import CanonicalPPOState, PPODiscoveryError
from brain_api.core.ppo_discovery.synthetic import make_synthetic_state
from brain_api.core.ppo_discovery.universe_snapshot import build_universe_snapshot
from brain_api.storage.ppo_discovery.local import PPODiscoveryHalalNewModelStorage

SYMBOLS = [f"T{i:02d}" for i in range(MIN_ELIGIBLE_ASSETS + 1)]


def _sessions(n: int = 400) -> pd.DatetimeIndex:
    calendar = xcals.get_calendar("XNYS")
    sessions = calendar.sessions_in_range(
        pd.Timestamp("2024-01-02"), pd.Timestamp("2026-08-01")
    )
    return pd.DatetimeIndex(sessions[-n:]).tz_localize(None).normalize()


def _ohlcv_frame(index: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.012, size=len(index))
    close = 100.0 * np.cumprod(1.0 + rets)
    open_ = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    volume = np.full(len(index), 1e6)
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=index,
    )


def _yahoo_multiindex(index: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    frames = {
        symbol: _ohlcv_frame(index, seed=i + 1) for i, symbol in enumerate(tickers)
    }
    columns = pd.MultiIndex.from_tuples(
        [(symbol, field) for symbol in tickers for field in frames[symbol].columns],
        names=["Ticker", "Price"],
    )
    data = {}
    for symbol in tickers:
        for field in frames[symbol].columns:
            data[(symbol, field)] = frames[symbol][field]
    return pd.DataFrame(data, index=index, columns=columns)


@pytest.fixture
def xnys_index() -> pd.DatetimeIndex:
    return _sessions(400)


def test_historical_train_eval_candidate_with_yfinance_mocked(
    tmp_path: Path, xnys_index: pd.DatetimeIndex
) -> None:
    tickers = [*SYMBOLS, "SPY", "^VIX"]
    yahoo = _yahoo_multiindex(xnys_index, tickers)
    start = xnys_index[0].date()
    end = xnys_index[-1].date()
    snapshot = build_universe_snapshot(
        SYMBOLS,
        retrieved_at=datetime.fromisoformat(f"{end.isoformat()}T20:00:00+00:00"),
    )
    storage = PPODiscoveryHalalNewModelStorage(base_path=tmp_path)
    config = PPODiscoveryConfig(
        dropout=0.0,
        total_timesteps=4,
        ppo_epochs=1,
        minibatch_size=2,
        freeze_encoder_updates=20,
        pretrain_max_epochs=1,
        pretrain_patience=1,
        seeds=(42,),
    )

    def _download(*_args, **_kwargs):
        return yahoo

    def _empty_news(cutoffs, symbols, store=None):
        empty = PPOSymbolNewsFeatures(
            raw_sentiment=0.0,
            article_count=0,
            log1p_article_count=0.0,
            recency=0.0,
            sentiment_dispersion=0.0,
        )
        return {
            cutoff: {
                symbol: features_to_schema(symbol, empty, [], cutoff=cutoff)
                for symbol in symbols
            }
            for cutoff in cutoffs
        }

    with (
        patch("brain_api.core.prices.yf.download", side_effect=_download),
        patch(
            "brain_api.core.ppo_discovery.pipeline.load_historical_ppo_news_features",
            side_effect=_empty_news,
        ),
        patch("brain_api.core.prices.yf.Ticker") as ticker_cls,
    ):
        ticker_cls.return_value.history.side_effect = AssertionError(
            "Ticker.history must not run when yahoo download parses"
        )
        result = run_ppo_discovery_training(
            snapshot,
            config=config,
            storage=storage,
            end_date=end,
            start_date=start,
            experiment_id="e2e",
            experiment_variant="diagnostic",
            base_path=tmp_path,
            alpha_hrp_weekly_log=None,
        )

    assert result["promoted"] is False
    assert storage.read_current_version() is None
    artifacts = storage.load_artifacts(result["version"])
    assert artifacts.metadata["experiment_variant"] == "diagnostic"
    assert artifacts.regime_hmm.get("schema_version") == 3
    assert "terminal_posterior" in artifacts.regime_hmm
    evaluation = result["evaluation"]
    assert np.isfinite(evaluation["test_cagr"])
    for name in REQUIRED_ABLATIONS:
        assert evaluation["ablations"][name]["status"] in {"ok", "failed"}
        assert evaluation["ablations"][name]["status"] != "unavailable"
    check = evaluate_ppo_discovery_promotion(
        metadata=artifacts.metadata,
        evaluation=evaluation,
        approved_by="razin",
        expected_config_hash=artifacts.metadata["config_hash"],
    )
    assert check.is_healthy is False
    mutated = make_synthetic_state()
    payload = mutated.to_dict()
    payload["asset_features"][0][0] = float(payload["asset_features"][0][0]) + 1.0
    with pytest.raises(PPODiscoveryError, match="state_digest"):
        CanonicalPPOState.from_dict(payload)
    state = make_synthetic_state()
    with pytest.raises(PPODiscoveryError, match="full experiment variant"):
        run_ppo_discovery_inference(
            state, expected_digest=state.state_digest, artifacts=artifacts
        )
    assert HISTORY_BARS == 253

"""Tests for the SAC v3 raw-evidence handoff."""

import math

import pytest

from activities import execution as execution_module
from activities import inference as inference_module
from activities.sac_context import (
    _compute_momentum_1w,
    _compute_momentum_4w,
    _compute_momentum_12_1,
    _compute_realized_vol_20d,
    build_sac_feature_bundle,
)
from models import (
    AdjustedClosesResponse,
    AlpacaPortfolioResponse,
    MarketHistoryResponse,
    NewsSignalResponse,
    PatchTSTInferenceResponse,
    SACInferenceResponse,
)
from tests._fake_client import FakeClient, patch_client


def _canonical_inputs():
    symbols = ["AAPL"]
    news = NewsSignalResponse(
        run_id="paper:2026-02-05",
        as_of_date="2026-02-05",
        per_symbol=[
            {
                "symbol": "AAPL",
                "sentiment_score": 0.0,
                "article_count_fetched": 0,
                "article_count_used": 0,
            }
        ],
    )
    patchtst = PatchTSTInferenceResponse(
        predictions=[
            {
                "symbol": "AAPL",
                "predicted_weekly_return_pct": 3.0,
                "direction": "up",
                "has_enough_history": True,
            }
        ],
        model_version="patch-v1",
        as_of_date="2026-02-05",
    )
    return symbols, news, patchtst


def _canonical_prices(symbols: list[str]) -> AdjustedClosesResponse:
    return AdjustedClosesResponse(
        as_of_date="2026-02-05",
        adjusted_closes={
            symbol: [100.0 + i * 0.1 for i in range(253)] for symbol in symbols
        },
        execution_prices={symbol: 125.2 for symbol in symbols},
        provenance={"provider": "yfinance", "adjusted": True},
    )


def _canonical_market() -> MarketHistoryResponse:
    return MarketHistoryResponse(
        start_date="2026-02-03",
        as_of_date="2026-02-04",
        rows=[
            {
                "date": "2026-02-03",
                "spy_adjusted_close": 600.0,
                "vix_close": 15.0,
            },
            {
                "date": "2026-02-04",
                "spy_adjusted_close": 601.0,
                "vix_close": 14.5,
            },
        ],
        provenance={"provider": "yfinance", "ordered": True},
    )


def test_feature_bundle_contains_only_raw_evidence_and_provenance():
    symbols, news, patchtst = _canonical_inputs()

    bundle = build_sac_feature_bundle(
        symbols=symbols,
        as_of_date="2026-02-05",
        news=news,
        patchtst=patchtst,
        prices=_canonical_prices(symbols),
        market=_canonical_market(),
    )

    assert set(bundle) == {
        "symbols",
        "adjusted_closes",
        "news_sentiment",
        "news_article_counts",
        "patchtst_forecasts",
        "execution_prices",
        "market_history",
        "provenance",
    }
    assert bundle["news_sentiment"] == {"AAPL": 0.0}
    assert bundle["news_article_counts"] == {"AAPL": 0}
    assert bundle["patchtst_forecasts"] == {"AAPL": 0.03}
    assert bundle["execution_prices"] == {"AAPL": 125.2}
    assert bundle["market_history"][0]["date"] == "2026-02-03"
    assert set(bundle["provenance"]) == {
        "as_of_date",
        "adjusted_closes",
        "news",
        "patchtst",
        "market_history",
    }
    assert "signals" not in bundle
    assert "news_coverage" not in repr(bundle)


@pytest.mark.parametrize("mutation", ["missing_news", "missing_forecast"])
def test_feature_bundle_rejects_incomplete_inputs(mutation: str):
    symbols, news, patchtst = _canonical_inputs()
    if mutation == "missing_news":
        news.per_symbol = []
    elif mutation == "missing_forecast":
        patchtst.predictions = []

    with pytest.raises(ValueError):
        build_sac_feature_bundle(
            symbols=symbols,
            as_of_date="2026-02-05",
            news=news,
            patchtst=patchtst,
            prices=_canonical_prices(symbols),
            market=_canonical_market(),
        )


def test_feature_bundle_preserves_incomplete_price_and_forecast_evidence():
    symbols, news, patchtst = _canonical_inputs()
    patchtst.predictions[0].predicted_weekly_return_pct = None
    prices = _canonical_prices(symbols)
    prices.adjusted_closes["AAPL"] = []
    prices.execution_prices.clear()

    bundle = build_sac_feature_bundle(
        symbols=symbols,
        as_of_date="2026-02-05",
        news=news,
        patchtst=patchtst,
        prices=prices,
        market=_canonical_market(),
    )

    assert bundle["adjusted_closes"] == {"AAPL": []}
    assert bundle["patchtst_forecasts"] == {}
    assert bundle["execution_prices"] == {}


def test_feature_bundle_keeps_held_execution_price_outside_policy_slate():
    symbols, news, patchtst = _canonical_inputs()
    prices = _canonical_prices(symbols)
    prices.adjusted_closes["OLD"] = [50.0]
    prices.execution_prices["OLD"] = 50.0

    bundle = build_sac_feature_bundle(
        symbols=symbols,
        as_of_date="2026-02-05",
        news=news,
        patchtst=patchtst,
        prices=prices,
        market=_canonical_market(),
    )

    assert set(bundle["adjusted_closes"]) == {"AAPL"}
    assert bundle["execution_prices"]["OLD"] == 50.0


def test_temporal_formula_parity_helpers_use_adjusted_close_contract():
    closes = [100.0 + index**2 / 1000.0 for index in range(253)]

    assert _compute_momentum_1w(closes, as_of_index=252) == pytest.approx(
        closes[252] / closes[247] - 1.0
    )
    assert _compute_momentum_4w(closes, as_of_index=252) == pytest.approx(
        closes[252] / closes[232] - 1.0
    )
    assert _compute_momentum_12_1(closes, as_of_index=252) == pytest.approx(
        closes[231] / closes[0] - 1.0
    )
    returns = [math.log(closes[index] / closes[index - 1]) for index in range(233, 253)]
    mean = sum(returns) / len(returns)
    expected_vol = math.sqrt(
        sum((value - mean) ** 2 for value in returns) / 19
    ) * math.sqrt(252.0)
    assert _compute_realized_vol_20d(closes, as_of_index=252) == pytest.approx(
        expected_vol
    )


def test_raw_evidence_fetch_activities_send_point_in_time_contracts():
    fake = FakeClient(
        {
            "/signals/prices": _canonical_prices(["AAPL"]).model_dump(),
            "/signals/market-history": _canonical_market().model_dump(),
        }
    )
    with patch_client(inference_module, fake):
        prices = inference_module.get_adjusted_closes(
            ["AAPL"], "2026-02-05", lookback_bars=253
        )
        market = inference_module.get_market_history("2026-02-02", "2026-02-05")

    assert prices.adjusted_closes["AAPL"][0] == 100.0
    assert market.rows[0].spy_adjusted_close == 600.0
    assert fake.calls[0]["json"] == {
        "symbols": ["AAPL"],
        "as_of_date": "2026-02-05",
        "lookback_bars": 253,
    }
    assert fake.calls[1]["json"] == {
        "start_date": "2026-02-03",
        "as_of_date": "2026-02-04",
    }


def test_market_history_skips_fetch_when_decision_is_training_cutoff():
    fake = FakeClient({})
    with patch_client(inference_module, fake):
        market = inference_module.get_market_history("2026-02-05", "2026-02-05")

    assert market.rows == []
    assert market.start_date == "2026-02-06"
    assert market.as_of_date == "2026-02-04"
    assert fake.calls == []


def test_market_history_monday_preopen_never_requests_partial_monday():
    response = MarketHistoryResponse(
        start_date="2026-08-08",
        as_of_date="2026-08-09",
        rows=[],
        provenance={"calendar": "XNYS"},
    )
    fake = FakeClient({"/signals/market-history": response.model_dump()})
    with patch_client(inference_module, fake):
        market = inference_module.get_market_history("2026-08-07", "2026-08-10")

    assert market.as_of_date == "2026-08-09"
    assert fake.calls[0]["json"] == {
        "start_date": "2026-08-08",
        "as_of_date": "2026-08-09",
    }


def test_market_history_rejects_future_training_cutoff():
    with pytest.raises(ValueError, match="cutoff cannot be after"):
        inference_module.get_market_history("2026-02-06", "2026-02-05")


class _InferenceClient(FakeClient):
    def post(self, path, *, params=None, json=None):
        self.calls.append(
            {"method": "POST", "path": path, "params": params, "json": json}
        )
        return self._response(path)

    def _response(self, path):
        from tests._fake_client import FakeResponse

        return FakeResponse(self._responses[path])


@pytest.mark.parametrize("universe", ["halal_filtered", "halal"])
def test_infer_sac_sends_exact_feature_bundle_and_reads_audit_state(universe: str):
    symbols, news, patchtst = _canonical_inputs()
    decision_state = {
        "vector": [0.0],
        "context": {"as_of_date": "2026-02-05"},
        "digest": "abc123",
    }
    fake = _InferenceClient(
        {
            "/inference/sac": {
                "target_weights": {"AAPL": 0.4, "CASH": 0.6},
                "turnover": 0.1,
                "model_version": "sac-v2",
                "target_week_start": "2026-02-09",
                "target_week_end": "2026-02-13",
                "weight_changes": [],
                "decision_state": decision_state,
                "state_digest": "abc123",
                "forced_liquidations": [{"symbol": "OLD", "market_value": 50.0}],
            }
        }
    )
    with patch_client(inference_module, fake):
        result = inference_module.infer_sac(
            portfolio=AlpacaPortfolioResponse(
                cash=1000.0, positions=[], open_orders_count=0
            ),
            as_of_date="2026-02-05",
            universe=universe,
            symbols=symbols,
            news=news,
            patchtst=patchtst,
            prices=_canonical_prices(symbols),
            market=_canonical_market(),
        )

    payload = fake.calls[0]["json"]
    assert fake.calls[0]["params"] == {"universe": universe}
    assert payload["feature_bundle"]["news_sentiment"] == {"AAPL": 0.0}
    assert "signals" not in payload["feature_bundle"]
    assert "news_coverage" not in repr(payload["feature_bundle"])
    assert result.decision_state == decision_state
    assert result.state_digest == "abc123"
    assert result.forced_liquidations[0].symbol == "OLD"


def test_store_experience_persists_allocator_state_and_digest_verbatim():
    state = {
        "vector": [1.0, 2.0],
        "context": {"as_of_date": "2026-02-05"},
        "digest": "canonical-digest",
    }
    allocation = SACInferenceResponse(
        target_weights={"AAPL": 0.4, "CASH": 0.6},
        turnover=0.1,
        model_version="sac-v2",
        decision_state=state,
        state_digest="canonical-digest",
    )
    fake = FakeClient(
        {
            "/experience/store": {
                "record_id": "record-1",
                "stored": True,
                "model_type": "sac",
            }
        }
    )
    with patch_client(execution_module, fake):
        execution_module.store_experience_sac(
            "paper:2026-02-05",
            "2026-02-09",
            "2026-02-13",
            allocation,
            "halal_filtered",
        )

    payload = fake.calls[0]["json"]
    assert payload["state"] == state
    assert payload["state_digest"] == "canonical-digest"

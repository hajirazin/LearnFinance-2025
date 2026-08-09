"""Tests for the canonical five-signal SAC feature/state handoff."""

import pytest

from activities import execution as execution_module
from activities import inference as inference_module
from activities.sac_context import build_sac_feature_bundle
from models import (
    AlpacaPortfolioResponse,
    ClosesResponse,
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


def _canonical_closes(symbols: list[str]) -> dict[str, list[float]]:
    return {symbol: [100.0 + i * 0.1 for i in range(253)] for symbol in symbols}


def test_feature_bundle_uses_exact_five_signal_order():
    symbols, news, patchtst = _canonical_inputs()

    bundle = build_sac_feature_bundle(
        symbols=symbols,
        as_of_date="2026-02-05",
        news=news,
        patchtst=patchtst,
        closes=_canonical_closes(symbols),
    )

    assert list(bundle["signals"]["AAPL"]) == [
        "news_sentiment",
        "news_coverage",
        "momentum_1w",
        "momentum_4w",
        "momentum_12_1",
    ]
    assert bundle["signals"]["AAPL"]["news_sentiment"] == 0.0
    assert bundle["signals"]["AAPL"]["news_coverage"] == 0.0
    assert bundle["patchtst_forecasts"] == {"AAPL": 0.03}
    assert set(bundle["provenance"]) == {"as_of_date", "news", "patchtst"}


@pytest.mark.parametrize("mutation", ["missing_news", "missing_forecast", "closes"])
def test_feature_bundle_rejects_incomplete_inputs(mutation: str):
    symbols, news, patchtst = _canonical_inputs()
    closes = _canonical_closes(symbols)
    if mutation == "missing_news":
        news.per_symbol = []
    elif mutation == "missing_forecast":
        patchtst.predictions = []
    else:
        closes.clear()

    with pytest.raises(ValueError):
        build_sac_feature_bundle(
            symbols=symbols,
            as_of_date="2026-02-05",
            news=news,
            patchtst=patchtst,
            closes=closes,
        )


class _InferenceClient(FakeClient):
    def post(self, path, *, params=None, json=None):
        self.calls.append(
            {"method": "POST", "path": path, "params": params, "json": json}
        )
        return self._response(path)

    def _response(self, path):
        from tests._fake_client import FakeResponse

        return FakeResponse(self._responses[path])


def test_infer_sac_sends_exact_feature_bundle_and_reads_audit_state():
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
            universe="halal_filtered",
            symbols=symbols,
            news=news,
            patchtst=patchtst,
            closes=ClosesResponse(
                as_of_date="2026-02-05", closes=_canonical_closes(symbols)
            ),
        )

    payload = fake.calls[0]["json"]
    assert list(payload["feature_bundle"]["signals"]["AAPL"]) == [
        "news_sentiment",
        "news_coverage",
        "momentum_1w",
        "momentum_4w",
        "momentum_12_1",
    ]
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

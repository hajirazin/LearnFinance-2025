"""Tests for the canonical SAC feature/state handoff."""

import pytest

from activities import execution as execution_module
from activities import inference as inference_module
from activities.sac_context import build_sac_feature_bundle
from models import (
    AlpacaPortfolioResponse,
    FundamentalsResponse,
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
    fundamentals = FundamentalsResponse(
        as_of_date="2026-02-05",
        per_symbol=[
            {
                "symbol": "AAPL",
                "ratios": {
                    "gross_margin": 0.42,
                    "net_margin": 0.24,
                    "debt_to_equity": 0.3,
                    "filing_available_date": "2026-01-30",
                    "filing_accession_number": "0001",
                    "filing_form": "10-Q",
                    "filing_source": "sec_submissions",
                },
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
    return symbols, news, fundamentals, patchtst


def test_confirmed_zero_news_is_neutral_with_zero_coverage():
    symbols, news, fundamentals, patchtst = _canonical_inputs()

    bundle = build_sac_feature_bundle(
        symbols=symbols,
        as_of_date="2026-02-05",
        news=news,
        fundamentals=fundamentals,
        patchtst=patchtst,
    )

    assert bundle["signals"]["AAPL"]["news_sentiment"] == 0.0
    assert bundle["signals"]["AAPL"]["news_coverage"] == 0.0
    assert bundle["signals"]["AAPL"]["fundamental_age"] == 6.0
    assert "gross_margin" in bundle["signals"]["AAPL"]
    assert "lstm_forecasts" not in bundle
    assert bundle["patchtst_forecasts"] == {"AAPL": 0.03}


def test_fundamentals_activity_forwards_decision_date():
    fake = FakeClient(
        {
            "/signals/fundamentals": {
                "as_of_date": "2026-02-05",
                "per_symbol": [],
            }
        }
    )
    with patch_client(inference_module, fake):
        inference_module.get_fundamentals(["AAPL"], "2026-02-05")

    assert fake.calls[0]["json"] == {
        "symbols": ["AAPL"],
        "as_of_date": "2026-02-05",
    }


@pytest.mark.parametrize(
    "mutation",
    ["missing_fundamentals", "fundamental_error", "missing_forecast"],
)
def test_feature_bundle_rejects_incomplete_inputs(mutation: str):
    symbols, news, fundamentals, patchtst = _canonical_inputs()
    if mutation == "missing_fundamentals":
        fundamentals.per_symbol = []
    elif mutation == "fundamental_error":
        fundamentals.per_symbol[0].ratios = None
        fundamentals.per_symbol[0].error = "provider unavailable"
    else:
        patchtst.predictions = []

    with pytest.raises(ValueError):
        build_sac_feature_bundle(
            symbols=symbols,
            as_of_date="2026-02-05",
            news=news,
            fundamentals=fundamentals,
            patchtst=patchtst,
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
    symbols, news, fundamentals, patchtst = _canonical_inputs()
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
            fundamentals=fundamentals,
            patchtst=patchtst,
        )

    payload = fake.calls[0]["json"]
    assert payload["feature_bundle"]["signals"]["AAPL"]["fundamental_age"] == 6.0
    assert "lstm_forecasts" not in payload["feature_bundle"]
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

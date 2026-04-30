"""Activity-level tests for the US Alpha-HRP strategy.

Each test patches the brain_api ``httpx.Client`` returned by
``activities.client.get_client`` and asserts that the activity hits
the right path with the right JSON body, mapping the response back
into the expected typed Pydantic model.

These are fast, deterministic unit tests — they do **not** spin up
the Temporal worker; the workflow-level tests in
``test_us_alpha_hrp.py`` cover orchestration.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import pytest

from activities import execution as execution_module
from activities import inference as inference_module
from activities import reporting as reporting_module
from models import (
    AlpacaPortfolioResponse,
    GenerateOrdersResponse,
    HRPAllocationResponse,
    PatchTSTBatchScores,
    PositionModel,
    RankBandTopNResponse,
    SkippedAllocation,
    SkippedSubmitResponse,
    SubmitOrdersResponse,
    WeeklyReportEmailResponse,
    WeeklySummaryResponse,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, json_payload: dict, status: int = 200) -> None:
        self._payload = json_payload
        self.status_code = status
        self.text = str(json_payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    """Records the path + json body of each POST/GET, returns a queued response."""

    def __init__(
        self,
        responses: dict[str, dict],
        statuses: dict[str, int] | None = None,
    ) -> None:
        self._responses = responses
        self._statuses = statuses or {}
        self.calls: list[dict[str, Any]] = []

    def post(self, path: str, json: dict | None = None) -> _FakeResponse:
        self.calls.append({"method": "POST", "path": path, "json": json})
        if path not in self._responses:
            raise AssertionError(f"Unexpected POST {path}")
        return _FakeResponse(
            self._responses[path], status=self._statuses.get(path, 200)
        )

    def get(self, path: str) -> _FakeResponse:
        self.calls.append({"method": "GET", "path": path, "json": None})
        if path not in self._responses:
            raise AssertionError(f"Unexpected GET {path}")
        return _FakeResponse(self._responses[path])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


@contextmanager
def _patch_client(module, fake: _FakeClient):
    """Swap ``module.get_client`` for one that yields the fake client."""
    original = module.get_client
    module.get_client = lambda: fake
    try:
        yield fake
    finally:
        module.get_client = original


# ---------------------------------------------------------------------------
# C1: score_halal_new_with_patchtst
# ---------------------------------------------------------------------------


class TestScoreHalalNewWithPatchTST:
    """The activity is now a thin HTTP wrapper around
    ``POST /inference/patchtst/score-batch`` with ``market='us'``.
    Math invariants (non-finite rejection, ``min_predictions`` floor)
    live in :mod:`brain_api.core.patchtst.score_validation` and are
    enforced inside the brain_api endpoint -- these activity tests
    only assert the wire contract.
    """

    def test_calls_score_batch_with_market_us_and_full_symbol_list(self):
        symbols = [f"SYM{i}" for i in range(20)]
        fake_response = {
            "scores": {s: float(20 - i) for i, s in enumerate(symbols)},
            "model_version": "v2026-04-26-abc",
            "as_of_date": "2026-04-28",
            "target_week_start": "2026-04-27",
            "target_week_end": "2026-05-01",
            "requested_count": 20,
            "predicted_count": 20,
            "excluded_symbols": [],
        }
        fake = _FakeClient({"/inference/patchtst/score-batch": fake_response})
        with _patch_client(inference_module, fake):
            result = inference_module.score_halal_new_with_patchtst(
                symbols=symbols,
                as_of_date="2026-04-28",
            )

        assert isinstance(result, PatchTSTBatchScores)
        assert result.predicted_count == 20
        assert result.requested_count == 20
        assert result.scores["SYM0"] == 20.0
        assert result.scores["SYM19"] == 1.0
        assert result.excluded_symbols == []
        assert len(fake.calls) == 1
        call = fake.calls[0]
        assert call["path"] == "/inference/patchtst/score-batch"
        assert call["json"] == {
            "market": "us",
            "symbols": symbols,
            "as_of_date": "2026-04-28",
            "min_predictions": 15,
        }

    def test_passes_through_excluded_symbols_from_endpoint(self):
        # The endpoint already drops symbols whose prediction is None;
        # the activity simply maps the response into the typed model.
        fake_response = {
            "scores": {"A": 1.5, "C": 0.5},
            "model_version": "v",
            "as_of_date": "2026-04-28",
            "target_week_start": "2026-04-27",
            "target_week_end": "2026-05-01",
            "requested_count": 3,
            "predicted_count": 2,
            "excluded_symbols": ["B"],
        }
        fake = _FakeClient({"/inference/patchtst/score-batch": fake_response})
        with _patch_client(inference_module, fake):
            result = inference_module.score_halal_new_with_patchtst(
                symbols=["A", "B", "C"],
                as_of_date="2026-04-28",
                min_predictions=2,
            )
        assert set(result.scores) == {"A", "C"}
        assert result.excluded_symbols == ["B"]

    def test_re_raises_422_as_runtime_error(self):
        # The brain_api endpoint returns 422 when the math invariants
        # (non-finite scores or below ``min_predictions`` floor) fail.
        # The activity must surface that as RuntimeError so the
        # workflow's RetryPolicy treats it as terminal.
        fake = _FakeClient(
            responses={
                "/inference/patchtst/score-batch": {
                    "detail": (
                        "PatchTST batch produced 1 valid score but min_predictions=15"
                    )
                }
            },
            statuses={"/inference/patchtst/score-batch": 422},
        )
        with (
            _patch_client(inference_module, fake),
            pytest.raises(RuntimeError, match="min_predictions"),
        ):
            inference_module.score_halal_new_with_patchtst(
                symbols=["A"], as_of_date="2026-04-28", min_predictions=15
            )


class TestScoreHalalIndiaWithPatchTST:
    """Mirrors the US wrapper but with ``market='india'``."""

    def test_calls_score_batch_with_market_india(self):
        symbols = ["NSE001.NS", "NSE002.NS", "NSE003.NS"]
        fake_response = {
            "scores": {s: float(3 - i) for i, s in enumerate(symbols)},
            "model_version": "v2026-04-26-india",
            "as_of_date": "2026-04-28",
            "target_week_start": "2026-04-27",
            "target_week_end": "2026-05-01",
            "requested_count": 3,
            "predicted_count": 3,
            "excluded_symbols": [],
        }
        fake = _FakeClient({"/inference/patchtst/score-batch": fake_response})
        with _patch_client(inference_module, fake):
            result = inference_module.score_halal_india_with_patchtst(
                symbols=symbols,
                as_of_date="2026-04-28",
                min_predictions=2,
            )
        assert isinstance(result, PatchTSTBatchScores)
        assert result.model_version == "v2026-04-26-india"
        assert result.scores["NSE001.NS"] == 3.0
        call = fake.calls[0]
        assert call["path"] == "/inference/patchtst/score-batch"
        assert call["json"] == {
            "market": "india",
            "symbols": symbols,
            "as_of_date": "2026-04-28",
            "min_predictions": 2,
        }


# ---------------------------------------------------------------------------
# C2: select_rank_band_top_n
# ---------------------------------------------------------------------------


class TestSelectRankBandTopN:
    def test_calls_rank_band_top_n_with_universe_and_thresholds(self):
        fake_response = {
            "selected": ["A", "B", "C"],
            "reasons": {"A": "top_rank", "B": "top_rank", "C": "top_rank"},
            "kept_count": 0,
            "fillers_count": 3,
            "evicted_from_previous": {},
            "previous_year_week_used": None,
            "universe": "halal_new_alpha",
            "year_week": "202618",
            "top_n": 3,
            "hold_threshold": 5,
        }
        fake = _FakeClient({"/allocation/rank-band-top-n": fake_response})
        with _patch_client(inference_module, fake):
            result = inference_module.select_rank_band_top_n(
                scores={"A": 1.0, "B": 0.5, "C": 0.25},
                universe="halal_new_alpha",
                year_week="202618",
                as_of_date="2026-04-28",
                run_id="paper:2026-04-28",
                top_n=3,
                hold_threshold=5,
            )

        assert isinstance(result, RankBandTopNResponse)
        assert result.selected == ["A", "B", "C"]
        call = fake.calls[0]
        assert call["path"] == "/allocation/rank-band-top-n"
        assert call["json"]["universe"] == "halal_new_alpha"
        assert call["json"]["top_n"] == 3
        assert call["json"]["hold_threshold"] == 5
        assert call["json"]["current_scores"] == {"A": 1.0, "B": 0.5, "C": 0.25}


# ---------------------------------------------------------------------------
# C3: generate_orders_alpha_hrp
# ---------------------------------------------------------------------------


class TestGenerateOrdersAlphaHrp:
    def test_calls_orders_generate_with_alpha_hrp_algorithm(self):
        allocation = HRPAllocationResponse(
            percentage_weights={"A": 60.0, "B": 40.0},
            symbols_used=2,
            symbols_excluded=[],
            lookback_days=252,
            as_of_date="2026-04-28",
        )
        portfolio = AlpacaPortfolioResponse(
            cash=10000.0,
            positions=[PositionModel(symbol="A", qty=5.0, market_value=500.0)],
            open_orders_count=0,
        )
        fake_response = {
            "orders": [
                {
                    "client_order_id": "paper:2026-04-28:attempt-1:A:BUY",
                    "symbol": "A",
                    "side": "buy",
                    "qty": 5.0,
                    "type": "market",
                    "time_in_force": "day",
                }
            ],
            "summary": {
                "buys": 1,
                "sells": 0,
                "total_buy_value": 500.0,
                "total_sell_value": 0.0,
                "turnover_pct": 5.0,
                "skipped_small_orders": 0,
                "skipped_below_threshold": 0,
            },
            "prices_used": {"A": 100.0},
        }
        fake = _FakeClient({"/orders/generate": fake_response})
        with _patch_client(execution_module, fake):
            result = execution_module.generate_orders_alpha_hrp(
                allocation=allocation,
                portfolio=portfolio,
                run_id="paper:2026-04-28",
                attempt=1,
            )

        assert isinstance(result, GenerateOrdersResponse)
        call = fake.calls[0]
        assert call["path"] == "/orders/generate"
        body = call["json"]
        assert body["algorithm"] == "alpha_hrp"
        # pp -> fraction conversion preserved (60.0 -> 0.6)
        assert body["target_weights"] == {"A": 0.6, "B": 0.4}
        assert body["run_id"] == "paper:2026-04-28"
        assert body["attempt"] == 1

    def test_skipped_allocation_returns_skipped_orders(self):
        portfolio = AlpacaPortfolioResponse(
            cash=10000.0, positions=[], open_orders_count=0
        )
        skipped = SkippedAllocation(skipped=True, algorithm="alpha_hrp")
        # No HTTP call should happen on the skipped path; pass an empty
        # fake to make any call explode.
        fake = _FakeClient({})
        with _patch_client(execution_module, fake):
            result = execution_module.generate_orders_alpha_hrp(
                allocation=skipped,
                portfolio=portfolio,
                run_id="paper:2026-04-28",
                attempt=1,
            )
        assert getattr(result, "skipped", False) is True
        assert result.algorithm == "alpha_hrp"
        assert fake.calls == []


# ---------------------------------------------------------------------------
# C4: reporting activities (LLM + email)
# ---------------------------------------------------------------------------


def _alpha_payload_fixtures():
    scores = PatchTSTBatchScores(
        scores={f"SYM{i}": float(20 - i) for i in range(20)},
        model_version="v2026-04-26-abc",
        as_of_date="2026-04-28",
        target_week_start="2026-04-27",
        target_week_end="2026-05-01",
        requested_count=410,
        predicted_count=20,
        excluded_symbols=[],
    )
    sticky = RankBandTopNResponse(
        selected=[f"SYM{i}" for i in range(15)],
        reasons={f"SYM{i}": "top_rank" for i in range(15)},
        kept_count=0,
        fillers_count=15,
        evicted_from_previous={},
        previous_year_week_used=None,
        universe="halal_new_alpha",
        year_week="202618",
        top_n=15,
        hold_threshold=30,
    )
    stage2 = HRPAllocationResponse(
        percentage_weights={f"SYM{i}": round(100.0 / 15, 2) for i in range(15)},
        symbols_used=15,
        symbols_excluded=[],
        lookback_days=252,
        as_of_date="2026-04-28",
    )
    return scores, sticky, stage2


class TestGenerateUSAlphaHrpSummary:
    def test_posts_to_alpha_hrp_summary_endpoint_with_top_25_scores(self):
        scores, sticky, stage2 = _alpha_payload_fixtures()
        fake = _FakeClient(
            {
                "/llm/us-alpha-hrp-summary": {
                    "summary": {"para_1_market_outlook": "Top names look strong."},
                    "provider": "openai",
                    "model_used": "gpt-4o-mini",
                    "tokens_used": 400,
                }
            }
        )
        with _patch_client(reporting_module, fake):
            result = reporting_module.generate_us_alpha_hrp_summary(
                scores=scores,
                sticky=sticky,
                stage2=stage2,
                universe="halal_new",
                top_n=15,
                hold_threshold=30,
            )

        assert isinstance(result, WeeklySummaryResponse)
        body = fake.calls[0]["json"]
        assert fake.calls[0]["path"] == "/llm/us-alpha-hrp-summary"
        assert body["top_n"] == 15
        assert body["hold_threshold"] == 30
        assert body["universe"] == "halal_new"
        assert body["selected_symbols"] == sticky.selected
        # Top 20 valid scores -> top_25 returns all 20, ordered by score desc.
        assert len(body["stage1_top_scores"]) == 20
        assert body["stage1_top_scores"][0]["symbol"] == "SYM0"
        assert body["stage1_top_scores"][0]["rank"] == 1
        assert body["stage1_top_scores"][-1]["symbol"] == "SYM19"
        assert body["stage1_top_scores"][-1]["rank"] == 20


class TestSendUSAlphaHrpEmail:
    def test_happy_path_includes_order_results(self):
        scores, sticky, stage2 = _alpha_payload_fixtures()
        summary = WeeklySummaryResponse(
            summary={"para_1_market_outlook": "x"},
            provider="openai",
            model_used="gpt-4o-mini",
            tokens_used=10,
        )
        order_results = SubmitOrdersResponse(
            account="hrp",
            orders_submitted=14,
            orders_failed=1,
            skipped=False,
            results=[],
        )
        fake = _FakeClient(
            {
                "/email/us-alpha-hrp-report": {
                    "is_success": True,
                    "subject": "US Alpha-HRP Report",
                    "body": "<html>x</html>",
                }
            }
        )
        with _patch_client(reporting_module, fake):
            result = reporting_module.send_us_alpha_hrp_email(
                summary=summary,
                scores=scores,
                sticky=sticky,
                stage2=stage2,
                universe="halal_new",
                top_n=15,
                hold_threshold=30,
                target_week_start="2026-04-27",
                target_week_end="2026-05-01",
                as_of_date="2026-04-28",
                order_results=order_results,
                skipped=False,
            )

        assert isinstance(result, WeeklyReportEmailResponse)
        body = fake.calls[0]["json"]
        assert fake.calls[0]["path"] == "/email/us-alpha-hrp-report"
        assert body["skipped"] is False
        assert body["order_results"]["orders_submitted"] == 14
        assert body["order_results"]["orders_failed"] == 1
        assert body["selected_symbols"] == sticky.selected
        assert body["stage2"]["symbols_used"] == 15

    def test_skip_path_marks_skipped_true(self):
        scores, sticky, stage2 = _alpha_payload_fixtures()
        summary = WeeklySummaryResponse(
            summary={"para_1_market_outlook": "x"},
            provider="openai",
            model_used="gpt-4o-mini",
            tokens_used=10,
        )
        order_results = SkippedSubmitResponse(account="hrp", skipped=True)
        fake = _FakeClient(
            {
                "/email/us-alpha-hrp-report": {
                    "is_success": True,
                    "subject": "US Alpha-HRP Report (skipped)",
                    "body": "<html>x</html>",
                }
            }
        )
        with _patch_client(reporting_module, fake):
            reporting_module.send_us_alpha_hrp_email(
                summary=summary,
                scores=scores,
                sticky=sticky,
                stage2=stage2,
                universe="halal_new",
                top_n=15,
                hold_threshold=30,
                target_week_start="2026-04-27",
                target_week_end="2026-05-01",
                as_of_date="2026-04-28",
                order_results=order_results,
                skipped=True,
            )

        body = fake.calls[0]["json"]
        assert body["skipped"] is True
        assert body["order_results"]["skipped"] is True
        assert body["order_results"]["orders_submitted"] == 0

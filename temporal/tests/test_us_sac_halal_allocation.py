"""Happy-path test for ``USSACHalalAllocationWorkflow``.

This is the parallel A/B sibling of ``USWeeklyAllocationWorkflow``;
it must:

1. Use the variant ``run_id`` form ``paper:halal:YYYY-MM-DD`` per
   AGENTS.md "Run identity & rerun semantics" (because it trades on
   the dedicated ``sac_halal`` IBKR paper account, isolated from
   every Alpaca-routed strategy by virtue of being on a different
   broker entirely).
2. Resolve next attempt against the ``sac_halal`` account only, via
   the IBKR sibling activity ``resolve_next_attempt_ibkr`` (the Alpaca
   ``resolve_next_attempt`` would 422 because ``AlpacaAccount`` no
   longer has a ``sac_halal`` entry post IBKR migration).
3. Read symbols from the ``halal`` SAC bucket (mandatory ``universe``
   arg on ``get_active_symbols``).
4. Send ``universe='halal'`` to ``infer_sac``, ``generate_summary``
   and ``send_weekly_email``.
5. Tag SAC orders with ``algorithm='sac_halal'``.
6. Submit through the IBKR ``sac_halal`` submitter
   (``submit_orders_ibkr_sac_halal``), NEVER through the legacy
   Alpaca ``submit_orders_sac`` activity, AND poll status through
   ``check_order_statuses_ibkr`` -- the IBKR sibling of
   ``check_order_statuses`` -- so the broker-agnostic
   ``sell_wait_buy`` helper never branches on ``account``.

The size of the active-symbols list is parametrized over [10, 14, 15]
to enforce the n-agnostic claim from the plan -- the legacy halal
universe is variable size depending on yfinance ETF top-holdings.
"""

from __future__ import annotations

import inspect

import pytest
from temporalio import activity

from activities.reporting import send_weekly_email as _send_weekly_email_signature
from models import (
    ActiveSymbolsResponse,
    AlpacaPortfolioResponse,
    GenerateOrdersResponse,
    OrderModel,
    OrderSummary,
    PositionModel,
    SACInferenceResponse,
    SkippedOrdersResponse,
    SkippedSubmitResponse,
    SubmitOrdersResponse,
    WeeklyReportEmailResponse,
    WeeklySummaryResponse,
)
from tests.harness import worker_with_activities
from workflows.us_sac_halal_allocation import USSACHalalAllocationWorkflow


def _make_active_symbols(n: int) -> ActiveSymbolsResponse:
    return ActiveSymbolsResponse(
        symbols=[f"H{i}" for i in range(n)],
        source_model="sac_halal",
        model_version="v2026-05-01-halal",
    )


def _make_sac_alloc(symbols: list[str]) -> SACInferenceResponse:
    even = round(0.95 / max(1, len(symbols)), 4)
    weights = {s: even for s in symbols}
    weights["CASH"] = round(1.0 - even * len(symbols), 4)
    return SACInferenceResponse(
        target_weights=weights,
        turnover=0.10,
        model_version="v2026-05-01-halal",
        target_week_start="2026-05-04",
        target_week_end="2026-05-08",
    )


def _make_orders() -> GenerateOrdersResponse:
    return GenerateOrdersResponse(
        orders=[
            OrderModel(
                client_order_id="paper:halal:2026-05-04:attempt-1:H0:BUY",
                symbol="H0",
                side="buy",
                qty=5.0,
                type="market",
                time_in_force="day",
            ),
        ],
        summary=OrderSummary(
            buys=1,
            sells=0,
            total_buy_value=500.0,
            total_sell_value=0,
            turnover_pct=5.0,
            skipped_small_orders=0,
            skipped_below_threshold=0,
        ),
        prices_used={"H0": 100.0},
    )


def _make_sac_halal_activities(
    *,
    n_symbols: int,
    sac_portfolio: AlpacaPortfolioResponse,
    lstm_resp,
    patchtst_resp,
    news_resp,
    fundamentals_resp,
    captured_calls: dict,
):
    """Mock activities matching the new ``USSACHalalAllocationWorkflow`` shape.

    Uses a thin local factory rather than the SAC-only harness because
    we want to capture the per-call args (universe, algorithm,
    accounts) and assert them against the mandatory-arg contract.
    """
    active_symbols = _make_active_symbols(n_symbols)
    sac_alloc = _make_sac_alloc(active_symbols.symbols)
    orders = _make_orders()

    @activity.defn(name="resolve_next_attempt_ibkr")
    def mock_resolve_next_attempt_ibkr(run_id, as_of_date, accounts) -> int:
        captured_calls["resolve_ibkr"] = {
            "run_id": run_id,
            "as_of_date": as_of_date,
            "accounts": list(accounts) if accounts else accounts,
        }
        return 1

    @activity.defn(name="resolve_next_attempt")
    def mock_resolve_next_attempt(run_id, as_of_date, accounts=None) -> int:
        # Post-IBKR-migration the halal workflow MUST resolve attempts
        # via the IBKR ledger (resolve_next_attempt_ibkr). The Alpaca
        # resolver hits ``/alpaca/order-history``, whose ``account``
        # query rejects ``sac_halal`` since the IBKR migration
        # stripped ``SAC_HALAL`` from ``AlpacaAccount``. Registering
        # this stub purely to detect accidental re-introduction.
        captured_calls["forbidden_resolve_next_attempt_alpaca"] = True
        return 1

    @activity.defn(name="get_active_symbols")
    def mock_get_active_symbols(universe: str) -> ActiveSymbolsResponse:
        captured_calls["get_active_symbols_universe"] = universe
        return active_symbols

    @activity.defn(name="get_ibkr_sac_halal_portfolio")
    def mock_get_ibkr_sac_halal_portfolio() -> AlpacaPortfolioResponse:
        return sac_portfolio

    @activity.defn(name="get_sac_halal_portfolio")
    def mock_get_sac_halal_portfolio() -> AlpacaPortfolioResponse:
        # Post-IBKR-migration, the halal workflow MUST NOT call the
        # legacy Alpaca sac_halal portfolio activity (it doesn't exist
        # in production any more). Registered here purely to detect
        # accidental re-introduction.
        captured_calls["forbidden_get_sac_halal_portfolio"] = True
        return AlpacaPortfolioResponse(cash=0.0, positions=[], open_orders_count=0)

    @activity.defn(name="get_sac_portfolio")
    def mock_get_sac_portfolio() -> AlpacaPortfolioResponse:
        # The halal workflow MUST NOT call the halal_filtered portfolio.
        captured_calls["forbidden_get_sac_portfolio"] = True
        return AlpacaPortfolioResponse(cash=0.0, positions=[], open_orders_count=0)

    @activity.defn(name="get_fundamentals")
    def mock_get_fundamentals(symbols):
        captured_calls["fundamentals_symbols_count"] = len(symbols)
        return fundamentals_resp

    @activity.defn(name="get_news_sentiment")
    def mock_get_news_sentiment(symbols, as_of_date, run_id):
        return news_resp

    @activity.defn(name="get_lstm_forecast")
    def mock_get_lstm_forecast(as_of_date, symbols=None):
        return lstm_resp

    @activity.defn(name="get_patchtst_forecast")
    def mock_get_patchtst_forecast(as_of_date, symbols=None):
        return patchtst_resp

    @activity.defn(name="infer_sac")
    def mock_infer_sac(portfolio, as_of_date, universe):
        captured_calls["infer_sac_universe"] = universe
        return sac_alloc

    @activity.defn(name="generate_orders_sac")
    def mock_generate_orders_sac(allocation, portfolio, run_id, attempt, algorithm):
        captured_calls["generate_orders_sac_algorithm"] = algorithm
        captured_calls["generate_orders_sac_run_id"] = run_id
        return orders

    @activity.defn(name="store_experience_sac")
    def mock_store_experience_sac(*args):
        captured_calls["store_experience_run_id"] = args[0]
        # ``universe`` is the trailing positional arg (10th); capture it
        # so we can assert the labeller will route this record to the
        # ``sac_halal`` Alpaca account (not the legacy ``sac`` account).
        captured_calls["store_experience_universe"] = args[9] if len(args) > 9 else None
        return None

    @activity.defn(name="submit_orders_ibkr_sac_halal")
    def mock_submit_orders_ibkr_sac_halal(orders_resp):
        captured_calls.setdefault("submit_calls", []).append("ibkr_sac_halal")
        if isinstance(orders_resp, SkippedOrdersResponse) or getattr(
            orders_resp, "skipped", False
        ):
            return SkippedSubmitResponse(account="sac_halal")
        return SubmitOrdersResponse(
            account="sac_halal",
            orders_submitted=1,
            orders_failed=0,
            skipped=False,
            results=[],
        )

    @activity.defn(name="submit_orders_sac_halal")
    def mock_submit_orders_sac_halal(orders_resp):
        # Post-IBKR-migration, the halal workflow MUST NOT submit
        # through the legacy Alpaca sac_halal submitter (it doesn't
        # exist in production any more).
        captured_calls["forbidden_submit_orders_sac_halal"] = True
        return SkippedSubmitResponse(account="sac_halal")

    @activity.defn(name="submit_orders_sac")
    def mock_submit_orders_sac(orders_resp):
        # The halal workflow MUST NOT submit to the sac (halal_filtered) account.
        captured_calls["forbidden_submit_orders_sac"] = True
        return SkippedSubmitResponse(account="sac")

    @activity.defn(name="check_order_statuses_ibkr")
    def mock_check_order_statuses_ibkr(account, client_order_ids):
        captured_calls.setdefault("check_status_ibkr_accounts", []).append(account)
        return [
            {"client_order_id": cid, "status": "filled"} for cid in client_order_ids
        ]

    @activity.defn(name="check_order_statuses")
    def mock_check_order_statuses(account, client_order_ids):
        # The halal workflow MUST poll IBKR statuses, not Alpaca ones.
        captured_calls["forbidden_check_order_statuses_alpaca"] = True
        return [
            {"client_order_id": cid, "status": "filled"} for cid in client_order_ids
        ]

    @activity.defn(name="get_order_history_ibkr_sac_halal")
    def mock_get_order_history_ibkr_sac_halal(after_date):
        captured_calls.setdefault("history_calls", []).append("ibkr_sac_halal")
        return []

    @activity.defn(name="get_order_history_sac_halal")
    def mock_get_order_history_sac_halal(after_date):
        captured_calls["forbidden_get_order_history_sac_halal"] = True
        return []

    @activity.defn(name="update_execution_sac")
    def mock_update_execution_sac(
        run_id, orders_resp, history, post_trade_portfolio=None
    ):
        captured_calls["update_execution_post_trade_portfolio"] = (
            post_trade_portfolio is not None
        )
        return None

    @activity.defn(name="generate_summary")
    def mock_generate_summary(
        lstm, patchtst, news, fundamentals, sac, universe
    ) -> WeeklySummaryResponse:
        captured_calls["summary_universe"] = universe
        return WeeklySummaryResponse(
            summary={"overview": "halal A/B summary"},
            provider="openai",
            model_used="gpt-5-mini",
            tokens_used=400,
        )

    @activity.defn(name="send_weekly_email")
    def mock_send_weekly_email(*args, **kwargs) -> WeeklyReportEmailResponse:
        # Bind to the real activity signature so test assertions read by
        # name rather than positional index. New optional kwargs added
        # later won't shift any indices here.
        target = getattr(
            _send_weekly_email_signature, "__wrapped__", _send_weekly_email_signature
        )
        bound = inspect.signature(target).bind_partial(*args, **kwargs)
        bound.apply_defaults()
        captured_calls["email_universe"] = bound.arguments.get("universe")
        captured_calls["email_order_details"] = bound.arguments.get("order_details")
        captured_calls["email_prior_allocation"] = bound.arguments.get(
            "prior_allocation"
        )
        return WeeklyReportEmailResponse(
            is_success=True,
            subject=(
                "US SAC (halal) Weekly Portfolio Analysis (2026-05-04 -> 2026-05-08)"
            ),
            body="<html>halal report</html>",
        )

    return [
        mock_resolve_next_attempt_ibkr,
        mock_resolve_next_attempt,
        mock_get_active_symbols,
        mock_get_ibkr_sac_halal_portfolio,
        mock_get_sac_halal_portfolio,
        mock_get_sac_portfolio,
        mock_get_fundamentals,
        mock_get_news_sentiment,
        mock_get_lstm_forecast,
        mock_get_patchtst_forecast,
        mock_infer_sac,
        mock_generate_orders_sac,
        mock_store_experience_sac,
        mock_submit_orders_ibkr_sac_halal,
        mock_submit_orders_sac_halal,
        mock_submit_orders_sac,
        mock_check_order_statuses_ibkr,
        mock_check_order_statuses,
        mock_get_order_history_ibkr_sac_halal,
        mock_get_order_history_sac_halal,
        mock_update_execution_sac,
        mock_generate_summary,
        mock_send_weekly_email,
    ]


class TestUSSACHalalAllocationHappyPath:
    """End-to-end happy path with ``n_stocks`` parametrization."""

    @pytest.mark.parametrize("n_symbols", [10, 14, 15])
    @pytest.mark.asyncio
    async def test_halal_allocation_runs_with_variable_n_stocks(
        self,
        n_symbols,
        lstm_resp,
        patchtst_resp,
        news_resp,
        fundamentals_resp,
    ):
        """halal SAC pipeline propagates universe + n_stocks end-to-end.

        Asserts:
          * run_id uses the ``paper:halal:`` variant prefix
          * resolve_next_attempt scopes to the ``sac_halal`` account
          * universe='halal' threads through to infer_sac / summary / email
          * algorithm='sac_halal' tags generate_orders_sac
          * submit goes through ``submit_orders_ibkr_sac_halal``
            (never the legacy Alpaca submitters)
          * the returned ``symbols_count`` matches the parametrized n
            (n-agnostic claim)
        """
        sac_portfolio = AlpacaPortfolioResponse(
            cash=10000.0,
            positions=[
                PositionModel(symbol="H0", qty=10.0, market_value=1000.0),
            ],
            open_orders_count=0,
        )

        captured: dict = {}
        activities = _make_sac_halal_activities(
            n_symbols=n_symbols,
            sac_portfolio=sac_portfolio,
            lstm_resp=lstm_resp,
            patchtst_resp=patchtst_resp,
            news_resp=news_resp,
            fundamentals_resp=fundamentals_resp,
            captured_calls=captured,
        )

        async with worker_with_activities(
            [USSACHalalAllocationWorkflow], activities
        ) as env:
            result = await env.client.execute_workflow(
                USSACHalalAllocationWorkflow.run,
                id=f"test-us-sac-halal-{n_symbols}",
                task_queue="test-queue",
            )

        # n-agnostic propagation
        assert result["symbols_count"] == n_symbols
        assert result["universe"] == "halal"

        # run_id MUST use the variant prefix; full form
        # paper:halal:YYYY-MM-DD per AGENTS.md.
        assert result["run_id"].startswith("paper:halal:")
        assert captured["resolve_ibkr"]["run_id"].startswith("paper:halal:")

        # resolve_next_attempt_ibkr scoped to the dedicated IBKR
        # sac_halal account. The Alpaca resolver MUST NOT be invoked
        # -- it would 422 because ``AlpacaAccount`` no longer has a
        # ``sac_halal`` entry post IBKR migration.
        assert captured["resolve_ibkr"]["accounts"] == ["sac_halal"]
        assert "forbidden_resolve_next_attempt_alpaca" not in captured

        # Mandatory universe arg propagated everywhere.
        assert captured["get_active_symbols_universe"] == "halal"
        assert captured["infer_sac_universe"] == "halal"
        assert captured["summary_universe"] == "halal"
        assert captured["email_universe"] == "halal"

        # Mandatory algorithm tag for orders.
        assert captured["generate_orders_sac_algorithm"] == "sac_halal"
        assert captured["generate_orders_sac_run_id"].startswith("paper:halal:")

        # Experience records share model_type='sac' but are isolated by
        # the run_id prefix on disk -- store activity received the
        # variant run_id.
        assert captured["store_experience_run_id"].startswith("paper:halal:")
        # universe='halal' MUST be plumbed onto the experience record so
        # /experience/label/sac routes this record to the sac_halal
        # Alpaca account (not the legacy sac account) at label time.
        assert captured["store_experience_universe"] == "halal"
        # Post-trade portfolio MUST flow into update_execution so the
        # labeller never falls back to a live Alpaca query for
        # actual_weights.
        assert captured["update_execution_post_trade_portfolio"] is True

        # Submit went through the IBKR sac_halal submitter, never the
        # legacy Alpaca submitter (sac or sac_halal). sell-wait-buy
        # invokes the submitter twice (sells then buys).
        submit_calls = captured.get("submit_calls", [])
        assert submit_calls and all(c == "ibkr_sac_halal" for c in submit_calls)
        assert "forbidden_submit_orders_sac" not in captured
        assert "forbidden_submit_orders_sac_halal" not in captured
        assert "forbidden_get_sac_portfolio" not in captured
        assert "forbidden_get_sac_halal_portfolio" not in captured

        # Sell-wait-buy polling (if any) must scope to the IBKR
        # status-check activity AND the sac_halal account.
        for acct in captured.get("check_status_ibkr_accounts", []):
            assert acct == "sac_halal"
        assert "forbidden_check_order_statuses_alpaca" not in captured

        # Order history must come from the IBKR ledger, not the legacy
        # Alpaca route.
        assert captured.get("history_calls") == ["ibkr_sac_halal"]
        assert "forbidden_get_order_history_sac_halal" not in captured

        # Email subject reflects the new universe-tagged form.
        assert "halal" in result["email"]["subject"]
        assert result["email"]["is_success"] is True

        # Per the email-enhancement plan, the per-order detail table
        # plus the "Going Into This Week" prior-allocation snapshot
        # must reach send_weekly_email on the halal A/B path too.
        # Prior allocation is sourced from the live IBKR sac_halal
        # portfolio (not from the DB).
        def _attr(obj, name):
            return obj[name] if isinstance(obj, dict) else getattr(obj, name)

        order_details = captured.get("email_order_details")
        assert order_details is not None
        assert len(order_details) >= 1
        first_detail = order_details[0]
        assert _attr(first_detail, "symbol")
        assert _attr(first_detail, "side") in {"buy", "sell"}
        assert _attr(first_detail, "stop_loss_reason") in {
            "atr14",
            "atr_unavailable",
            "sell_no_stop",
        }

        prior = captured.get("email_prior_allocation")
        assert prior is not None
        assert _attr(prior, "source_label")  # IBKR live-broker label
        assert isinstance(_attr(prior, "weights"), dict)

    @pytest.mark.asyncio
    async def test_halal_skipped_when_open_orders(
        self,
        lstm_resp,
        patchtst_resp,
        news_resp,
        fundamentals_resp,
    ):
        """Open orders on sac_halal account -> SAC step is skipped.

        Mirrors the existing skip path on ``USWeeklyAllocationWorkflow``
        but verifies the dedicated account guards the halal variant.
        """
        sac_portfolio = AlpacaPortfolioResponse(
            cash=10000.0,
            positions=[],
            open_orders_count=2,
        )

        captured: dict = {}
        activities = _make_sac_halal_activities(
            n_symbols=12,
            sac_portfolio=sac_portfolio,
            lstm_resp=lstm_resp,
            patchtst_resp=patchtst_resp,
            news_resp=news_resp,
            fundamentals_resp=fundamentals_resp,
            captured_calls=captured,
        )

        async with worker_with_activities(
            [USSACHalalAllocationWorkflow], activities
        ) as env:
            result = await env.client.execute_workflow(
                USSACHalalAllocationWorkflow.run,
                id="test-us-sac-halal-skip",
                task_queue="test-queue",
            )

        assert result["sac"]["skipped"] is True
        assert "SAC" in result["skipped_algorithms"]
        # Universe MUST still be threaded through summary / email even
        # when SAC was skipped (so the LLM and inbox subject still
        # identify the A/B run correctly).
        assert captured["summary_universe"] == "halal"
        assert captured["email_universe"] == "halal"

"""Activity-mock factory for the SAC-only ``USWeeklyAllocationWorkflow``.

Retired HRP activities are intentionally registered but raise (or
record) when invoked, proving the workflow no longer depends on them.
"""

from __future__ import annotations

import inspect

from temporalio import activity

from activities.reporting import send_weekly_email
from models import (
    ActiveSymbolsResponse,
    AdjustedClosesResponse,
    AlpacaPortfolioResponse,
    MarketClockResponse,
    MarketHistoryResponse,
    SkippedOrdersResponse,
    SkippedSubmitResponse,
    WeeklySummaryResponse,
)

# momentum_12_1 needs >= MOM_12_1_LOOKBACK_BARS(252) + 1 daily closes;
# the harness's synthetic series is a simple increasing sequence so
# momentum values are non-zero (never a silent zero-fill).
_DEFAULT_MOMENTUM_BARS = 253

# Default clock payload: market is open. Tests that need a closed
# market or a specific ``next_open`` override via the
# ``get_alpaca_clock_fn`` factory hook below. The non-zero
# next_open / next_close strings are placeholders -- the helper
# only inspects ``is_open`` when the market is reported open.
_DEFAULT_CLOCK_OPEN = MarketClockResponse(
    timestamp="2026-05-11T13:30:00+00:00",
    is_open=True,
    next_open="2026-05-12T13:30:00+00:00",
    next_close="2026-05-11T20:00:00+00:00",
)


def _bind_email_args(args, kwargs) -> dict:
    """Bind to ``send_weekly_email``'s signature for assertion-by-name."""
    target = getattr(send_weekly_email, "__wrapped__", send_weekly_email)
    bound = inspect.signature(target).bind_partial(*args, **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)


def make_sac_only_activities(
    *,
    active_symbols,
    sac_portfolio,
    news_resp,
    patchtst_resp,
    sac_alloc,
    sac_orders,
    sac_submit_resp,
    summary_resp,
    email_resp,
    forbidden_calls: list[str] | None = None,
    check_order_statuses_fn=None,
    summary_calls: list[dict] | None = None,
    email_calls: list[dict] | None = None,
    store_experience_calls: list[dict] | None = None,
    update_execution_calls: list[dict] | None = None,
    get_alpaca_clock_fn=None,
    get_alpaca_clock_calls: list[None] | None = None,
    price_fetch_calls: list[list[str]] | None = None,
):
    """Build mock activities for the SAC-only ``USWeeklyAllocationWorkflow``."""

    def _forbid(name: str):
        if forbidden_calls is not None:
            forbidden_calls.append(name)
            return None
        raise AssertionError(f"Workflow must not invoke retired HRP activity '{name}'")

    @activity.defn(name="resolve_next_attempt")
    def mock_resolve_next_attempt(run_id, as_of_date, accounts=None) -> int:
        return 1

    @activity.defn(name="get_active_symbols")
    def mock_get_active_symbols(universe: str) -> ActiveSymbolsResponse:
        return active_symbols

    @activity.defn(name="get_sac_portfolio")
    def mock_get_sac_portfolio() -> AlpacaPortfolioResponse:
        return sac_portfolio

    @activity.defn(name="get_hrp_portfolio")
    def mock_get_hrp_portfolio() -> AlpacaPortfolioResponse:
        return _forbid("get_hrp_portfolio") or AlpacaPortfolioResponse(
            cash=0.0, positions=[], open_orders_count=0
        )

    @activity.defn(name="allocate_hrp")
    def mock_allocate_hrp(symbols, as_of_date, lookback_days=252):
        _forbid("allocate_hrp")
        return None

    @activity.defn(name="submit_orders_hrp")
    def mock_submit_orders_hrp(orders):
        _forbid("submit_orders_hrp")
        return SkippedSubmitResponse(account="hrp")

    @activity.defn(name="get_news_sentiment")
    def mock_get_news_sentiment(symbols, as_of_date, run_id):
        return news_resp

    @activity.defn(name="get_lstm_forecast")
    def mock_get_lstm_forecast(as_of_date, symbols=None):
        _forbid("get_lstm_forecast")
        return None

    @activity.defn(name="get_patchtst_forecast")
    def mock_get_patchtst_forecast(as_of_date, symbols=None):
        return patchtst_resp

    @activity.defn(name="get_adjusted_closes")
    def mock_get_adjusted_closes(
        symbols, as_of_date, lookback_bars=_DEFAULT_MOMENTUM_BARS
    ):
        if price_fetch_calls is not None:
            price_fetch_calls.append(list(symbols))
        return AdjustedClosesResponse(
            as_of_date=as_of_date,
            adjusted_closes={
                symbol: [100.0 + i * 0.1 for i in range(lookback_bars)]
                for symbol in symbols
            },
            execution_prices={symbol: 125.2 for symbol in symbols},
            provenance={"provider": "test"},
        )

    @activity.defn(name="get_market_history")
    def mock_get_market_history(training_cutoff_date, as_of_date):
        return MarketHistoryResponse(
            start_date="2026-05-09",
            as_of_date=as_of_date,
            rows=[],
            provenance={"provider": "test"},
        )

    @activity.defn(name="infer_sac")
    def mock_infer_sac(
        portfolio,
        as_of_date,
        universe,
        symbols,
        news,
        patchtst,
        prices,
        market,
    ):
        return sac_alloc

    @activity.defn(name="generate_orders_sac")
    def mock_generate_orders_sac(allocation, portfolio, run_id, attempt, algorithm):
        return sac_orders

    @activity.defn(name="store_experience_sac")
    def mock_store_experience_sac(*args):
        if store_experience_calls is not None:
            # ``universe`` is the trailing positional arg (5th).
            allocation = args[3] if len(args) > 3 else None
            decision_state = (
                allocation.get("decision_state")
                if isinstance(allocation, dict)
                else getattr(allocation, "decision_state", None)
            )
            state_digest = (
                allocation.get("state_digest")
                if isinstance(allocation, dict)
                else getattr(allocation, "state_digest", None)
            )
            store_experience_calls.append(
                {
                    "run_id": args[0] if args else None,
                    "universe": args[4] if len(args) > 4 else None,
                    "decision_state": decision_state,
                    "state_digest": state_digest,
                }
            )
        return None

    @activity.defn(name="submit_orders_sac")
    def mock_submit_orders_sac(orders):
        if isinstance(orders, SkippedOrdersResponse) or getattr(
            orders, "skipped", False
        ):
            return SkippedSubmitResponse(account="sac")
        return sac_submit_resp

    @activity.defn(name="check_order_statuses")
    def mock_check_order_statuses(account, client_order_ids):
        if check_order_statuses_fn is not None:
            return check_order_statuses_fn(account, client_order_ids)
        return [
            {"client_order_id": cid, "status": "filled"} for cid in client_order_ids
        ]

    @activity.defn(name="get_alpaca_clock")
    def mock_get_alpaca_clock() -> MarketClockResponse:
        if get_alpaca_clock_calls is not None:
            get_alpaca_clock_calls.append(None)
        if get_alpaca_clock_fn is not None:
            return get_alpaca_clock_fn()
        return _DEFAULT_CLOCK_OPEN

    @activity.defn(name="get_order_history_sac")
    def mock_get_order_history_sac(after_date):
        return []

    @activity.defn(name="update_execution_sac")
    def mock_update_execution_sac(run_id, orders, history, post_trade_portfolio=None):
        if update_execution_calls is not None:
            update_execution_calls.append(
                {
                    "run_id": run_id,
                    "has_post_trade_portfolio": post_trade_portfolio is not None,
                }
            )
        return None

    def _coerce(value, key):
        if value is None:
            return None
        if hasattr(value, key):
            return getattr(value, key)
        if isinstance(value, dict):
            return value.get(key)
        return None

    @activity.defn(name="generate_summary")
    def mock_generate_summary(patchtst, news, sac, universe) -> WeeklySummaryResponse:
        if summary_calls is not None:
            summary_calls.append(
                {
                    "sac_skipped": _coerce(sac, "skipped") or False,
                    "sac_model_version": _coerce(sac, "model_version"),
                    "universe": universe,
                }
            )
        return summary_resp

    @activity.defn(name="send_weekly_email")
    def mock_send_weekly_email(*args, **kwargs):
        if email_calls is not None:
            bound = _bind_email_args(args, kwargs)
            email_calls.append(
                {
                    "sac_skipped": _coerce(bound.get("sac"), "skipped") or False,
                    "sac_submit_skipped": (
                        _coerce(bound.get("sac_submit"), "skipped") or False
                    ),
                    "universe": bound.get("universe"),
                    "order_details": bound.get("order_details"),
                    "prior_allocation": bound.get("prior_allocation"),
                }
            )
        return email_resp

    return [
        mock_resolve_next_attempt,
        mock_get_active_symbols,
        mock_get_sac_portfolio,
        mock_get_hrp_portfolio,
        mock_allocate_hrp,
        mock_submit_orders_hrp,
        mock_get_news_sentiment,
        mock_get_lstm_forecast,
        mock_get_patchtst_forecast,
        mock_get_adjusted_closes,
        mock_get_market_history,
        mock_infer_sac,
        mock_generate_orders_sac,
        mock_store_experience_sac,
        mock_submit_orders_sac,
        mock_check_order_statuses,
        mock_get_alpaca_clock,
        mock_get_order_history_sac,
        mock_update_execution_sac,
        mock_generate_summary,
        mock_send_weekly_email,
    ]

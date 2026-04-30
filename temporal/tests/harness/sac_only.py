"""Activity-mock factory for the SAC-only ``USWeeklyAllocationWorkflow``.

Retired HRP activities are intentionally registered but raise (or
record) when invoked, proving the workflow no longer depends on them.
"""

from __future__ import annotations

from temporalio import activity

from models import (
    ActiveSymbolsResponse,
    AlpacaPortfolioResponse,
    SkippedOrdersResponse,
    SkippedSubmitResponse,
    WeeklySummaryResponse,
)


def make_sac_only_activities(
    *,
    active_symbols,
    sac_portfolio,
    fundamentals_resp,
    news_resp,
    lstm_resp,
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
):
    """Build mock activities for the SAC-only ``USWeeklyAllocationWorkflow``."""

    def _forbid(name: str):
        if forbidden_calls is not None:
            forbidden_calls.append(name)
            return None
        raise AssertionError(f"Workflow must not invoke retired HRP activity '{name}'")

    @activity.defn(name="resolve_next_attempt")
    def mock_resolve_next_attempt(run_id, as_of_date) -> int:
        return 1

    @activity.defn(name="get_active_symbols")
    def mock_get_active_symbols() -> ActiveSymbolsResponse:
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

    @activity.defn(name="generate_orders_hrp")
    def mock_generate_orders_hrp(*args, **kwargs):
        _forbid("generate_orders_hrp")
        return SkippedOrdersResponse(skipped=True, algorithm="hrp")

    @activity.defn(name="submit_orders_hrp")
    def mock_submit_orders_hrp(orders):
        _forbid("submit_orders_hrp")
        return SkippedSubmitResponse(account="hrp")

    @activity.defn(name="get_fundamentals")
    def mock_get_fundamentals(symbols):
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
    def mock_infer_sac(portfolio, as_of_date):
        return sac_alloc

    @activity.defn(name="generate_orders_sac")
    def mock_generate_orders_sac(allocation, portfolio, run_id, attempt):
        return sac_orders

    @activity.defn(name="store_experience_sac")
    def mock_store_experience_sac(*args):
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

    @activity.defn(name="get_order_history_sac")
    def mock_get_order_history_sac(after_date):
        return []

    @activity.defn(name="update_execution_sac")
    def mock_update_execution_sac(run_id, orders, history):
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
    def mock_generate_summary(
        lstm, patchtst, news, fundamentals, sac
    ) -> WeeklySummaryResponse:
        if summary_calls is not None:
            summary_calls.append(
                {
                    "sac_skipped": _coerce(sac, "skipped") or False,
                    "sac_model_version": _coerce(sac, "model_version"),
                }
            )
        return summary_resp

    @activity.defn(name="send_weekly_email")
    def mock_send_weekly_email(*args, **kwargs):
        if email_calls is not None:
            sac = args[3] if len(args) > 3 else None
            sac_submit = args[4] if len(args) > 4 else None
            email_calls.append(
                {
                    "sac_skipped": _coerce(sac, "skipped") or False,
                    "sac_submit_skipped": _coerce(sac_submit, "skipped") or False,
                }
            )
        return email_resp

    return [
        mock_resolve_next_attempt,
        mock_get_active_symbols,
        mock_get_sac_portfolio,
        mock_get_hrp_portfolio,
        mock_allocate_hrp,
        mock_generate_orders_hrp,
        mock_submit_orders_hrp,
        mock_get_fundamentals,
        mock_get_news_sentiment,
        mock_get_lstm_forecast,
        mock_get_patchtst_forecast,
        mock_infer_sac,
        mock_generate_orders_sac,
        mock_store_experience_sac,
        mock_submit_orders_sac,
        mock_check_order_statuses,
        mock_get_order_history_sac,
        mock_update_execution_sac,
        mock_generate_summary,
        mock_send_weekly_email,
    ]

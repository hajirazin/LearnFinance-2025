"""Activity-mock factory for ``USAlphaHRPWorkflow`` tests.

Lifted out of ``test_us_alpha_hrp.py`` so the four scenario files
(happy / skip / sell-wait-buy / attempt-isolation) share a single
mock-activity harness.
"""

from __future__ import annotations

from temporalio import activity

from models import (
    AlpacaPortfolioResponse,
    SkippedOrdersResponse,
    SkippedSubmitResponse,
)


def make_us_alpha_hrp_activities(
    *,
    universe_data,
    hrp_portfolio,
    scores,
    stage2,
    sticky,
    record_final,
    orders,
    submit_resp,
    summary,
    email,
    resolve_calls=None,
    score_calls=None,
    select_calls=None,
    hrp_calls=None,
    record_final_calls=None,
    submit_calls=None,
    check_order_statuses_fn=None,
):
    """Build mock activity functions for ``USAlphaHRPWorkflow``."""

    @activity.defn(name="resolve_next_attempt")
    def mock_resolve_next_attempt(run_id, as_of_date, accounts=None) -> int:
        if resolve_calls is not None:
            resolve_calls.append(
                {"run_id": run_id, "as_of_date": as_of_date, "accounts": accounts}
            )
        return 1

    @activity.defn(name="fetch_halal_new_universe")
    def mock_fetch_universe() -> dict:
        return universe_data

    @activity.defn(name="get_hrp_portfolio")
    def mock_get_hrp_portfolio() -> AlpacaPortfolioResponse:
        return hrp_portfolio

    @activity.defn(name="score_halal_new_with_patchtst")
    def mock_score(symbols, as_of_date, min_predictions=15):
        if score_calls is not None:
            score_calls.append(
                {
                    "symbols": symbols,
                    "as_of_date": as_of_date,
                    "min_predictions": min_predictions,
                }
            )
        return scores

    @activity.defn(name="select_rank_band_top_n")
    def mock_select_rank_band(
        scores_arg, universe, year_week, as_of_date, run_id, top_n, hold_threshold
    ):
        if select_calls is not None:
            select_calls.append(
                {
                    "scores_count": len(scores_arg),
                    "universe": universe,
                    "year_week": year_week,
                    "as_of_date": as_of_date,
                    "run_id": run_id,
                    "top_n": top_n,
                    "hold_threshold": hold_threshold,
                }
            )
        return sticky

    @activity.defn(name="allocate_hrp")
    def mock_allocate_hrp(symbols, as_of_date, lookback_days=252):
        if hrp_calls is not None:
            hrp_calls.append(
                {
                    "symbols": symbols,
                    "as_of_date": as_of_date,
                    "lookback_days": lookback_days,
                }
            )
        return stage2

    @activity.defn(name="record_final_weights")
    def mock_record_final(universe, year_week, final_weights_pct):
        if record_final_calls is not None:
            record_final_calls.append(
                {
                    "universe": universe,
                    "year_week": year_week,
                    "n_weights": len(final_weights_pct),
                }
            )
        return record_final

    @activity.defn(name="generate_orders_alpha_hrp")
    def mock_generate_orders_alpha_hrp(allocation, portfolio, run_id, attempt):
        return orders

    @activity.defn(name="submit_orders_hrp")
    def mock_submit_orders_hrp(orders_resp):
        if submit_calls is not None:
            submit_calls.append(orders_resp)
        if isinstance(orders_resp, SkippedOrdersResponse) or getattr(
            orders_resp, "skipped", False
        ):
            return SkippedSubmitResponse(account="hrp")
        return submit_resp

    @activity.defn(name="check_order_statuses")
    def mock_check_order_statuses(account, client_order_ids):
        if check_order_statuses_fn is not None:
            return check_order_statuses_fn(account, client_order_ids)
        return [
            {"client_order_id": cid, "status": "filled"} for cid in client_order_ids
        ]

    @activity.defn(name="generate_us_alpha_hrp_summary")
    def mock_generate_summary(*args, **kwargs):
        return summary

    @activity.defn(name="send_us_alpha_hrp_email")
    def mock_send_email(*args, **kwargs):
        return email

    return [
        mock_resolve_next_attempt,
        mock_fetch_universe,
        mock_get_hrp_portfolio,
        mock_score,
        mock_select_rank_band,
        mock_allocate_hrp,
        mock_record_final,
        mock_generate_orders_alpha_hrp,
        mock_submit_orders_hrp,
        mock_check_order_statuses,
        mock_generate_summary,
        mock_send_email,
    ]

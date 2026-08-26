"""Weekly ppo_discovery allocation: news-conditioned PPO on frozen halal_new."""

from __future__ import annotations

from datetime import date, timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError

from workflows._order_execution import (
    SHORT_TIMEOUT,
    sell_wait_buy,
    split_orders_by_side,
)
from workflows._run_identity import in_ist

with workflow.unsafe.imports_passed_through():
    from activities.inference import get_monday_decision_window
    from activities.portfolio import resolve_next_attempt
    from activities.ppo_discovery_execution import (
        generate_orders_ppo_discovery,
        get_order_history_ppo_discovery,
        label_ppo_discovery_experience,
        store_experience_ppo_discovery,
        submit_orders_ppo_discovery,
        update_execution_ppo_discovery,
    )
    from activities.ppo_discovery_inference import (
        build_ppo_discovery_state,
        get_ppo_discovery_portfolio,
        infer_ppo_discovery,
    )
    from activities.ppo_discovery_reporting import (
        generate_ppo_discovery_summary,
        send_ppo_discovery_email,
    )
    from models.ppo_discovery import PPOInferenceResponse

ACTIVITY_TIMEOUT = timedelta(minutes=5)
PPO_STATE_TIMEOUT = timedelta(hours=2)
PPO_STATE_HEARTBEAT = timedelta(seconds=60)
ACTIVITY_RETRY = 2


def experience_week_bounds(as_of_date: str) -> tuple[str, str]:
    """Monday decision date and the following Monday used for open-to-open labels."""
    week_start = date.fromisoformat(as_of_date)
    return as_of_date, (week_start + timedelta(days=7)).isoformat()


def _activity_failure_reason(exc: BaseException) -> str:
    cause = getattr(exc, "cause", None)
    if cause is not None:
        return str(cause)
    return str(exc)


def _portfolio_weights(portfolio) -> dict[str, float]:
    total = float(portfolio.cash) + sum(p.market_value for p in portfolio.positions)
    if total <= 0:
        return {"CASH": 1.0}
    weights = {p.symbol: p.market_value / total for p in portfolio.positions}
    weights["CASH"] = float(portfolio.cash) / total
    return weights


@workflow.defn
class USPPODiscoveryAllocationWorkflow:
    @workflow.run
    async def run(self) -> dict:
        now_ist = in_ist(workflow.now())
        as_of_date = now_ist.date().isoformat()
        run_id = f"paper:halal_new:{as_of_date}"
        decision_window = await workflow.execute_activity(
            get_monday_decision_window,
            args=[as_of_date],
            start_to_close_timeout=SHORT_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
        )
        as_of = decision_window.cutoff.isoformat()

        attempt = await workflow.execute_activity(
            resolve_next_attempt,
            args=[run_id, as_of_date, ["ppo_discovery"]],
            start_to_close_timeout=SHORT_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
        )
        portfolio = await workflow.execute_activity(
            get_ppo_discovery_portfolio,
            start_to_close_timeout=SHORT_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
        )
        empty = PPOInferenceResponse(
            model_type="ppo_discovery",
            model_version="",
            universe="halal_new",
            selected_symbols=[],
            selection_order=[],
            k=0,
            percentage_weights={"CASH": 1.0},
            state_digest="",
            evidence_manifest_sha256="",
            skipped=True,
            skip_reason="open_orders",
        )
        if portfolio.open_orders_count > 0:
            summary = await workflow.execute_activity(
                generate_ppo_discovery_summary,
                args=[empty],
                start_to_close_timeout=ACTIVITY_TIMEOUT,
                retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
            )
            email = await workflow.execute_activity(
                send_ppo_discovery_email,
                args=[empty, summary, as_of_date, True, "open_orders"],
                start_to_close_timeout=ACTIVITY_TIMEOUT,
                retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
            )
            return {
                "as_of_date": as_of_date,
                "run_id": run_id,
                "skipped": True,
                "skip_reason": "open_orders",
                "email": {"is_success": email.is_success, "subject": email.subject},
            }

        try:
            state = await workflow.execute_activity(
                build_ppo_discovery_state,
                args=[as_of, run_id, attempt, _portfolio_weights(portfolio)],
                start_to_close_timeout=PPO_STATE_TIMEOUT,
                heartbeat_timeout=PPO_STATE_HEARTBEAT,
                retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
            )
            allocation = await workflow.execute_activity(
                infer_ppo_discovery,
                args=[state, state["state_digest"]],
                start_to_close_timeout=ACTIVITY_TIMEOUT,
                retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
            )
        except ActivityError as exc:
            empty.skip_reason = _activity_failure_reason(exc)
            summary = await workflow.execute_activity(
                generate_ppo_discovery_summary,
                args=[empty],
                start_to_close_timeout=ACTIVITY_TIMEOUT,
            )
            email = await workflow.execute_activity(
                send_ppo_discovery_email,
                args=[empty, summary, as_of_date, True, empty.skip_reason],
                start_to_close_timeout=ACTIVITY_TIMEOUT,
            )
            return {
                "as_of_date": as_of_date,
                "run_id": run_id,
                "skipped": True,
                "skip_reason": empty.skip_reason,
                "email": {"is_success": email.is_success, "subject": email.subject},
            }

        week_start, week_end = experience_week_bounds(as_of_date)
        await workflow.execute_activity(
            store_experience_ppo_discovery,
            args=[
                run_id,
                week_start,
                week_end,
                allocation,
                state,
            ],
            start_to_close_timeout=SHORT_TIMEOUT,
        )
        orders = await workflow.execute_activity(
            generate_orders_ppo_discovery,
            args=[allocation, portfolio, run_id, attempt, "all"],
            start_to_close_timeout=SHORT_TIMEOUT,
        )
        sells, buys = split_orders_by_side(orders)
        submit = await sell_wait_buy(
            "ppo_discovery", sells, buys, orders, submit_orders_ppo_discovery
        )
        history = await workflow.execute_activity(
            get_order_history_ppo_discovery,
            args=[as_of_date],
            start_to_close_timeout=SHORT_TIMEOUT,
        )
        post_trade = await workflow.execute_activity(
            get_ppo_discovery_portfolio,
            start_to_close_timeout=SHORT_TIMEOUT,
        )
        await workflow.execute_activity(
            update_execution_ppo_discovery,
            args=[run_id, orders, history, post_trade],
            start_to_close_timeout=SHORT_TIMEOUT,
        )
        await workflow.execute_activity(
            label_ppo_discovery_experience,
            args=[None],
            start_to_close_timeout=SHORT_TIMEOUT,
        )
        summary = await workflow.execute_activity(
            generate_ppo_discovery_summary,
            args=[allocation],
            start_to_close_timeout=ACTIVITY_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
        )
        email = await workflow.execute_activity(
            send_ppo_discovery_email,
            args=[allocation, summary, as_of_date, False, ""],
            start_to_close_timeout=ACTIVITY_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
        )
        return {
            "as_of_date": as_of_date,
            "run_id": run_id,
            "attempt": attempt,
            "k": allocation.k,
            "selected_symbols": allocation.selected_symbols,
            "model_version": allocation.model_version,
            "skipped": False,
            "submit": submit.model_dump() if hasattr(submit, "model_dump") else submit,
            "email": {"is_success": email.is_success, "subject": email.subject},
        }

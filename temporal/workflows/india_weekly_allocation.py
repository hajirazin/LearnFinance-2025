"""India weekly allocation workflow.

Runs every Monday at 09:00 IST:
1. Validate halal_india universe
2. Run HRP allocation with universe=halal_india
3. Generate India AI summary via LLM
4. Send India weekly report email

No forecasters, no RL allocators, no order execution.
"""

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from activities.inference import allocate_hrp, get_halal_india_universe
    from activities.reporting import generate_india_summary, send_india_weekly_email

ACTIVITY_TIMEOUT = timedelta(minutes=5)
ACTIVITY_RETRY = 2


@workflow.defn
class IndiaWeeklyAllocationWorkflow:
    @workflow.run
    async def run(self) -> dict:
        now_ist = workflow.now().astimezone()
        as_of_date = now_ist.strftime("%Y-%m-%d")
        target_week_start = as_of_date
        target_week_end = (now_ist + timedelta(days=4)).strftime("%Y-%m-%d")

        workflow.logger.info(
            f"India weekly allocation: as_of={as_of_date}, "
            f"week={target_week_start} -> {target_week_end}"
        )

        # Phase 0: Validate universe
        universe_data = await workflow.execute_activity(
            get_halal_india_universe,
            start_to_close_timeout=ACTIVITY_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
        )
        symbols = [s["symbol"] for s in universe_data.get("stocks", [])]
        stock_count = len(symbols)

        # Phase 1: HRP allocation
        hrp = await workflow.execute_activity(
            allocate_hrp,
            args=[symbols, as_of_date],
            start_to_close_timeout=ACTIVITY_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
        )

        # Phase 2: AI summary
        summary = await workflow.execute_activity(
            generate_india_summary,
            args=[hrp, "halal_india"],
            start_to_close_timeout=ACTIVITY_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
        )

        # Phase 3: Email
        email_result = await workflow.execute_activity(
            send_india_weekly_email,
            args=[
                summary,
                hrp,
                "halal_india",
                target_week_start,
                target_week_end,
                as_of_date,
            ],
            start_to_close_timeout=ACTIVITY_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
        )

        return {
            "as_of_date": as_of_date,
            "target_week_start": target_week_start,
            "target_week_end": target_week_end,
            "universe_stocks": stock_count,
            "hrp_symbols": hrp.symbols_used,
            "summary_provider": summary.provider,
            "email": {
                "is_success": email_result.is_success,
                "subject": email_result.subject,
            },
        }

"""India weekly email workflow for LearnFinance-2025.

This flow runs every Monday at 09:00 IST and executes the India HRP pipeline:
1. Validate halal_india universe (Phase 0)
2. Run HRP allocation with universe=halal_india (Phase 1)
3. Generate India AI summary via LLM (Phase 2)
4. Send India weekly report email (Phase 3)

No forecasters, no RL allocators, no order execution. Those are deferred
to a future plan (Angel One integration + LSTM/PatchTST for India).
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from prefect import flow
from prefect.logging import get_run_logger

from flows.tasks import (
    allocate_hrp,
    generate_india_summary,
    get_halal_india_universe,
    send_india_weekly_email,
)


@flow(
    name="India Weekly Email",
    description="India HRP allocation analysis and weekly email report",
    retries=0,
    timeout_seconds=3600,
    persist_result=True,
)
def india_weekly_email_flow():
    """Run India weekly pipeline: universe -> HRP -> AI summary -> email."""
    logger = get_run_logger()

    now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
    as_of_date = now_ist.strftime("%Y-%m-%d")
    target_week_start = as_of_date
    target_week_end = (now_ist + timedelta(days=4)).strftime("%Y-%m-%d")

    logger.info(
        f"India weekly email flow starting: as_of={as_of_date}, "
        f"week={target_week_start} -> {target_week_end}"
    )

    # Phase 0: Validate universe
    logger.info("=== Phase 0: Validate Universe ===")
    universe_data = get_halal_india_universe()
    stock_count = len(universe_data.get("stocks", []))
    logger.info(f"Universe validated: {stock_count} stocks")

    # Phase 1: HRP allocation
    logger.info("=== Phase 1: HRP Allocation ===")
    hrp = allocate_hrp(as_of_date=as_of_date, universe="halal_india")
    logger.info(f"HRP done: {hrp.symbols_used} symbols allocated")

    # Phase 2: AI summary
    logger.info("=== Phase 2: AI Summary ===")
    summary = generate_india_summary(hrp)
    logger.info(f"Summary generated via {summary.provider}")

    # Phase 3: Email
    logger.info("=== Phase 3: Send Email ===")
    email_result = send_india_weekly_email(
        summary=summary,
        hrp=hrp,
        target_week_start=target_week_start,
        target_week_end=target_week_end,
        as_of_date=as_of_date,
    )
    logger.info(f"Email result: success={email_result.is_success}")

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


if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        india_weekly_email_flow()
    else:
        # Monday 09:00 IST = 03:30 UTC
        india_weekly_email_flow.serve(
            name="india-weekly-email",
            cron="30 3 * * 1",
        )

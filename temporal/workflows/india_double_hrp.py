"""India Double HRP workflow.

Two-stage Hierarchical Risk Parity:
1. Fetch full Nifty Shariah 500 universe (~210 stocks)
2. Run HRP with 3-year lookback to rank all stocks
3. Select top 15 by HRP weight
4. Re-run HRP on those 15 with 1-year lookback for final allocation
5. Generate AI summary + send email with both stages
"""

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from activities.inference import allocate_hrp
    from activities.reporting import generate_double_hrp_summary, send_double_hrp_email
    from activities.training import fetch_nifty_shariah_500_universe

ACTIVITY_TIMEOUT = timedelta(minutes=5)
HRP_TIMEOUT = timedelta(minutes=10)
ACTIVITY_RETRY = 2

UNIVERSE_LABEL = "nifty_shariah_500"
STAGE1_LOOKBACK = 756  # ~3 years
STAGE2_LOOKBACK = 252  # ~1 year
TOP_N = 15


@workflow.defn
class IndiaDoubleHRPWorkflow:
    @workflow.run
    async def run(self) -> dict:
        now_ist = workflow.now().astimezone()
        as_of_date = now_ist.strftime("%Y-%m-%d")
        target_week_start = as_of_date
        target_week_end = (now_ist + timedelta(days=4)).strftime("%Y-%m-%d")

        workflow.logger.info(
            f"Double HRP: as_of={as_of_date}, "
            f"week={target_week_start} -> {target_week_end}"
        )

        # Phase 0: Fetch full Nifty Shariah 500 universe
        universe_data = await workflow.execute_activity(
            fetch_nifty_shariah_500_universe,
            start_to_close_timeout=ACTIVITY_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
        )
        all_symbols = [s["symbol"] for s in universe_data.get("stocks", [])]
        workflow.logger.info(f"Universe: {len(all_symbols)} symbols")

        # Phase 1: HRP on full universe, 3-year lookback
        stage1 = await workflow.execute_activity(
            allocate_hrp,
            args=[all_symbols, as_of_date, STAGE1_LOOKBACK],
            start_to_close_timeout=HRP_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
        )
        workflow.logger.info(
            f"Stage 1: {stage1.symbols_used} symbols allocated, "
            f"{len(stage1.symbols_excluded)} excluded"
        )

        # Selection: top N by weight (already sorted descending by brain_api)
        top_n_symbols = list(stage1.percentage_weights.keys())[:TOP_N]
        workflow.logger.info(f"Selected top {TOP_N}: {top_n_symbols}")

        # Phase 2: HRP on top N, 1-year lookback
        stage2 = await workflow.execute_activity(
            allocate_hrp,
            args=[top_n_symbols, as_of_date, STAGE2_LOOKBACK],
            start_to_close_timeout=HRP_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
        )
        workflow.logger.info(f"Stage 2: {stage2.symbols_used} symbols allocated")

        # Phase 3: AI summary
        summary = await workflow.execute_activity(
            generate_double_hrp_summary,
            args=[stage1, stage2, UNIVERSE_LABEL, TOP_N],
            start_to_close_timeout=ACTIVITY_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
        )

        # Phase 4: Email
        email_result = await workflow.execute_activity(
            send_double_hrp_email,
            args=[
                summary,
                stage1,
                stage2,
                UNIVERSE_LABEL,
                TOP_N,
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
            "universe_symbols": len(all_symbols),
            "stage1_symbols_used": stage1.symbols_used,
            "stage1_symbols_excluded": len(stage1.symbols_excluded),
            "top_n": TOP_N,
            "top_n_symbols": top_n_symbols,
            "stage2_symbols_used": stage2.symbols_used,
            "summary_provider": summary.provider,
            "email": {
                "is_success": email_result.is_success,
                "subject": email_result.subject,
            },
        }

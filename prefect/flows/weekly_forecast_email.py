"""Weekly forecast email workflow for LearnFinance-2025.

This flow runs every Monday at 18:00 IST and executes the full inference pipeline:
1. Fetch halal universe and Alpaca portfolios (Phase 0)
2. Get signals and forecasts (Phase 1)
3. Run allocators: PPO, SAC, HRP (Phase 2)
4. Generate orders and store experience (Phase 3)
5. Submit orders to Alpaca (Phase 4)
6. Fetch order history and update execution reports (Phase 5)
7. Generate LLM summary and send email (Phase 6)

Algorithms are skipped if they have open orders from a previous run.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

from prefect import flow
from prefect.logging import get_run_logger

from flows.models import SkippedAllocation
from flows.tasks import (
    allocate_hrp,
    generate_orders_hrp,
    generate_orders_ppo,
    generate_orders_sac,
    generate_summary,
    get_fundamentals,
    get_halal_universe,
    get_hrp_portfolio,
    get_lstm_forecast,
    get_news_sentiment,
    get_order_history_ppo,
    get_order_history_sac,
    get_patchtst_forecast,
    get_ppo_portfolio,
    get_sac_portfolio,
    infer_ppo,
    infer_sac,
    send_weekly_email,
    store_experience_ppo,
    store_experience_sac,
    submit_orders_hrp,
    submit_orders_ppo,
    submit_orders_sac,
    update_execution_ppo,
    update_execution_sac,
)


@flow(
    name="Weekly Forecast Email",
    description="Weekly portfolio analysis, order execution, and email report",
    retries=0,
    timeout_seconds=7200,  # 2 hours
    persist_result=True,  # Enable resume from failure
)
def weekly_forecast_email_flow() -> dict:
    """Execute the full weekly forecast email pipeline.

    Flow phases:
    - Phase 0: Get universe + portfolios (parallel)
    - Phase 1: Get signals + forecasts (parallel)
    - Phase 2: Run allocators (parallel, conditional on open orders)
    - Phase 3: Generate orders + store experience (parallel, conditional)
    - Phase 4: Submit orders (parallel, conditional)
    - Phase 5: Get order history + update execution (parallel, conditional)
    - Phase 6: Generate summary + send email (sequential)

    Returns:
        dict with summary of execution results
    """
    logger = get_run_logger()
    logger.info("Starting weekly forecast email pipeline...")

    # Calculate dates in IST
    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    as_of_date = now.strftime("%Y-%m-%d")
    run_id = f"paper:{as_of_date}"
    attempt = 1

    # Phase 0: Get universe + portfolios (parallel)
    universe_future = get_halal_universe.submit()
    ppo_portfolio_future = get_ppo_portfolio.submit()
    sac_portfolio_future = get_sac_portfolio.submit()
    hrp_portfolio_future = get_hrp_portfolio.submit()

    universe = universe_future.result()
    ppo_portfolio = ppo_portfolio_future.result()
    sac_portfolio = sac_portfolio_future.result()
    hrp_portfolio = hrp_portfolio_future.result()

    # Get top 20 symbols
    symbols = universe.symbols[:20]
    logger.info(f"Using {len(symbols)} symbols for analysis")

    # Determine skip flags based on open orders
    run_ppo = ppo_portfolio.open_orders_count == 0
    run_sac = sac_portfolio.open_orders_count == 0
    run_hrp = hrp_portfolio.open_orders_count == 0

    skipped_algorithms = []
    if not run_ppo:
        skipped_algorithms.append("PPO")
        logger.warning(f"Skipping PPO: {ppo_portfolio.open_orders_count} open orders")
    if not run_sac:
        skipped_algorithms.append("SAC")
        logger.warning(f"Skipping SAC: {sac_portfolio.open_orders_count} open orders")
    if not run_hrp:
        skipped_algorithms.append("HRP")
        logger.warning(f"Skipping HRP: {hrp_portfolio.open_orders_count} open orders")

    # Phase 1: Get signals + forecasts (parallel)
    fundamentals_future = get_fundamentals.submit(symbols)
    news_future = get_news_sentiment.submit(symbols, as_of_date, run_id)
    lstm_future = get_lstm_forecast.submit(symbols, as_of_date)
    patchtst_future = get_patchtst_forecast.submit(symbols, as_of_date)

    fundamentals = fundamentals_future.result()
    news = news_future.result()
    lstm = lstm_future.result()
    patchtst = patchtst_future.result()

    # Get target week dates from LSTM response
    target_week_start = lstm.target_week_start or as_of_date
    target_week_end = lstm.target_week_end or as_of_date

    # Phase 2: Run allocators (parallel, conditional)
    if run_ppo:
        ppo_alloc_future = infer_ppo.submit(ppo_portfolio, as_of_date)
    if run_sac:
        sac_alloc_future = infer_sac.submit(sac_portfolio, as_of_date)
    if run_hrp:
        hrp_alloc_future = allocate_hrp.submit(as_of_date)

    ppo_alloc = (
        ppo_alloc_future.result() if run_ppo else SkippedAllocation(algorithm="ppo")
    )
    sac_alloc = (
        sac_alloc_future.result() if run_sac else SkippedAllocation(algorithm="sac")
    )
    hrp_alloc = (
        hrp_alloc_future.result() if run_hrp else SkippedAllocation(algorithm="hrp")
    )

    # Phase 3: Generate orders + store experience (parallel, conditional)
    ppo_orders_future = generate_orders_ppo.submit(
        ppo_alloc, ppo_portfolio, run_id, attempt
    )
    sac_orders_future = generate_orders_sac.submit(
        sac_alloc, sac_portfolio, run_id, attempt
    )
    hrp_orders_future = generate_orders_hrp.submit(
        hrp_alloc, hrp_portfolio, run_id, attempt
    )

    # Store experience for RL algorithms (PPO and SAC only)
    if run_ppo:
        store_experience_ppo.submit(
            run_id,
            target_week_start,
            target_week_end,
            ppo_alloc,
            ppo_portfolio,
            news,
            fundamentals,
            lstm,
            patchtst,
        )
    if run_sac:
        store_experience_sac.submit(
            run_id,
            target_week_start,
            target_week_end,
            sac_alloc,
            sac_portfolio,
            news,
            fundamentals,
            lstm,
            patchtst,
        )

    ppo_orders = ppo_orders_future.result()
    sac_orders = sac_orders_future.result()
    hrp_orders = hrp_orders_future.result()

    # Phase 4: Submit orders (parallel, conditional)
    ppo_submit_future = submit_orders_ppo.submit(ppo_orders)
    sac_submit_future = submit_orders_sac.submit(sac_orders)
    hrp_submit_future = submit_orders_hrp.submit(hrp_orders)

    ppo_submit = ppo_submit_future.result()
    sac_submit = sac_submit_future.result()
    hrp_submit = hrp_submit_future.result()

    # Phase 5: Get order history + update execution (parallel, PPO and SAC only)
    if run_ppo:
        ppo_history_future = get_order_history_ppo.submit(target_week_start)
    if run_sac:
        sac_history_future = get_order_history_sac.submit(target_week_start)

    if run_ppo:
        ppo_history = ppo_history_future.result()
        update_execution_ppo(run_id, ppo_orders, ppo_history)
    if run_sac:
        sac_history = sac_history_future.result()
        update_execution_sac(run_id, sac_orders, sac_history)

    # Phase 6: Generate summary + send email
    summary = generate_summary(
        lstm, patchtst, news, fundamentals, hrp_alloc, sac_alloc, ppo_alloc
    )

    email_result = send_weekly_email(
        summary=summary,
        lstm=lstm,
        patchtst=patchtst,
        hrp=hrp_alloc,
        sac=sac_alloc,
        ppo=ppo_alloc,
        ppo_submit=ppo_submit,
        sac_submit=sac_submit,
        hrp_submit=hrp_submit,
        target_week_start=target_week_start,
        target_week_end=target_week_end,
        as_of_date=as_of_date,
        skipped_algorithms=skipped_algorithms,
    )

    logger.info("Weekly forecast email pipeline complete!")

    return {
        "run_id": run_id,
        "as_of_date": as_of_date,
        "symbols_count": len(symbols),
        "skipped_algorithms": skipped_algorithms,
        "ppo": {
            "orders_submitted": ppo_submit.orders_submitted
            if hasattr(ppo_submit, "orders_submitted")
            else 0,
            "skipped": not run_ppo,
        },
        "sac": {
            "orders_submitted": sac_submit.orders_submitted
            if hasattr(sac_submit, "orders_submitted")
            else 0,
            "skipped": not run_sac,
        },
        "hrp": {
            "orders_submitted": hrp_submit.orders_submitted
            if hasattr(hrp_submit, "orders_submitted")
            else 0,
            "skipped": not run_hrp,
        },
        "email": {
            "is_success": email_result.is_success,
            "subject": email_result.subject,
        },
    }


if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        # Run flow once immediately for testing
        weekly_forecast_email_flow()
    else:
        # Create deployment and serve with cron schedule
        # Every Monday at 18:00 IST: "0 18 * * 1"
        weekly_forecast_email_flow.serve(
            name="weekly-forecast-email",
            cron="0 18 * * 1",
            timezone="Asia/Kolkata",
        )

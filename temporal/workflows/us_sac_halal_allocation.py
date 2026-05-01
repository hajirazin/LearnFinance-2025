"""US SAC weekly allocation workflow on the legacy ``halal`` universe.

Sibling of :mod:`workflows.us_weekly_allocation` and parallel A/B
counterpart for inference. Both workflows fire on Monday but for
different SAC buckets:

* ``USWeeklyAllocationWorkflow``      -- universe ``halal_filtered``
  (sticky-15 from PatchTST), Mon 11:00 UTC, account ``sac``,
  algorithm tag ``sac``, run_id ``paper:YYYY-MM-DD``.
* ``USSACHalalAllocationWorkflow``    -- universe ``halal``
  (legacy yfinance ETF top-holdings, variable size 10-15),
  Mon 12:30 UTC, account ``sac_halal``, algorithm tag ``sac_halal``,
  run_id ``paper:halal:YYYY-MM-DD``.

Run-id namespace: ``paper:halal:YYYY-MM-DD`` is the documented variant
form per AGENTS.md "Run identity & rerun semantics". Allowed because
this workflow trades on a dedicated Alpaca paper account
(``sac_halal``), so ``client_order_id`` collisions across the two A/B
runs are impossible (Alpaca dedupes per account). The run_id prefix
also keeps experience-record file paths disjoint -- the new run writes
``data/experience/paper_halal_YYYY-MM-DD_sac.json`` while the existing
run writes ``data/experience/paper_YYYY-MM-DD_sac.json``.

Why ``model_type="sac"`` is reused (and not ``sac_halal``): the
``/experience/label`` loop hardcodes ``model_type=="sac"`` and
``AlpacaAccount(model_type)`` -- both invariants stay valid because we
disambiguate via run_id, not model_type. Future cleanup to plumb a
distinct model_type through the labeler is out of scope here.

Variable n_stocks (10/14/15/future):
The ``halal`` SAC bucket is sized by yfinance ETF top-holdings at
**training time**; ``/inference/sac`` uses the artifact's frozen
``symbol_order``. If the live ``halal`` universe scrape changes
between monthly trainings, SAC keeps its trained ``symbol_order`` and
may target stocks that have left the live universe (or ignore stocks
that just joined). Drift filtering / on-demand retraining is a
follow-up ticket; documented here so reviewers do not mistake it for
a workflow bug.

Phases (mirror ``USWeeklyAllocationWorkflow``):
0. Get active symbols (``halal``) + SAC halal portfolio (parallel)
1. Get signals + forecasts on the chosen halal slate (parallel)
2. Run SAC allocator with ``universe='halal'`` (skipped if open
   orders on the ``sac_halal`` account)
3. Generate SAC orders tagged ``algorithm='sac_halal'`` + store
   experience under run_id ``paper:halal:YYYY-MM-DD``
4. SAC sell-wait-buy via the ``sac_halal`` Alpaca submitter
5. Get ``sac_halal`` order history + update execution
6. Generate SAC LLM summary (``universe='halal'``) + send weekly
   email (subject ``US SAC (halal) Weekly Portfolio Analysis ...``)
"""

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

from workflows._order_execution import (
    SHORT_TIMEOUT,
    sell_wait_buy,
    split_orders_by_side,
)

with workflow.unsafe.imports_passed_through():
    from activities.execution import (
        generate_orders_sac,
        store_experience_sac,
        update_execution_sac,
    )
    from activities.inference import (
        get_fundamentals,
        get_lstm_forecast,
        get_news_sentiment,
        get_patchtst_forecast,
        infer_sac,
    )
    from activities.portfolio import (
        get_active_symbols,
        get_order_history_sac_halal,
        get_sac_halal_portfolio,
        resolve_next_attempt,
        submit_orders_sac_halal,
    )
    from activities.reporting import generate_summary, send_weekly_email
    from models import SkippedAllocation

INFERENCE_TIMEOUT = timedelta(minutes=20)

UNIVERSE = "halal"
ALGORITHM = "sac_halal"
ACCOUNT = "sac_halal"


@workflow.defn
class USSACHalalAllocationWorkflow:
    """SAC weekly allocation on the legacy ``halal`` universe.

    Parallel A/B sibling of ``USWeeklyAllocationWorkflow``. Trades
    through the dedicated ``sac_halal`` Alpaca paper account; LLM
    summary and weekly email are tagged ``universe='halal'``.
    """

    @workflow.run
    async def run(self) -> dict:
        now_ist = workflow.now().astimezone()
        as_of_date = now_ist.strftime("%Y-%m-%d")
        # Variant run_id form per AGENTS.md "Run identity & rerun
        # semantics" -- safe because we use a dedicated Alpaca account
        # so client_order_id collisions across A/B paths are impossible.
        run_id = f"paper:{UNIVERSE}:{as_of_date}"

        attempt = await workflow.execute_activity(
            resolve_next_attempt,
            args=[run_id, as_of_date, [ACCOUNT]],
            start_to_close_timeout=SHORT_TIMEOUT,
        )

        workflow.logger.info(
            f"Starting US SAC halal allocation pipeline "
            f"(universe={UNIVERSE}, attempt={attempt})..."
        )

        # Phase 0: Get active symbols (halal bucket) + SAC halal portfolio.
        active_symbols, sac_portfolio = await asyncio.gather(
            workflow.execute_activity(
                get_active_symbols,
                args=[UNIVERSE],
                start_to_close_timeout=SHORT_TIMEOUT,
            ),
            workflow.execute_activity(
                get_sac_halal_portfolio, start_to_close_timeout=SHORT_TIMEOUT
            ),
        )

        symbols = active_symbols.symbols
        run_sac = sac_portfolio.open_orders_count == 0

        skipped_algorithms = []
        if not run_sac:
            skipped_algorithms.append("SAC")

        # Phase 1: Get signals + forecasts (parallel) on the halal slate.
        # LSTM and PatchTST are halal_new-trained but are called per-symbol;
        # any halal symbol not in the forecaster metadata gets a zero-filled
        # state-vector slot at inference time (drift caveat documented in
        # the module docstring).
        fundamentals, news, lstm, patchtst = await asyncio.gather(
            workflow.execute_activity(
                get_fundamentals,
                args=[symbols],
                start_to_close_timeout=INFERENCE_TIMEOUT,
                retry_policy=RetryPolicy(maximum_attempts=3),
            ),
            workflow.execute_activity(
                get_news_sentiment,
                args=[symbols, as_of_date, run_id],
                start_to_close_timeout=INFERENCE_TIMEOUT,
                retry_policy=RetryPolicy(maximum_attempts=3),
            ),
            workflow.execute_activity(
                get_lstm_forecast,
                args=[as_of_date, symbols],
                start_to_close_timeout=INFERENCE_TIMEOUT,
                retry_policy=RetryPolicy(maximum_attempts=3),
            ),
            workflow.execute_activity(
                get_patchtst_forecast,
                args=[as_of_date, symbols],
                start_to_close_timeout=INFERENCE_TIMEOUT,
                retry_policy=RetryPolicy(maximum_attempts=3),
            ),
        )

        target_week_start = lstm.target_week_start or as_of_date
        target_week_end = lstm.target_week_end or as_of_date

        # Phase 2: SAC allocator with universe='halal'.
        if run_sac:
            sac_alloc = await workflow.execute_activity(
                infer_sac,
                args=[sac_portfolio, as_of_date, UNIVERSE],
                start_to_close_timeout=INFERENCE_TIMEOUT,
            )
        else:
            sac_alloc = SkippedAllocation(algorithm=ALGORITHM)

        # Phase 3: Generate SAC orders tagged algorithm='sac_halal'.
        sac_orders = await workflow.execute_activity(
            generate_orders_sac,
            args=[sac_alloc, sac_portfolio, run_id, attempt, ALGORITHM],
            start_to_close_timeout=SHORT_TIMEOUT,
        )

        # Store experience. model_type stays "sac" so the labeler keeps
        # working; disambiguation is via the run_id prefix
        # (paper:halal:...).
        if run_sac:
            await workflow.execute_activity(
                store_experience_sac,
                args=[
                    run_id,
                    target_week_start,
                    target_week_end,
                    sac_alloc,
                    sac_portfolio,
                    news,
                    fundamentals,
                    lstm,
                    patchtst,
                ],
                start_to_close_timeout=SHORT_TIMEOUT,
            )

        # Phase 4: SAC sell-wait-buy on the sac_halal Alpaca account.
        sac_sells, sac_buys = split_orders_by_side(sac_orders)
        sac_submit = await sell_wait_buy(
            ACCOUNT, sac_sells, sac_buys, sac_orders, submit_orders_sac_halal
        )

        # Phase 5: Get sac_halal order history + update execution.
        if run_sac:
            sac_history = await workflow.execute_activity(
                get_order_history_sac_halal,
                args=[target_week_start],
                start_to_close_timeout=SHORT_TIMEOUT,
            )
            await workflow.execute_activity(
                update_execution_sac,
                args=[run_id, sac_orders, sac_history],
                start_to_close_timeout=SHORT_TIMEOUT,
            )

        # Phase 6: Generate summary + send email tagged universe='halal'.
        summary = await workflow.execute_activity(
            generate_summary,
            args=[lstm, patchtst, news, fundamentals, sac_alloc, UNIVERSE],
            start_to_close_timeout=SHORT_TIMEOUT,
        )

        email_result = await workflow.execute_activity(
            send_weekly_email,
            args=[
                summary,
                lstm,
                patchtst,
                sac_alloc,
                sac_submit,
                target_week_start,
                target_week_end,
                as_of_date,
                skipped_algorithms,
                UNIVERSE,
            ],
            start_to_close_timeout=SHORT_TIMEOUT,
        )

        workflow.logger.info("US SAC halal allocation pipeline complete!")

        return {
            "run_id": run_id,
            "as_of_date": as_of_date,
            "universe": UNIVERSE,
            "symbols_count": len(symbols),
            "skipped_algorithms": skipped_algorithms,
            "sac": {
                "orders_submitted": getattr(sac_submit, "orders_submitted", 0),
                "skipped": not run_sac,
            },
            "email": {
                "is_success": email_result.is_success,
                "subject": email_result.subject,
            },
        }

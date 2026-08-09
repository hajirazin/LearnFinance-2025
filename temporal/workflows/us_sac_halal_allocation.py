"""US SAC weekly allocation workflow on the legacy ``halal`` universe.

Sibling of :mod:`workflows.us_weekly_allocation` and parallel A/B
counterpart for inference. Both workflows fire on Monday but for
different SAC buckets AND different brokers:

* ``USWeeklyAllocationWorkflow``      -- universe ``halal_filtered``
  (sticky-15 from PatchTST), Mon 08:00 America/New_York, **Alpaca** account
  ``sac``, algorithm tag ``sac``, run_id ``paper:YYYY-MM-DD``.
* ``USSACHalalAllocationWorkflow``    -- universe ``halal``
  (legacy yfinance ETF top-holdings, variable size 10-15),
  Mon 08:30 America/New_York, **IBKR** account ``sac_halal`` (env
  ``IBKR_SAC_HALAL_*``, IB Gateway on TCP 4002 paper / 4001 live),
  algorithm tag ``sac_halal``, run_id ``paper:halal:YYYY-MM-DD``.

Broker selection lives in this workflow: it calls the IBKR-flavored
``submit_orders_ibkr_sac_halal`` / ``get_ibkr_sac_halal_portfolio`` /
``get_order_history_ibkr_sac_halal`` / ``resolve_next_attempt_ibkr``
activities, which hit brain_api's ``/ibkr/*`` routes. brain_api itself
never branches on "which broker" -- the ``/alpaca/*`` and ``/ibkr/*``
route trees are disjoint.

Run-id namespace: ``paper:halal:YYYY-MM-DD`` is the documented variant
form per AGENTS.md "Run identity & rerun semantics". Allowed because
this workflow trades on a dedicated IBKR paper account (``sac_halal``,
isolated from every Alpaca-backed strategy by virtue of being on a
different broker entirely), so ``client_order_id`` collisions across
A/B runs are impossible. brain_api enforces the broker-agnostic
guardrail too via the ``ibkr_submitted_orders`` ledger + open-trades
scan because IBKR (unlike Alpaca) does not auto-reject duplicate
``Order.orderRef``. The run_id prefix also keeps experience-record
file paths disjoint -- this workflow writes
``data/experience/paper_halal_YYYY-MM-DD_sac.json`` while the existing
run writes ``data/experience/paper_YYYY-MM-DD_sac.json``.

Why ``model_type="sac"`` is reused (and not ``sac_halal``): the
labeller routes by ``(model_type, universe)``. The two A/B SAC
workflows share ``model_type='sac'`` and disambiguate via the
``universe`` field on the experience record (this workflow stores
``universe='halal'``, the sibling stores ``universe='halal_filtered'``).
With this IBKR migration, ``universe='halal'`` no longer maps to an
Alpaca account at all -- the ``halal`` entry has been dropped from
``_SAC_UNIVERSE_TO_ACCOUNT`` so a halal record reaching the labeller
without ``actual_weights`` raises by design (per AGENTS.md rule #1)
rather than silently labelling against an Alpaca account that never
held the IBKR positions. Plumbing ``actual_weights`` from the post-
trade IBKR snapshot is therefore mandatory for this workflow.

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
0. Get active symbols (``halal``) + SAC halal IBKR portfolio (parallel)
1. Get signals + forecasts on the chosen halal slate (parallel)
2. Run SAC allocator with ``universe='halal'`` (skipped if open
   orders on the IBKR ``sac_halal`` account)
3. Generate SAC orders tagged ``algorithm='sac_halal'`` + store
   experience under run_id ``paper:halal:YYYY-MM-DD``
4. SAC sell-wait-buy via the IBKR ``sac_halal`` submitter (with
   ``check_order_statuses_ibkr`` for status polling)
5. Get IBKR ``sac_halal`` order history + post-trade IBKR portfolio
   (parallel) + update execution (post-trade portfolio plumbs
   ``actual_weights`` into the labeller)
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
from workflows._run_identity import ist_calendar_date

with workflow.unsafe.imports_passed_through():
    from activities.email_enrichment import (
        build_order_details,
        build_prior_allocation_from_portfolio,
    )
    from activities.execution import (
        generate_orders_sac,
        store_experience_sac,
        update_execution_sac,
    )
    from activities.inference import (
        get_closes,
        get_news_sentiment,
        get_patchtst_forecast,
        infer_sac,
    )
    from activities.portfolio import (
        check_order_statuses_ibkr,
        get_active_symbols,
        get_ibkr_sac_halal_portfolio,
        get_order_history_ibkr_sac_halal,
        resolve_next_attempt_ibkr,
        submit_orders_ibkr_sac_halal,
    )
    from activities.reporting import (
        generate_summary,
        send_weekly_email,
    )
    from models import SkippedAllocation

INFERENCE_TIMEOUT = timedelta(minutes=20)

UNIVERSE = "halal"
ALGORITHM = "sac_halal"
ACCOUNT = "sac_halal"


@workflow.defn
class USSACHalalAllocationWorkflow:
    """SAC weekly allocation on the legacy ``halal`` universe.

    Parallel A/B sibling of ``USWeeklyAllocationWorkflow``. Trades
    through the dedicated ``sac_halal`` IBKR paper account (env
    ``IBKR_SAC_HALAL_*``, IB Gateway on TCP 4002 paper / 4001 live);
    LLM summary and weekly email are tagged ``universe='halal'``.
    """

    @workflow.run
    async def run(self) -> dict:
        as_of_date = ist_calendar_date(workflow.now())
        # Variant run_id form per AGENTS.md "Run identity & rerun
        # semantics" -- safe because we use a dedicated Alpaca account
        # so client_order_id collisions across A/B paths are impossible.
        run_id = f"paper:{UNIVERSE}:{as_of_date}"

        attempt = await workflow.execute_activity(
            resolve_next_attempt_ibkr,
            args=[run_id, as_of_date, [ACCOUNT]],
            start_to_close_timeout=SHORT_TIMEOUT,
        )

        workflow.logger.info(
            f"Starting US SAC halal allocation pipeline "
            f"(universe={UNIVERSE}, attempt={attempt})..."
        )

        # Phase 0: Get active symbols (halal bucket) + SAC halal IBKR portfolio.
        active_symbols, sac_portfolio = await asyncio.gather(
            workflow.execute_activity(
                get_active_symbols,
                args=[UNIVERSE],
                start_to_close_timeout=SHORT_TIMEOUT,
            ),
            workflow.execute_activity(
                get_ibkr_sac_halal_portfolio, start_to_close_timeout=SHORT_TIMEOUT
            ),
        )

        symbols = active_symbols.symbols
        run_sac = sac_portfolio.open_orders_count == 0

        skipped_algorithms = []
        if not run_sac:
            skipped_algorithms.append("SAC")

        # Phase 1: Get signals + PatchTST forecast (parallel) on the
        # halal slate. PatchTST is called per-symbol; any missing
        # forecast fails canonical SAC context construction.
        news, patchtst, closes = await asyncio.gather(
            workflow.execute_activity(
                get_news_sentiment,
                args=[symbols, as_of_date, run_id],
                start_to_close_timeout=INFERENCE_TIMEOUT,
                retry_policy=RetryPolicy(maximum_attempts=3),
            ),
            workflow.execute_activity(
                get_patchtst_forecast,
                args=[as_of_date, symbols],
                start_to_close_timeout=INFERENCE_TIMEOUT,
                retry_policy=RetryPolicy(maximum_attempts=3),
            ),
            workflow.execute_activity(
                get_closes,
                args=[symbols, as_of_date],
                start_to_close_timeout=INFERENCE_TIMEOUT,
                retry_policy=RetryPolicy(maximum_attempts=3),
            ),
        )

        target_week_start = patchtst.target_week_start or as_of_date
        target_week_end = patchtst.target_week_end or as_of_date

        # Phase 2: SAC allocator with universe='halal'.
        if run_sac:
            sac_alloc = await workflow.execute_activity(
                infer_sac,
                args=[
                    sac_portfolio,
                    as_of_date,
                    UNIVERSE,
                    symbols,
                    news,
                    patchtst,
                    closes,
                ],
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

        # Store experience. model_type stays "sac" but universe='halal'
        # is now plumbed through so the labeller can short-circuit on
        # the post-trade actual_weights snapshot from IBKR (Phase 5).
        # `('sac', 'halal')` no longer maps to any Alpaca account post
        # IBKR migration -- the labeller will fail-loud per AGENTS.md
        # rule #1 if actual_weights is missing.
        if run_sac:
            await workflow.execute_activity(
                store_experience_sac,
                args=[
                    run_id,
                    target_week_start,
                    target_week_end,
                    sac_alloc,
                    UNIVERSE,
                ],
                start_to_close_timeout=SHORT_TIMEOUT,
            )

        # Phase 4: SAC sell-wait-buy on the sac_halal IBKR account.
        # ``check_order_statuses_ibkr`` is passed explicitly so the
        # broker-agnostic sell_wait_buy helper never branches on
        # ``account`` to pick between Alpaca and IBKR.
        sac_sells, sac_buys = split_orders_by_side(sac_orders)
        sac_submit = await sell_wait_buy(
            ACCOUNT,
            sac_sells,
            sac_buys,
            sac_orders,
            submit_orders_ibkr_sac_halal,
            check_status_activity=check_order_statuses_ibkr,
        )

        # Phase 5: Get sac_halal IBKR order history + post-trade IBKR
        # portfolio (parallel), then update execution with both.
        # Post-trade portfolio becomes the labeller's actual_weights,
        # eliminating the live-broker fallback at label time -- which
        # is mandatory here because ('sac', 'halal') no longer maps to
        # any Alpaca account, so the labeller has no fallback path.
        if run_sac:
            sac_history, sac_post_portfolio = await asyncio.gather(
                workflow.execute_activity(
                    get_order_history_ibkr_sac_halal,
                    args=[target_week_start],
                    start_to_close_timeout=SHORT_TIMEOUT,
                ),
                workflow.execute_activity(
                    get_ibkr_sac_halal_portfolio,
                    start_to_close_timeout=SHORT_TIMEOUT,
                ),
            )
            await workflow.execute_activity(
                update_execution_sac,
                args=[run_id, sac_orders, sac_history, sac_post_portfolio],
                start_to_close_timeout=SHORT_TIMEOUT,
            )

        # Phase 6: Generate summary + send email tagged universe='halal'.
        summary = await workflow.execute_activity(
            generate_summary,
            args=[patchtst, news, sac_alloc, UNIVERSE],
            start_to_close_timeout=SHORT_TIMEOUT,
        )

        # Build per-order detail rows and "going into this week" snapshot.
        # The IBKR portfolio shape is identical to Alpaca's (broker-agnostic
        # ``PortfolioResponse``), so the same helpers that serve the
        # Alpaca-backed workflows work here without a per-broker branch.
        sac_order_details = build_order_details(sac_orders, sac_submit)
        sac_prior_allocation = build_prior_allocation_from_portfolio(
            sac_portfolio,
            source_label="live IBKR account: sac_halal",
            as_of=as_of_date,
        )

        email_result = await workflow.execute_activity(
            send_weekly_email,
            args=[
                summary,
                patchtst,
                sac_alloc,
                sac_submit,
                target_week_start,
                target_week_end,
                as_of_date,
                skipped_algorithms,
                UNIVERSE,
                sac_order_details,
                sac_prior_allocation,
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

"""US Alpha-HRP workflow: PatchTST alpha screen + HRP risk-parity sizing.

Pipeline (Monday weekly, ``hrp`` Alpaca paper account):

0. Resolve attempt + fetch halal_new universe + get HRP portfolio (parallel).
   - If ``open_orders_count > 0`` -> skip path (email skipped=True), exit.
1. Phase 1: PatchTST batch inference across the full halal_new universe
   (~410 symbols) -> {symbol -> predicted_weekly_return_pct}.
1.5. Phase 1.5: Rank-band sticky selection
   (``universe='halal_new_alpha'``, K_in=15, K_hold=30) -> 15 chosen
   symbols. Persists score rows in sticky_history.db so next week's
   selection has a basis for stickiness.
2. Phase 2: HRP allocation (``lookback_days=252``) on the 15 selected
   symbols only -- math invariant: Stage 2 risk-parity must run on the
   exact selected set.
2.5. Phase 2.5: Record final stage 2 weights (``record_final_weights``)
   so next week's sticky read sees which stocks were *actually* held.
3. Phase 3: Translate stage 2 percentage weights into limit orders for
   the ``hrp`` Alpaca account (``algorithm='alpha_hrp'`` for forward-
   going audit clarity; account label is unchanged).
4. Phase 4: Reused sell-wait-buy machinery against the ``hrp`` account.
5. Phase 5: LLM summary (``/llm/us-alpha-hrp-summary``).
6. Phase 6: Email report (``/email/us-alpha-hrp-report``).

Math correctness invariants:
- Stage 2 HRP runs **only** on the selected set; stickiness affects
  *which 15* go into Stage 2, never the HRP math itself.
- Rank-band selection is scale-invariant: K_in/K_hold are integer
  ranks, not score thresholds. PatchTST forecasts can be noisy at the
  cardinal level without destabilising selection.

This workflow is **separate** from ``USDoubleHRPWorkflow``: that one
screens by HRP weight (covariance-based) on ``halal_new``; this one
screens by PatchTST predicted return (alpha-based) on the same
universe. Both run weekly to support side-by-side comparison.
"""

import asyncio
from dataclasses import dataclass
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

from workflows._order_execution import (
    SHORT_TIMEOUT,
    sell_wait_buy,
    split_orders_by_side,
)

with workflow.unsafe.imports_passed_through():
    from activities.execution import generate_orders_alpha_hrp
    from activities.inference import (
        allocate_hrp,
        record_final_weights,
        score_halal_new_with_patchtst,
        select_rank_band_top_n,
    )
    from activities.portfolio import (
        get_hrp_portfolio,
        resolve_next_attempt,
        submit_orders_hrp,
    )
    from activities.reporting import (
        generate_us_alpha_hrp_summary,
        send_us_alpha_hrp_email,
    )
    from activities.training import fetch_halal_new_universe
    from models import (
        HRPAllocationResponse,
        PatchTSTBatchScores,
        RankBandTopNResponse,
        SkippedSubmitResponse,
    )

ACTIVITY_TIMEOUT = timedelta(minutes=5)
HRP_TIMEOUT = timedelta(minutes=10)
PATCHTST_TIMEOUT = timedelta(minutes=10)
ACTIVITY_RETRY = 2


@dataclass(frozen=True)
class UsAlphaHrpStrategyParams:
    """Typed strategy knobs for the US Alpha-HRP weekly run.

    These are the policy choices that define the strategy as opposed
    to plumbing/timeouts. Centralising them here means a future tuning
    pass can subclass / replace the dataclass without scattering magic
    numbers across orchestration code.
    """

    sticky_partition: str = "halal_new_alpha"
    stage2_lookback_days: int = 252
    top_n: int = 15  # K_in (entry threshold)
    hold_threshold: int = 30  # K_hold (sticky retention threshold)


# Default parameters for the production schedule. Tests / experiments
# can construct a different ``UsAlphaHrpStrategyParams`` without
# touching the workflow code.
DEFAULT_STRATEGY_PARAMS = UsAlphaHrpStrategyParams()


@workflow.defn
class USAlphaHRPWorkflow:
    @workflow.run
    async def run(self) -> dict:
        params = DEFAULT_STRATEGY_PARAMS
        now_ist = workflow.now().astimezone()
        as_of_date = now_ist.strftime("%Y-%m-%d")
        # ISO year-week 'YYYYWW'; %G%V is correct across year boundaries
        # (week 1 of 2026 -> '202601' even when calendar is Dec 2025).
        year_week = now_ist.strftime("%G%V")
        run_id = f"paper:{as_of_date}"
        target_week_start = as_of_date
        target_week_end = (now_ist + timedelta(days=4)).strftime("%Y-%m-%d")

        workflow.logger.info(
            f"US Alpha-HRP: as_of={as_of_date} year_week={year_week} "
            f"week={target_week_start} -> {target_week_end}"
        )

        # Phase 0: attempt + universe + portfolio (parallel)
        attempt, universe_data, hrp_portfolio = await asyncio.gather(
            workflow.execute_activity(
                resolve_next_attempt,
                args=[run_id, as_of_date, ["hrp"]],
                start_to_close_timeout=ACTIVITY_TIMEOUT,
            ),
            workflow.execute_activity(
                fetch_halal_new_universe,
                start_to_close_timeout=ACTIVITY_TIMEOUT,
                retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
            ),
            workflow.execute_activity(
                get_hrp_portfolio,
                start_to_close_timeout=ACTIVITY_TIMEOUT,
            ),
        )

        halal_new_symbols = [s["symbol"] for s in universe_data.get("stocks", [])]
        workflow.logger.info(
            f"halal_new universe={len(halal_new_symbols)} symbols, "
            f"hrp_portfolio open_orders={hrp_portfolio.open_orders_count}, "
            f"attempt={attempt}"
        )

        # Skip path: open orders on the hrp account from a prior run mean
        # we cannot rebalance without breaking idempotency. Send the
        # skipped email and return early. We pass empty placeholders for
        # scores/sticky/stage2 so the email template can branch off
        # ``skipped=True`` without dealing with optionals.
        if hrp_portfolio.open_orders_count > 0:
            workflow.logger.warning(
                f"Alpha-HRP skipped: {hrp_portfolio.open_orders_count} open "
                f"orders on hrp account"
            )
            empty_scores = PatchTSTBatchScores(
                scores={},
                model_version="skipped",
                as_of_date=as_of_date,
                target_week_start=target_week_start,
                target_week_end=target_week_end,
                requested_count=len(halal_new_symbols),
                predicted_count=0,
                excluded_symbols=[],
            )
            empty_sticky = RankBandTopNResponse(
                selected=[],
                reasons={},
                kept_count=0,
                fillers_count=0,
                evicted_from_previous={},
                previous_year_week_used=None,
                universe=params.sticky_partition,
                year_week=year_week,
                top_n=params.top_n,
                hold_threshold=params.hold_threshold,
            )
            empty_stage2 = HRPAllocationResponse(
                percentage_weights={},
                symbols_used=0,
                symbols_excluded=[],
                lookback_days=0,
                as_of_date=as_of_date,
            )
            summary = await workflow.execute_activity(
                generate_us_alpha_hrp_summary,
                args=[
                    empty_scores,
                    empty_sticky,
                    empty_stage2,
                    params.sticky_partition,
                    params.top_n,
                    params.hold_threshold,
                ],
                start_to_close_timeout=ACTIVITY_TIMEOUT,
                retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
            )
            email = await workflow.execute_activity(
                send_us_alpha_hrp_email,
                args=[
                    summary,
                    empty_scores,
                    empty_sticky,
                    empty_stage2,
                    params.sticky_partition,
                    params.top_n,
                    params.hold_threshold,
                    target_week_start,
                    target_week_end,
                    as_of_date,
                    SkippedSubmitResponse(account="hrp", skipped=True),
                    True,
                ],
                start_to_close_timeout=ACTIVITY_TIMEOUT,
                retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
            )
            return {
                "as_of_date": as_of_date,
                "year_week": year_week,
                "skipped": True,
                "skip_reason": "open_orders",
                "email": {
                    "is_success": email.is_success,
                    "subject": email.subject,
                },
            }

        # Phase 1: PatchTST alpha scores across the full halal_new universe.
        scores = await workflow.execute_activity(
            score_halal_new_with_patchtst,
            args=[halal_new_symbols, as_of_date, params.top_n],
            start_to_close_timeout=PATCHTST_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
        )
        workflow.logger.info(
            f"Stage 1: PatchTST scored {scores.predicted_count}/"
            f"{scores.requested_count} (excluded={len(scores.excluded_symbols)}), "
            f"model={scores.model_version}"
        )

        # Phase 1.5: Rank-band sticky selection (persists scores server-side).
        sticky = await workflow.execute_activity(
            select_rank_band_top_n,
            args=[
                scores.scores,
                params.sticky_partition,
                year_week,
                as_of_date,
                run_id,
                params.top_n,
                params.hold_threshold,
            ],
            start_to_close_timeout=ACTIVITY_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
        )
        selected_symbols = sticky.selected
        workflow.logger.info(
            f"Sticky: kept={sticky.kept_count} fillers={sticky.fillers_count} "
            f"prev_yw={sticky.previous_year_week_used} "
            f"selected={selected_symbols}"
        )

        # Phase 2: HRP on the chosen N (math runs only on the selected set).
        stage2 = await workflow.execute_activity(
            allocate_hrp,
            args=[selected_symbols, as_of_date, params.stage2_lookback_days],
            start_to_close_timeout=HRP_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
        )
        workflow.logger.info(f"Stage 2: {stage2.symbols_used} symbols allocated")

        # Phase 2.5: Record stage 2 weights so next week's sticky read
        # can see which stocks were *actually* held.
        await workflow.execute_activity(
            record_final_weights,
            args=[params.sticky_partition, year_week, stage2.percentage_weights],
            start_to_close_timeout=ACTIVITY_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
        )

        # Phase 3: Translate Stage 2 weights into orders for the hrp account.
        orders = await workflow.execute_activity(
            generate_orders_alpha_hrp,
            args=[stage2, hrp_portfolio, run_id, attempt],
            start_to_close_timeout=SHORT_TIMEOUT,
        )

        # Phase 4: Sell -> wait -> buy. The shared helper keeps the
        # durable polling loop in one place.
        sells, buys = split_orders_by_side(orders)
        submit = await sell_wait_buy("hrp", sells, buys, orders, submit_orders_hrp)

        # Phase 5: LLM summary across alpha screen + Stage 2 HRP.
        summary = await workflow.execute_activity(
            generate_us_alpha_hrp_summary,
            args=[
                scores,
                sticky,
                stage2,
                params.sticky_partition,
                params.top_n,
                params.hold_threshold,
            ],
            start_to_close_timeout=ACTIVITY_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
        )

        # Phase 6: Email report.
        email = await workflow.execute_activity(
            send_us_alpha_hrp_email,
            args=[
                summary,
                scores,
                sticky,
                stage2,
                params.sticky_partition,
                params.top_n,
                params.hold_threshold,
                target_week_start,
                target_week_end,
                as_of_date,
                submit,
                False,
            ],
            start_to_close_timeout=ACTIVITY_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=ACTIVITY_RETRY),
        )

        if isinstance(submit, SkippedSubmitResponse):
            orders_submitted = 0
            orders_failed = 0
        else:
            orders_submitted = submit.orders_submitted
            orders_failed = submit.orders_failed

        return {
            "as_of_date": as_of_date,
            "year_week": year_week,
            "target_week_start": target_week_start,
            "target_week_end": target_week_end,
            "universe_symbols": len(halal_new_symbols),
            "stage1_predicted_count": scores.predicted_count,
            "stage1_excluded_count": len(scores.excluded_symbols),
            "model_version": scores.model_version,
            "top_n": params.top_n,
            "hold_threshold": params.hold_threshold,
            "selected_symbols": selected_symbols,
            "kept_count": sticky.kept_count,
            "fillers_count": sticky.fillers_count,
            "previous_year_week_used": sticky.previous_year_week_used,
            "stage2_symbols_used": stage2.symbols_used,
            "orders_submitted": orders_submitted,
            "orders_failed": orders_failed,
            "skipped": False,
            "email": {
                "is_success": email.is_success,
                "subject": email.subject,
            },
        }

"""Order generation and experience management activities."""

import logging

from temporalio import activity

from activities.client import get_client
from models import (
    AlpacaPortfolioResponse,
    GenerateOrdersResponse,
    HRPAllocationResponse,
    OrderHistoryItem,
    PaperAllocationResponse,
    PortfolioResponse,
    SACInferenceResponse,
    SkippedAllocation,
    SkippedOrdersResponse,
    StoreExperienceResponse,
    UpdateExecutionResponse,
)

logger = logging.getLogger(__name__)


@activity.defn
def generate_orders_sac(
    allocation: SACInferenceResponse | SkippedAllocation,
    portfolio: AlpacaPortfolioResponse,
    run_id: str,
    attempt: int,
    algorithm: str,
) -> GenerateOrdersResponse | SkippedOrdersResponse:
    """Generate orders for a SAC allocation, tagged by ``algorithm``.

    The ``algorithm`` arg is mandatory (no default) so the two parallel
    A/B SAC workflows tag their orders with distinct buckets:
    ``USWeeklyAllocationWorkflow`` passes ``"sac"`` (halal_filtered
    universe), ``USSACHalalAllocationWorkflow`` passes ``"sac_halal"``
    (halal universe). Per AGENTS.md rule #1 (no silent fallbacks); also
    keeps brain_api's order persistence unambiguous between the two
    runs even before the per-account Alpaca dedup kicks in.
    """
    if isinstance(allocation, SkippedAllocation) or getattr(
        allocation, "skipped", False
    ):
        logger.info(f"{algorithm.upper()} skipped - returning empty orders")
        return SkippedOrdersResponse(skipped=True, algorithm=algorithm)

    logger.info(f"Generating {algorithm.upper()} orders...")
    with get_client() as client:
        response = client.post(
            "/orders/generate",
            json={
                "target_weights": allocation.target_weights,
                "portfolio": {
                    "cash": portfolio.cash,
                    "currency": getattr(portfolio, "currency", "USD"),
                    "positions": [p.model_dump() for p in portfolio.positions],
                },
                "run_id": run_id,
                "attempt": attempt,
                "algorithm": algorithm,
                "execution_prices": allocation.execution_prices,
            },
        )
        response.raise_for_status()
    result = GenerateOrdersResponse(**response.json())
    logger.info(
        f"{algorithm.upper()} orders: "
        f"{result.summary.buys} buys, {result.summary.sells} sells"
    )
    return result


@activity.defn
def generate_orders_dhrp(
    allocation: HRPAllocationResponse | SkippedAllocation,
    portfolio: AlpacaPortfolioResponse,
    run_id: str,
    attempt: int,
) -> GenerateOrdersResponse | SkippedOrdersResponse:
    """Generate orders for the US Double HRP allocation.

    Same conversion math as HRP (percentage -> fractional weights), but
    tags orders with ``algorithm='dhrp'`` so brain_api persists them
    against the right algorithm bucket.
    """
    return _generate_orders_from_hrp(
        allocation=allocation,
        portfolio=portfolio,
        run_id=run_id,
        attempt=attempt,
        algorithm="dhrp",
    )


@activity.defn
def generate_orders_alpha_hrp(
    allocation: HRPAllocationResponse | SkippedAllocation,
    portfolio: AlpacaPortfolioResponse,
    run_id: str,
    attempt: int,
) -> GenerateOrdersResponse | SkippedOrdersResponse:
    """Generate orders for the US Alpha-HRP allocation.

    The strategy runs PatchTST as Stage 1 alpha screen on halal_new and
    HRP as Stage 2 sizing on the chosen 15. Stage 2 output is a
    standard ``HRPAllocationResponse``, so the percentage->fraction
    conversion is identical to other HRP-style allocators.

    Tagged with ``algorithm='alpha_hrp'`` so brain_api persists orders
    against the new strategy bucket; the underlying Alpaca paper
    account is still ``hrp`` (same submitter).
    """
    return _generate_orders_from_hrp(
        allocation=allocation,
        portfolio=portfolio,
        run_id=run_id,
        attempt=attempt,
        algorithm="alpha_hrp",
    )


def _generate_orders_from_hrp(
    *,
    allocation: HRPAllocationResponse | SkippedAllocation,
    portfolio: AlpacaPortfolioResponse,
    run_id: str,
    attempt: int,
    algorithm: str,
) -> GenerateOrdersResponse | SkippedOrdersResponse:
    """Shared body for HRP-style allocators (DHRP, Alpha-HRP).

    Math is identical for both: convert pp weights to fractions and POST
    to /orders/generate with the correct ``algorithm`` tag. Kept as a
    single helper because the conversion truly is the same for any HRP
    output; allocators that diverge mathematically should not call this.
    """
    if isinstance(allocation, SkippedAllocation) or getattr(
        allocation, "skipped", False
    ):
        logger.info(f"{algorithm.upper()} skipped - returning empty orders")
        return SkippedOrdersResponse(skipped=True, algorithm=algorithm)

    logger.info(f"Generating {algorithm.upper()} orders...")
    target_weights = {
        sym: wt / 100 for sym, wt in allocation.percentage_weights.items()
    }
    with get_client() as client:
        response = client.post(
            "/orders/generate",
            json={
                "target_weights": target_weights,
                "portfolio": {
                    "cash": portfolio.cash,
                    "currency": getattr(portfolio, "currency", "USD"),
                    "positions": [p.model_dump() for p in portfolio.positions],
                },
                "run_id": run_id,
                "attempt": attempt,
                "algorithm": algorithm,
            },
        )
        response.raise_for_status()
    result = GenerateOrdersResponse(**response.json())
    logger.info(
        f"{algorithm.upper()} orders: {result.summary.buys} buys, "
        f"{result.summary.sells} sells"
    )
    return result


@activity.defn
def store_experience_sac(
    run_id: str,
    week_start: str,
    week_end: str,
    allocation: SACInferenceResponse | SkippedAllocation,
    universe: str,
) -> StoreExperienceResponse | None:
    """Store SAC experience for future reward labeling.

    The ``universe`` arg is mandatory (no default) so the two parallel
    A/B SAC workflows persist their bucket onto the experience record:
    ``USWeeklyAllocationWorkflow`` passes ``"halal_filtered"`` and
    ``USSACHalalAllocationWorkflow`` passes ``"halal"``. The labeller
    reads it back to route each record to the correct Alpaca account
    via ``resolve_alpaca_account`` -- without it, every record would
    silently label against the legacy ``sac`` account.
    """
    if isinstance(allocation, SkippedAllocation) or getattr(
        allocation, "skipped", False
    ):
        logger.info("SAC skipped - not storing experience")
        return None

    logger.info(f"Storing SAC experience (universe={universe})...")
    if allocation.decision_state is None or allocation.state_digest is None:
        raise ValueError("SAC allocation is missing its canonical decision state")
    if allocation.decision_state.get("digest") != allocation.state_digest:
        raise ValueError("SAC decision-state digest does not match response digest")
    with get_client() as client:
        response = client.post(
            "/experience/store",
            json={
                "run_id": run_id,
                "week_start": week_start,
                "week_end": week_end,
                "model_type": "sac",
                "model_version": allocation.model_version,
                "universe": universe,
                "state": allocation.decision_state,
                "state_digest": allocation.state_digest,
                "intended_action": allocation.target_weights,
                "intended_turnover": allocation.turnover,
            },
        )
        response.raise_for_status()
    result = StoreExperienceResponse(**response.json())
    logger.info(f"Stored SAC experience: {result.record_id}")
    return result


def _portfolio_to_weights(portfolio: PortfolioResponse) -> dict[str, float]:
    """Convert a portfolio response to weights including ``CASH``.

    Broker-agnostic: ``PortfolioResponse`` aliases the same Pydantic
    shape returned by both ``GET /alpaca/portfolio`` and
    ``GET /ibkr/portfolio`` (cash, positions[], open_orders_count),
    so the same aggregation math feeds Alpaca-routed and IBKR-routed
    workflows with no per-broker code path. No math difference between
    brokers, so AGENTS.md rule #2 (math correctness vs DRY) doesn't
    require duplication here.

    Uses the same equity-denominator convention as
    :meth:`AlpacaClient.get_portfolio_weights` so the two paths stay
    interchangeable for the labeller. If total equity is non-positive
    (an empty paper account before first deposit) we surface an
    all-cash slate rather than divide-by-zero.
    """
    total_value = _portfolio_total_value(portfolio)
    if total_value <= 0:
        return {"CASH": 1.0}
    weights = {p.symbol: p.market_value / total_value for p in portfolio.positions}
    weights["CASH"] = portfolio.cash / total_value
    return weights


def _portfolio_total_value(portfolio: PortfolioResponse) -> float:
    """Cash + sum of position market values, in the account base currency.

    Plumbed into ``/experience/update-execution`` as ``nav_usd`` so
    the labeller's IBKR-SG cost model can size shares against the
    actual portfolio NAV at the time of the post-trade snapshot
    (instead of the cost-config default anchor). Both Alpaca and IBKR
    paper accounts denominate in USD today, so no FX conversion is
    needed.
    """
    return portfolio.cash + sum(p.market_value for p in portfolio.positions)


@activity.defn
def update_execution_sac(
    run_id: str,
    orders: GenerateOrdersResponse | SkippedOrdersResponse,
    history: list[OrderHistoryItem],
    post_trade_portfolio: PortfolioResponse | None = None,
) -> UpdateExecutionResponse | None:
    """Update SAC experience with execution report and actual weights.

    ``post_trade_portfolio`` is an optional snapshot of the SAC broker
    account taken AFTER the sell-wait-buy cycle completes. Broker-
    agnostic: the snapshot may come from Alpaca (halal_filtered SAC)
    or IBKR (halal SAC) -- the response shapes match by design, so
    ``actual_weights`` is computed the same way for both. When
    provided we send ``actual_weights`` to
    ``/experience/update-execution`` so the labeller never has to fall
    back to a live broker query.
    """
    if isinstance(orders, SkippedOrdersResponse) or getattr(orders, "skipped", False):
        logger.info("SAC skipped - not updating execution")
        return None
    if not orders.orders:
        logger.info("No SAC orders - not updating execution")
        return None

    logger.info("Updating SAC execution report...")
    intended_orders = [
        {
            "symbol": o.symbol,
            "qty": o.qty,
            "side": o.side,
            "client_order_id": o.client_order_id,
        }
        for o in orders.orders
    ]
    executed_orders = [h.model_dump() for h in history]
    body: dict = {
        "run_id": run_id,
        "model_type": "sac",
        "intended_orders": intended_orders,
        "executed_orders": executed_orders,
    }
    if post_trade_portfolio is not None:
        body["actual_weights"] = _portfolio_to_weights(post_trade_portfolio)
        body["nav_usd"] = _portfolio_total_value(post_trade_portfolio)
    with get_client() as client:
        response = client.post("/experience/update-execution", json=body)
        response.raise_for_status()
    result = UpdateExecutionResponse(**response.json())
    logger.info(
        f"Updated SAC execution: filled={result.orders_filled}, "
        f"partial={result.orders_partial}, expired={result.orders_expired}"
    )
    return result


@activity.defn
def generate_paper_allocation(
    percentage_weights: dict[str, float],
    total_nav: float,
) -> PaperAllocationResponse:
    """Convert HRP percentage weights to whole shares (paper-only, no broker).

    India workflows call this to see what a theoretical portfolio would
    look like in whole shares at current market prices. There is no
    portfolio input or Alpaca interaction — just a price lookup and
    integer-floor share math.

    Args:
        percentage_weights: ``{symbol: weight_pct}`` from Stage 2 HRP
            (e.g. ``{"RELIANCE.NS": 14.32, ...}``).
        total_nav: Notional NAV to size positions against (e.g. 1 000 000
            for 1M INR).

    Returns:
        PaperAllocationResponse with per-symbol whole-share details.
    """
    logger.info(f"Generating paper allocation (nav={total_nav})...")
    with get_client() as client:
        response = client.post(
            "/orders/paper-allocation",
            json={
                "percentage_weights": percentage_weights,
                "total_nav": total_nav,
            },
        )
        response.raise_for_status()
    result = PaperAllocationResponse(**response.json())
    logger.info(
        f"Paper allocation: {len(result.details)} symbols, "
        f"total_allocated={result.total_allocated_pct}%"
    )
    return result

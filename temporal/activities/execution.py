"""Order generation and experience management activities."""

import logging

from temporalio import activity

from activities.client import get_client
from models import (
    AlpacaPortfolioResponse,
    FundamentalsResponse,
    GenerateOrdersResponse,
    HRPAllocationResponse,
    LSTMInferenceResponse,
    NewsSignalResponse,
    OrderHistoryItem,
    PatchTSTInferenceResponse,
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
) -> GenerateOrdersResponse | SkippedOrdersResponse:
    """Generate orders for SAC allocation."""
    if isinstance(allocation, SkippedAllocation) or getattr(
        allocation, "skipped", False
    ):
        logger.info("SAC skipped - returning empty orders")
        return SkippedOrdersResponse(skipped=True, algorithm="sac")

    logger.info("Generating SAC orders...")
    with get_client() as client:
        response = client.post(
            "/orders/generate",
            json={
                "target_weights": allocation.target_weights,
                "portfolio": {
                    "cash": portfolio.cash,
                    "positions": [p.model_dump() for p in portfolio.positions],
                },
                "run_id": run_id,
                "attempt": attempt,
                "algorithm": "sac",
            },
        )
        response.raise_for_status()
    result = GenerateOrdersResponse(**response.json())
    logger.info(f"SAC orders: {result.summary.buys} buys, {result.summary.sells} sells")
    return result


@activity.defn
def generate_orders_hrp(
    allocation: HRPAllocationResponse | SkippedAllocation,
    portfolio: AlpacaPortfolioResponse,
    run_id: str,
    attempt: int,
) -> GenerateOrdersResponse | SkippedOrdersResponse:
    """Generate orders for HRP allocation.

    HRP returns percentage_weights (10.5 = 10.5%), converted to
    target_weights (0.105) before calling the orders endpoint.
    """
    return _generate_orders_from_hrp(
        allocation=allocation,
        portfolio=portfolio,
        run_id=run_id,
        attempt=attempt,
        algorithm="hrp",
    )


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
    """Shared body for HRP-style allocators (HRP, DHRP).

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


def _build_state_dict(
    portfolio: AlpacaPortfolioResponse,
    news: NewsSignalResponse,
    fundamentals: FundamentalsResponse,
    lstm: LSTMInferenceResponse,
    patchtst: PatchTSTInferenceResponse,
) -> dict:
    """Build state dict for experience storage."""
    total_value = portfolio.cash + sum(p.market_value for p in portfolio.positions)
    current_weights = {
        p.symbol: p.market_value / total_value for p in portfolio.positions
    }
    current_weights["CASH"] = portfolio.cash / total_value

    signals = {}
    for ns in news.per_symbol:
        signals[ns.symbol] = {"news_sentiment": ns.sentiment_score}
    for fs in fundamentals.per_symbol:
        if fs.ratios:
            if fs.symbol not in signals:
                signals[fs.symbol] = {}
            signals[fs.symbol].update(fs.ratios.model_dump())

    lstm_forecasts = {
        p.symbol: p.predicted_weekly_return_pct / 100.0 for p in lstm.predictions
    }
    patchtst_forecasts = {
        p.symbol: p.predicted_weekly_return_pct / 100.0 for p in patchtst.predictions
    }

    return {
        "signals": signals,
        "lstm_forecasts": lstm_forecasts,
        "patchtst_forecasts": patchtst_forecasts,
        "current_weights": current_weights,
    }


@activity.defn
def store_experience_sac(
    run_id: str,
    week_start: str,
    week_end: str,
    allocation: SACInferenceResponse | SkippedAllocation,
    portfolio: AlpacaPortfolioResponse,
    news: NewsSignalResponse,
    fundamentals: FundamentalsResponse,
    lstm: LSTMInferenceResponse,
    patchtst: PatchTSTInferenceResponse,
) -> StoreExperienceResponse | None:
    """Store SAC experience for future reward labeling."""
    if isinstance(allocation, SkippedAllocation) or getattr(
        allocation, "skipped", False
    ):
        logger.info("SAC skipped - not storing experience")
        return None

    logger.info("Storing SAC experience...")
    state = _build_state_dict(portfolio, news, fundamentals, lstm, patchtst)
    with get_client() as client:
        response = client.post(
            "/experience/store",
            json={
                "run_id": run_id,
                "week_start": week_start,
                "week_end": week_end,
                "model_type": "sac",
                "model_version": allocation.model_version,
                "state": state,
                "intended_action": allocation.target_weights,
                "intended_turnover": allocation.turnover,
            },
        )
        response.raise_for_status()
    result = StoreExperienceResponse(**response.json())
    logger.info(f"Stored SAC experience: {result.record_id}")
    return result


@activity.defn
def update_execution_sac(
    run_id: str,
    orders: GenerateOrdersResponse | SkippedOrdersResponse,
    history: list[OrderHistoryItem],
) -> UpdateExecutionResponse | None:
    """Update SAC experience with execution report."""
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
    with get_client() as client:
        response = client.post(
            "/experience/update-execution",
            json={
                "run_id": run_id,
                "model_type": "sac",
                "intended_orders": intended_orders,
                "executed_orders": executed_orders,
            },
        )
        response.raise_for_status()
    result = UpdateExecutionResponse(**response.json())
    logger.info(
        f"Updated SAC execution: filled={result.orders_filled}, "
        f"partial={result.orders_partial}, expired={result.orders_expired}"
    )
    return result

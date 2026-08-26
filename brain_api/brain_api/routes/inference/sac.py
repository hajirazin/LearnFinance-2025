"""SAC v3 inference endpoint from raw point-in-time evidence."""

import logging
import time
from datetime import date

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from brain_api.core.inference_utils import compute_week_from_cutoff
from brain_api.core.model_buckets import (
    ModelType,
    UnknownBucketError,
    get_bucket,
    list_universes_for,
)
from brain_api.core.portfolio_rl.constraints import compute_turnover_from_allocations
from brain_api.core.sac import run_sac_inference
from brain_api.core.sac.decision_context import (
    SACDecisionContext,
    SACDecisionContextError,
    SACDecisionState,
    SACFeatureBundle,
)
from brain_api.core.sac.news_window import sac_news_from_window
from brain_api.core.training_utils import get_device
from brain_api.core.weekly_decision import (
    MondayCutoffError,
    monday_window_bounds,
    require_monday_decision_cutoff,
)
from brain_api.news.models import NEWS_SCHEMA_VERSION, NEWS_SENTIMENT_REVISION
from brain_api.storage.policy import load_current_artifacts_for_bucket

from .dependencies import get_sac_as_of_date
from .models import (
    ForcedLiquidationAudit,
    SACInferenceRequest,
    SACInferenceResponse,
    WeightChange,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/sac", response_model=SACInferenceResponse)
def infer_sac(
    request: SACInferenceRequest,
    universe: str = Query(
        ...,
        description=(
            "Required SAC bucket universe (`halal_filtered` or `halal`); "
            "there is no default because the buckets are independent."
        ),
    ),
) -> SACInferenceResponse:
    """Get target portfolio weights from SAC policy.

    This endpoint:
    1. Loads the current SAC model via the active storage policy
    2. Normalizes the portfolio snapshot to weights
    3. Applies eligibility, ranking, and causal HMM filtering in Brain
    4. Runs the fixed masked-attention policy
    5. Returns target weights plus reproducible decision audit data
    """
    t_start = time.time()
    logger.info("[SAC] Starting inference")

    try:
        bucket = get_bucket(ModelType.SAC, universe)
    except UnknownBucketError as exc:
        allowed = sorted(list_universes_for(ModelType.SAC))
        raise HTTPException(
            status_code=422,
            detail=(f"Unknown universe '{universe}' for SAC. Allowed: {allowed}"),
        ) from exc

    cutoff_date = get_sac_as_of_date(request)
    decision_date = (
        date.fromisoformat(request.as_of_date) if request.as_of_date else date.today()
    )
    logger.info(f"[SAC] Cutoff date: {cutoff_date}")

    week_boundaries = compute_week_from_cutoff(cutoff_date)
    logger.info(
        f"[SAC] Target week: {week_boundaries.target_week_start} to {week_boundaries.target_week_end}"
    )

    logger.info("[SAC] Loading model artifacts...")
    artifacts = load_current_artifacts_for_bucket(
        bucket=bucket,
        model_label=bucket.model_label,
    )

    artifacts.actor.to(get_device())
    logger.info(
        f"[SAC] Model loaded: version={artifacts.version}, "
        f"device={next(artifacts.actor.parameters()).device}"
    )
    metadata = artifacts.metadata
    if (
        metadata.get("news_schema_version") != NEWS_SCHEMA_VERSION
        or metadata.get("finbert_revision") != NEWS_SENTIMENT_REVISION
    ):
        raise HTTPException(
            status_code=503,
            detail=(
                "current SAC artifact was not trained on news schema v1 / pinned FinBERT"
            ),
        )

    try:
        cutoff = require_monday_decision_cutoff(request.as_of)
    except MondayCutoffError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    expected_start, expected_end = monday_window_bounds(cutoff.date())
    news_window = request.news_window
    if (
        news_window.start_exclusive != expected_start
        or news_window.end_inclusive != expected_end
    ):
        raise HTTPException(
            status_code=422,
            detail="news_window bounds do not match the Monday 09:00 decision window",
        )
    try:
        news_sentiment, news_article_counts, news_audit = sac_news_from_window(
            news_window,
            symbols=request.feature_bundle.symbols,
            cutoff=cutoff,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    cash_value = request.portfolio.cash
    position_values = {
        pos.symbol: pos.market_value for pos in request.portfolio.positions
    }
    forced_liquidations = [
        ForcedLiquidationAudit(
            symbol=symbol,
            market_value=market_value,
        )
        for symbol, market_value in sorted(position_values.items())
        if symbol not in artifacts.symbol_order and market_value > 0
    ]

    forced_liquidation_value = sum(audit.market_value for audit in forced_liquidations)
    total_portfolio_value = cash_value + sum(position_values.values())
    if total_portfolio_value <= 0:
        raise HTTPException(
            status_code=422,
            detail="portfolio cash + positions must be positive for SAC inference",
        )

    # True pre-decision sleeve weights (off-slate included) for turnover.
    true_current_allocation = {
        symbol: market_value / total_portfolio_value
        for symbol, market_value in position_values.items()
        if market_value > 0
    }
    true_current_allocation["CASH"] = cash_value / total_portfolio_value

    # Observation / actor view folds off-slate MV into CASH so the simplex
    # matches the active model slate (same pattern as in-universe ineligible
    # folding inside build_state_vector).
    effective_cash_value = cash_value + forced_liquidation_value
    allocatable_value = effective_cash_value + sum(
        position_values.get(symbol, 0.0) for symbol in artifacts.symbol_order
    )
    n_stocks = len(artifacts.symbol_order)
    current_weights = np.zeros(n_stocks + 1)
    for i, symbol in enumerate(artifacts.symbol_order):
        if allocatable_value > 0:
            current_weights[i] = position_values.get(symbol, 0.0) / allocatable_value
    current_weights[-1] = (
        effective_cash_value / allocatable_value if allocatable_value > 0 else 1.0
    )

    try:
        feature_bundle = SACFeatureBundle.create(
            symbols=request.feature_bundle.symbols,
            adjusted_closes=request.feature_bundle.adjusted_closes,
            news_sentiment=news_sentiment,
            news_article_counts=news_article_counts,
            patchtst_forecasts=request.feature_bundle.patchtst_forecasts,
            market_history=[
                row.model_dump(mode="json")
                for row in request.feature_bundle.market_history
            ],
            provenance=request.feature_bundle.provenance,
        )
        if feature_bundle.symbols != tuple(artifacts.symbol_order):
            raise SACDecisionContextError(
                "feature_bundle symbols must exactly match active model symbol order"
            )
        decision_context = SACDecisionContext.create(
            as_of_date=decision_date,
            feature_bundle=feature_bundle,
            current_weights={
                **{
                    symbol: float(current_weights[index])
                    for index, symbol in enumerate(artifacts.symbol_order)
                },
                "CASH": float(current_weights[-1]),
            },
        )
    except SACDecisionContextError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    logger.info("[SAC] Running inference...")
    try:
        result = run_sac_inference(
            actor=artifacts.actor,
            scaler=artifacts.scaler,
            config=artifacts.config,
            decision_context=decision_context,
            regime_hmm=artifacts.v3_auxiliary.regime_hmm,
            model_version=artifacts.version,
        )
    except (SACDecisionContextError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    true_target_allocation = {
        **result.allocation,
        **{audit.symbol: 0.0 for audit in forced_liquidations},
    }
    turnover = compute_turnover_from_allocations(
        true_current_allocation, true_target_allocation
    )

    decision_state = SACDecisionState.create(
        vector=result.state_vector,
        context=decision_context,
    )

    weight_changes = []
    for symbol in artifacts.symbol_order:
        current_w = current_weights[artifacts.symbol_order.index(symbol)]
        target_w = result.allocation.get(symbol, 0.0)
        weight_changes.append(
            WeightChange(
                symbol=symbol,
                current_weight=current_w,
                target_weight=target_w,
                change=target_w - current_w,
            )
        )
    weight_changes.append(
        WeightChange(
            symbol="CASH",
            current_weight=current_weights[-1],
            target_weight=result.allocation.get("CASH", 0.0),
            change=result.allocation.get("CASH", 0.0) - current_weights[-1],
        )
    )

    t_total = time.time() - t_start
    logger.info(f"[SAC] Inference complete in {t_total:.2f}s, turnover={turnover:.4f}")

    return SACInferenceResponse(
        target_weights=result.allocation,
        turnover=turnover,
        target_week_start=week_boundaries.target_week_start.isoformat(),
        target_week_end=week_boundaries.target_week_end.isoformat(),
        model_version=result.model_version,
        weight_changes=weight_changes,
        decision_state=decision_state.to_dict(),
        state_digest=decision_state.digest,
        forced_liquidations=forced_liquidations,
        asset_eligibility={
            symbol: bool(result.asset_mask[index])
            for index, symbol in enumerate(artifacts.symbol_order)
        },
        regime_posterior=[float(value) for value in result.regime_posterior],
        sac_schema_version=artifacts.metadata["sac_schema_version"],
        architecture=artifacts.metadata["architecture"],
        news_audit=news_audit,
    )

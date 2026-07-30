"""SAC inference endpoint with dual forecasts (LSTM + PatchTST)."""

import logging
import time

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from brain_api.core.inference_utils import compute_week_from_cutoff
from brain_api.core.model_buckets import (
    ModelType,
    UnknownBucketError,
    get_bucket,
    list_universes_for,
)
from brain_api.core.sac import run_sac_inference
from brain_api.core.sac.decision_context import (
    SACDecisionContext,
    SACDecisionContextError,
    SACDecisionState,
    SACFeatureBundle,
)
from brain_api.core.training_utils import get_device
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
    universe: str | None = Query(
        default=None,
        description=(
            "Optional bucket override. Defaults to the only registered SAC "
            "universe (`halal_filtered`). Future buckets (e.g. `halal`) can "
            "be selected without breaking existing callers."
        ),
    ),
) -> SACInferenceResponse:
    """Get target portfolio weights from SAC policy.

    This endpoint:
    1. Loads the current SAC model via the active storage policy
       (``local_first`` / ``hf_first``); HF download under
       ``universe='halal'`` caches into ``data/models/sac_halal/`` so
       audit Bug 4 (bucket isolation) cannot regress on the read path.
    2. Normalizes the portfolio snapshot to weights
    3. Builds state vector with current signals + dual forecasts (LSTM + PatchTST)
    4. Runs SAC inference to get target weights
    5. Returns target weights and turnover

    Args:
        request: Portfolio snapshot (cash + positions)

    Returns:
        Target weights and execution metadata
    """
    t_start = time.time()
    logger.info("[SAC] Starting inference")

    resolved_universe = universe if universe is not None else "halal_filtered"
    try:
        bucket = get_bucket(ModelType.SAC, resolved_universe)
    except UnknownBucketError as exc:
        allowed = sorted(list_universes_for(ModelType.SAC))
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown universe '{resolved_universe}' for SAC. Allowed: {allowed}"
            ),
        ) from exc

    # Get cutoff date (always a Friday)
    cutoff_date = get_sac_as_of_date(request)
    logger.info(f"[SAC] Cutoff date: {cutoff_date}")

    # Compute target week boundaries for the week AFTER cutoff
    week_boundaries = compute_week_from_cutoff(cutoff_date)
    logger.info(
        f"[SAC] Target week: {week_boundaries.target_week_start} to {week_boundaries.target_week_end}"
    )

    # Load model artifacts via the active storage policy. The helper
    # raises HTTPException 503 with an actionable message on miss /
    # cold-start / HF unreachable.
    logger.info("[SAC] Loading model artifacts...")
    artifacts = load_current_artifacts_for_bucket(
        bucket=bucket,
        model_label=bucket.model_label,
    )

    # Storage layer always loads weights on CPU for portability (Mac/Pi/CUDA).
    # Move actor to the best available device for fast inference (MPS on Mac,
    # CPU on the Pi). The actor's get_action() reads device from its parameters,
    # so this is the only placement decision needed for SAC inference.
    artifacts.actor.to(get_device())
    logger.info(
        f"[SAC] Model loaded: version={artifacts.version}, "
        f"device={next(artifacts.actor.parameters()).device}"
    )

    # Convert portfolio snapshot to values dict
    cash_value = request.portfolio.cash
    position_values = {
        pos.symbol: pos.market_value for pos in request.portfolio.positions
    }

    forced_liquidations = [
        ForcedLiquidationAudit(symbol=symbol, market_value=market_value)
        for symbol, market_value in sorted(position_values.items())
        if symbol not in artifacts.symbol_order and market_value > 0
    ]

    # The actor has no slot for positions outside its active symbol order.
    # Because order generation force-liquidates them before rebalancing, model
    # their proceeds as cash rather than dropping them from NAV and
    # renormalizing the remaining active positions.
    forced_liquidation_value = sum(audit.market_value for audit in forced_liquidations)
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

    decision_context: SACDecisionContext | None = None
    if request.feature_bundle is not None:
        try:
            feature_bundle = SACFeatureBundle.create(
                symbols=request.feature_bundle.symbols,
                signals=request.feature_bundle.signals,
                lstm_forecasts=request.feature_bundle.lstm_forecasts,
                patchtst_forecasts=request.feature_bundle.patchtst_forecasts,
                provenance=request.feature_bundle.provenance,
            )
            if feature_bundle.symbols != tuple(artifacts.symbol_order):
                raise SACDecisionContextError(
                    "feature_bundle symbols must exactly match active model symbol order"
                )
            decision_context = SACDecisionContext.create(
                as_of_date=cutoff_date,
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
        signals = feature_bundle.signals
        lstm_forecasts = feature_bundle.lstm_forecasts
        patchtst_forecasts = feature_bundle.patchtst_forecasts
    elif artifacts.state_schema_version >= 2:
        raise HTTPException(
            status_code=422,
            detail=(
                "feature_bundle is required for SAC state-schema v2; Brain will "
                "not refetch or zero-fill actor inputs"
            ),
        )
    else:
        # Metadata-absent artifacts are legacy state-schema v1. Keep their
        # historical refetch path solely for migration compatibility.
        from .helpers import build_current_forecasts, build_current_signals

        logger.warning("[SAC] Using legacy v1 refetch compatibility path")
        signals = build_current_signals(artifacts.symbol_order, cutoff_date)
        lstm_forecasts = build_current_forecasts(
            artifacts.symbol_order, forecaster_type="lstm", as_of_date=cutoff_date
        )
        patchtst_forecasts = build_current_forecasts(
            artifacts.symbol_order, forecaster_type="patchtst", as_of_date=cutoff_date
        )

    # Run inference
    logger.info("[SAC] Running inference...")
    result = run_sac_inference(
        actor=artifacts.actor,
        scaler=artifacts.scaler,
        config=artifacts.config,
        symbol_order=artifacts.symbol_order,
        current_weights=current_weights,
        signals=signals,
        lstm_forecasts=lstm_forecasts,
        patchtst_forecasts=patchtst_forecasts,
        model_version=artifacts.version,
        state_schema_version=artifacts.state_schema_version,
    )
    decision_state = (
        SACDecisionState.create(
            schema_version=result.state_schema_version,
            vector=result.state_vector,
            context=decision_context,
        )
        if decision_context is not None
        else None
    )

    # Build weight changes list
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
    # Add CASH
    weight_changes.append(
        WeightChange(
            symbol="CASH",
            current_weight=current_weights[-1],
            target_weight=result.allocation.get("CASH", 0.0),
            change=result.allocation.get("CASH", 0.0) - current_weights[-1],
        )
    )

    t_total = time.time() - t_start
    logger.info(
        f"[SAC] Inference complete in {t_total:.2f}s, turnover={result.turnover:.4f}"
    )

    return SACInferenceResponse(
        target_weights=result.allocation,
        turnover=result.turnover,
        target_week_start=week_boundaries.target_week_start.isoformat(),
        target_week_end=week_boundaries.target_week_end.isoformat(),
        model_version=result.model_version,
        weight_changes=weight_changes,
        decision_state=decision_state.to_dict() if decision_state else None,
        state_digest=decision_state.digest if decision_state else None,
        forced_liquidations=forced_liquidations,
    )

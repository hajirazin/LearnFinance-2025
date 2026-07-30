"""Experience buffer and labeling endpoints.

This module provides:
- Storage for RL (SAC) experience tuples (state, action, turnover)
- Labeling endpoint to fill in realized rewards after the week ends
- Reading experience for fine-tuning
"""

import logging
from datetime import UTC, date, datetime

from fastapi import APIRouter, Depends, HTTPException

from brain_api.core.sac.experience_accounting import compute_realized_sac_reward
from brain_api.core.sac.trade_clock import experience_open_transition
from brain_api.routes.experience_models import (
    ExperienceRecord,
    ExperienceState,
    LabelExperienceRequest,
    LabelExperienceResponse,
    StoreExperienceRequest,
    StoreExperienceResponse,
    UpdateExecutionRequest,
    UpdateExecutionResponse,
)
from brain_api.storage.base import DEFAULT_DATA_PATH
from brain_api.storage.experience import ExperienceStorage

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Storage helpers
# ============================================================================


def get_experience_storage() -> ExperienceStorage:
    """Get experience storage instance."""
    return ExperienceStorage(base_path=DEFAULT_DATA_PATH)


# ============================================================================
# Order Matching
# ============================================================================


def match_orders(
    intended_orders: list[dict],
    executed_orders: list[dict],
) -> list[dict]:
    """Match intended orders with executed orders by client_order_id.

    Args:
        intended_orders: Orders we intended to submit (from /orders/generate).
            Each must have: symbol, qty, side, client_order_id
        executed_orders: Raw order history from Alpaca (from /alpaca/order-history).
            Each must have: client_order_id, status, filled_qty, filled_avg_price

    Returns:
        List of execution report dicts with:
            symbol, side, intended_qty, filled_qty, filled_avg_price, status, client_order_id
    """
    # Build lookup map for executed orders
    executed_map = {o.get("client_order_id", ""): o for o in executed_orders}

    execution_report = []
    for intended in intended_orders:
        client_order_id = intended.get("client_order_id", "")
        executed = executed_map.get(client_order_id, {})

        # Parse filled_qty (Alpaca returns as string)
        filled_qty_str = executed.get("filled_qty")
        filled_qty = float(filled_qty_str) if filled_qty_str else 0.0

        # Parse filled_avg_price (Alpaca returns as string)
        filled_price_str = executed.get("filled_avg_price")
        filled_avg_price = float(filled_price_str) if filled_price_str else None

        # Determine status
        status = executed.get("status", "not_found")

        execution_report.append(
            {
                "symbol": intended.get("symbol", ""),
                "side": intended.get("side", ""),
                "intended_qty": intended.get("qty", 0),
                "filled_qty": filled_qty,
                "filled_avg_price": filled_avg_price,
                "status": status,
                "client_order_id": client_order_id,
            }
        )

    return execution_report


# ============================================================================
# Reward computation
# ============================================================================


def _extract_prior_weights(record: ExperienceRecord) -> dict[str, float]:
    """Pull pre-rebalance weights off the experience record's state.

    ``state`` may be either an :class:`ExperienceState` or a raw dict
    (legacy serialised records). When ``current_weights`` is missing
    or empty we default to all-cash, which is the right zero-cost
    starting condition for the simulator and matches the env reset
    convention.
    """
    state = record.state
    if isinstance(state, ExperienceState):
        cw = state.current_weights or {}
    elif isinstance(state, dict):
        cw = state.get("current_weights") or {}
    else:
        cw = {}
    if not cw:
        return {"CASH": 1.0}
    return {k: float(v) for k, v in cw.items()}


def compute_realized_reward(
    action: dict[str, float],
    symbol_returns: dict[str, float],
    *,
    prior_weights: dict[str, float] | None = None,
    symbol_prices: dict[str, float] | None = None,
    nav_usd: float | None = None,
    reward_scale: float = 100.0,
) -> tuple[float, float]:
    """Compute realized reward from actual returns under the IBKR-SG cost model.

    The transaction cost is the sum of per-symbol per-leg IBKR
    Singapore Tiered fees (commission with $0.35 min / 1% max,
    sell-side regulatory, clearing, pass-through). See
    :mod:`brain_api.core.portfolio_rl.broker_costs` for the math.

    Args:
        action: Target weights at decision time (post-rebalance).
        symbol_returns: Realized weekly returns for each symbol.
        prior_weights: Pre-rebalance weights, used to compute the
            per-symbol delta the cost model charges for. Defaults to
            an all-cash slate (full opening of every position).
        symbol_prices: Per-symbol close prices used to convert
            dollar deltas into share counts. Required for any
            symbol with a non-zero weight delta; per AGENTS.md
            rule #1 a missing price raises rather than silently
            zero-costing the leg.
        nav_usd: Total portfolio equity in USD (used for the $0.35
            per-order minimum and 1% per-order ceiling). Falls back
            to the IBKR cost config's default anchor (USD 10k) when
            None -- WARNING-logged by callers.
        reward_scale: Reward scaling factor.

    Returns:
        Tuple of (reward, realized_return).
    """
    return compute_realized_sac_reward(
        action,
        symbol_returns,
        prior_weights=prior_weights,
        symbol_prices=symbol_prices,
        nav_usd=nav_usd,
        reward_scale=reward_scale,
    )


# ============================================================================
# Endpoints
# ============================================================================


@router.post("/store", response_model=StoreExperienceResponse)
def store_experience(
    request: StoreExperienceRequest,
    storage: ExperienceStorage = Depends(get_experience_storage),
) -> StoreExperienceResponse:
    """Store an experience record with full state.

    This is called after each RL (SAC) inference to record:
    - Full state (signals, forecasts, current_weights)
    - Intended action (target weights from policy)
    - Intended turnover

    Later, execution report and reward are filled by separate endpoints.
    """
    # Determine if caller is using new API (intended_action) or legacy (action)
    # If intended_action has values, use new fields; otherwise fall back to legacy
    using_new_api = len(request.intended_action) > 0

    if using_new_api:
        # New API: use intended_action and intended_turnover
        action = request.intended_action
        turnover = request.intended_turnover
    elif len(request.action) > 0:
        # Legacy API: use action and turnover
        action = request.action
        turnover = request.turnover
    else:
        # Neither provided - use new API defaults (empty action is valid for all-cash)
        action = request.intended_action
        turnover = request.intended_turnover

    # Create unique run_id including model_type to separate model types
    run_id = request.run_id
    if not run_id.endswith(f":{request.model_type}"):
        run_id = f"{request.run_id}:{request.model_type}"

    if request.state_digest is not None:
        state_payload = (
            request.state
            if isinstance(request.state, dict)
            else request.state.model_dump()
        )
        if state_payload.get("digest") != request.state_digest:
            raise HTTPException(
                status_code=422,
                detail="state_digest does not match canonical decision state digest",
            )

    record = ExperienceRecord(
        run_id=run_id,
        week_start=request.week_start,
        week_end=request.week_end,
        model_type=request.model_type,
        model_version=request.model_version,
        universe=request.universe,
        state=request.state,
        state_digest=request.state_digest,
        intended_action=action,
        intended_turnover=turnover,
        # Legacy fields for backward compatibility
        action=action,
        turnover=turnover,
    )

    record_id = storage.store(record)
    logger.info(
        f"[Experience] Stored {request.model_type.upper()} record: {record_id} "
        f"with {len(action)} positions"
    )

    return StoreExperienceResponse(
        record_id=record_id,
        stored=True,
        model_type=request.model_type,
    )


@router.post("/update-execution", response_model=UpdateExecutionResponse)
def update_execution(
    request: UpdateExecutionRequest,
    storage: ExperienceStorage = Depends(get_experience_storage),
) -> UpdateExecutionResponse:
    """Update experience record with execution report after orders settle.

    This endpoint supports two modes:

    1. **New mode (recommended)**: Provide `intended_orders` and `executed_orders`.
       The endpoint will match them by `client_order_id` and compute the execution report.

    2. **Legacy mode**: Provide pre-computed `execution_report` directly.

    Args:
        request: Contains run_id, model_type, and either:
            - intended_orders + executed_orders (new mode), or
            - execution_report (legacy mode)

    Returns:
        Update status with counts of filled/partial/expired orders.
    """
    # Build run_id including model_type
    run_id = request.run_id
    if not run_id.endswith(f":{request.model_type}"):
        run_id = f"{request.run_id}:{request.model_type}"

    record = storage.load(run_id)
    if record is None:
        logger.warning(f"[Experience] Record not found: {run_id}")
        return UpdateExecutionResponse(
            run_id=run_id,
            updated=False,
            orders_filled=0,
            orders_partial=0,
            orders_expired=0,
        )

    # Determine execution_report: either from raw data (new) or pre-computed (legacy)
    if request.intended_orders is not None and request.executed_orders is not None:
        # New mode: match intended vs executed orders internally
        logger.info(
            f"[Experience] Matching {len(request.intended_orders)} intended orders "
            f"with {len(request.executed_orders)} executed orders"
        )

        # Convert Pydantic models to dicts if needed
        intended_dicts = [
            o.model_dump() if hasattr(o, "model_dump") else o
            for o in request.intended_orders
        ]
        executed_dicts = [
            o.model_dump() if hasattr(o, "model_dump") else o
            for o in request.executed_orders
        ]

        execution_report = match_orders(intended_dicts, executed_dicts)
    elif request.execution_report is not None:
        # Legacy mode: use pre-computed execution report
        execution_report = request.execution_report
    else:
        # Neither provided - error
        logger.error(f"[Experience] No execution data provided for {run_id}")
        return UpdateExecutionResponse(
            run_id=run_id,
            updated=False,
            orders_filled=0,
            orders_partial=0,
            orders_expired=0,
        )

    # Count order statuses
    orders_filled = 0
    orders_partial = 0
    orders_expired = 0

    for order in execution_report:
        status = order.get("status", "").lower()
        if status == "filled":
            orders_filled += 1
        elif status == "partial" or status == "partially_filled":
            orders_partial += 1
        elif status in ("expired", "canceled", "rejected", "not_found"):
            orders_expired += 1

    # Update record
    record.execution_report = execution_report
    record.actual_weights = request.actual_weights
    if request.nav_usd is not None:
        record.nav_usd = request.nav_usd
    record.execution_updated_at = datetime.now(UTC).isoformat()

    storage.update(record)

    logger.info(
        f"[Experience] Updated {request.model_type.upper()} execution: {run_id} "
        f"(filled={orders_filled}, partial={orders_partial}, expired={orders_expired})"
    )

    return UpdateExecutionResponse(
        run_id=run_id,
        updated=True,
        orders_filled=orders_filled,
        orders_partial=orders_partial,
        orders_expired=orders_expired,
    )


@router.post("/label", response_model=LabelExperienceResponse)
def label_experience(
    request: LabelExperienceRequest,
    storage: ExperienceStorage = Depends(get_experience_storage),
) -> LabelExperienceResponse:
    """Label experience records with realized rewards.

    This endpoint:
    1. Finds unlabeled experience records where week_end < today
    2. Fetches realized weekly returns for each symbol
    3. Computes reward = log(1 + gross_return - transaction_cost) * scale
    4. Updates the experience record

    Should be called weekly (e.g., Sunday) to label the previous week's
    experience before fine-tuning.
    """
    from brain_api.core.lstm import load_prices_yfinance

    today = date.today()
    records_labeled = 0
    records_skipped = 0
    errors = []

    # Get records to label
    if request.run_id:
        # Try to load with exact run_id first
        record = storage.load(request.run_id)
        if record is None:
            # If not found, try with model_type suffix (SAC)
            # since store_experience appends :{model_type} to run_id
            for model_type in ["sac"]:
                suffixed_id = f"{request.run_id}:{model_type}"
                record = storage.load(suffixed_id)
                if record:
                    break
        records = [record] if record else []
    else:
        records = storage.list_unlabeled()

    logger.info(f"[Experience] Found {len(records)} records to potentially label")

    for record in records:
        try:
            # Check if week has ended
            week_end = date.fromisoformat(record.week_end)
            if week_end >= today:
                logger.info(f"[Experience] Skipping {record.run_id}: week not ended")
                records_skipped += 1
                continue

            # Get symbols from action
            symbols = [s for s in record.action if s != "CASH"]

            if not symbols:
                logger.warning(f"[Experience] No symbols in action for {record.run_id}")
                records_skipped += 1
                continue

            # Fetch realized returns
            week_start = date.fromisoformat(record.week_start)
            # Fetch a bit more data to ensure we capture the week
            from datetime import timedelta

            data_start = week_start - timedelta(days=7)
            data_end = week_end + timedelta(days=7)

            prices = load_prices_yfinance(symbols, data_start, data_end)

            # Compute first-session-open to next-week-first-session-open
            # returns and use the starting open to size rebalance costs.
            symbol_returns = {}
            symbol_prices: dict[str, float] = {}
            for symbol in symbols:
                df = prices.get(symbol)
                if df is None or df.empty:
                    raise ValueError(
                        f"Missing realized prices for SAC experience symbol {symbol}"
                    )

                trade_price, weekly_return = experience_open_transition(
                    df,
                    week_start,
                    symbol=symbol,
                )
                symbol_returns[symbol] = weekly_return
                symbol_prices[symbol] = trade_price

            prior_weights = _extract_prior_weights(record)
            if record.nav_usd is None:
                logger.warning(
                    f"[Experience] Record {record.run_id} has no nav_usd; "
                    f"falling back to IBKRSingaporeCostConfig default NAV anchor"
                )

            # Compute reward
            reward, realized_return = compute_realized_reward(
                action=record.action,
                symbol_returns=symbol_returns,
                prior_weights=prior_weights,
                symbol_prices=symbol_prices,
                nav_usd=record.nav_usd,
            )

            # Update record
            record.reward = reward
            record.realized_return = realized_return
            record.labeled_at = datetime.now(UTC).isoformat()

            storage.update(record)
            records_labeled += 1

            logger.info(
                f"[Experience] Labeled {record.run_id}: "
                f"reward={reward:.4f}, return={realized_return:.4f}"
            )

        except Exception as e:
            error_msg = f"Error labeling {record.run_id}: {e}"
            logger.error(f"[Experience] {error_msg}")
            errors.append(error_msg)

    logger.info(
        f"[Experience] Labeling complete: {records_labeled} labeled, "
        f"{records_skipped} skipped, {len(errors)} errors"
    )

    return LabelExperienceResponse(
        records_labeled=records_labeled,
        records_skipped=records_skipped,
        errors=errors,
    )


@router.get("/list", response_model=list[ExperienceRecord])
def list_experience(
    labeled_only: bool = False,
    storage: ExperienceStorage = Depends(get_experience_storage),
) -> list[ExperienceRecord]:
    """List experience records.

    Args:
        labeled_only: If True, only return labeled records (for fine-tuning).
    """
    if labeled_only:
        all_records = storage.list_all()
        return [r for r in all_records if r.reward is not None]
    return storage.list_all()


# ============================================================================
# Account-specific labeling endpoints
# ============================================================================


def _compute_reward_from_actual_weights(
    actual_weights: dict[str, float],
    symbol_returns: dict[str, float],
    *,
    prior_weights: dict[str, float] | None = None,
    symbol_prices: dict[str, float] | None = None,
    nav_usd: float | None = None,
    reward_scale: float = 100.0,
) -> tuple[float, float]:
    """Delegate strict SAC realized-reward accounting to the account service."""
    from brain_api.routes.experience_account_labeling import (
        compute_reward_from_actual_weights,
    )

    return compute_reward_from_actual_weights(
        actual_weights,
        symbol_returns,
        prior_weights=prior_weights,
        symbol_prices=symbol_prices,
        nav_usd=nav_usd,
        reward_scale=reward_scale,
    )


def _infer_universe_from_run_id(run_id: str) -> str:
    """Delegate bounded legacy-universe inference to the account service."""
    from brain_api.routes.experience_account_labeling import infer_universe_from_run_id

    return infer_universe_from_run_id(run_id)


def _label_experience_for_account(
    model_type: str,
    run_id: str | None,
    storage: ExperienceStorage,
) -> LabelExperienceResponse:
    """Delegate account-specific SAC labeling to the focused service."""
    from brain_api.routes.experience_account_labeling import (
        label_experience_for_account,
    )

    return label_experience_for_account(model_type, run_id, storage)


@router.post("/label/sac", response_model=LabelExperienceResponse)
def label_sac_experience(
    request: LabelExperienceRequest,
    storage: ExperienceStorage = Depends(get_experience_storage),
) -> LabelExperienceResponse:
    """Label SAC experience records using actual execution from SAC Alpaca account.

    This endpoint:
    1. Finds unlabeled SAC experience records where week_end < today
    2. Fetches actual portfolio weights from SAC Alpaca account
    3. Computes reward based on ACTUAL weights (not intended)
    4. Updates the experience record

    The key difference from the generic /label endpoint is that this uses
    the actual executed portfolio, accounting for any orders that expired
    or only partially filled.
    """
    return _label_experience_for_account(
        model_type="sac",
        run_id=request.run_id,
        storage=storage,
    )

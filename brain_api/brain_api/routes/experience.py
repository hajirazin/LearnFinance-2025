"""Experience buffer and labeling endpoints.

This module provides:
- Storage for PPO experience tuples (state, action, turnover)
- Labeling endpoint to fill in realized rewards after the week ends
- Reading experience for fine-tuning
"""

import json
import logging
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from brain_api.storage.base import DEFAULT_DATA_PATH

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Data models
# ============================================================================


class ExperienceState(BaseModel):
    """Full state at decision time for RL experience."""

    # Per-stock signals
    signals: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="Signals per symbol: {AAPL: {news_sentiment: 0.3, gross_margin: 0.42, ...}}",
    )

    # Forecaster predictions
    lstm_forecasts: dict[str, float] = Field(
        default_factory=dict,
        description="LSTM predicted weekly returns per symbol: {AAPL: 0.012, MSFT: -0.005}",
    )
    patchtst_forecasts: dict[str, float] = Field(
        default_factory=dict,
        description="PatchTST predicted weekly returns per symbol: {AAPL: 0.015, MSFT: -0.003}",
    )

    # Current portfolio weights (before action)
    current_weights: dict[str, float] = Field(
        default_factory=dict,
        description="Current portfolio weights including CASH: {AAPL: 0.10, MSFT: 0.08, CASH: 0.82}",
    )


class OrderExecutionReport(BaseModel):
    """Execution status for a single order."""

    symbol: str
    side: str  # "buy" or "sell"
    intended_qty: float
    filled_qty: float
    filled_avg_price: float | None = None
    status: str  # "filled", "partial", "expired", "rejected", "not_found"
    client_order_id: str | None = None


class ExperienceRecord(BaseModel):
    """A single experience record for RL (PPO/SAC) training.

    Lifecycle:
    1. Store: Called after inference with full state + intended action
    2. Update execution: Called after orders settle with execution_report
    3. Label: Called next week to fill in reward based on actual portfolio
    """

    run_id: str  # e.g., "paper:2026-01-12:ppo" (includes model_type)
    week_start: str  # ISO date
    week_end: str  # ISO date
    model_type: str  # "ppo" or "sac"
    model_version: str

    # Full state at decision time
    state: ExperienceState | dict[str, Any]  # Accept both for backward compatibility

    # Intended action (what the policy decided)
    intended_action: dict[str, float] = Field(
        default_factory=dict,
        description="Target weights from policy: {AAPL: 0.12, MSFT: 0.10, CASH: 0.78}",
    )
    intended_turnover: float = 0.0

    # Legacy field for backward compatibility
    action: dict[str, float] = Field(
        default_factory=dict,
        description="Deprecated: use intended_action instead",
    )
    turnover: float = 0.0  # Deprecated: use intended_turnover

    # Actual execution (filled by update-execution endpoint)
    actual_weights: dict[str, float] | None = Field(
        None,
        description="Actual portfolio weights after orders settled",
    )
    execution_report: list[OrderExecutionReport] | list[dict] | None = Field(
        None,
        description="Per-order execution status",
    )
    execution_updated_at: str | None = None

    # Reward (filled by labeling job)
    reward: float | None = None
    realized_return: float | None = None
    next_state: ExperienceState | dict[str, Any] | None = None
    labeled_at: str | None = None


class LabelExperienceRequest(BaseModel):
    """Request to label experience records with realized rewards."""

    run_id: str | None = Field(
        None,
        description="Specific run ID to label. If None, labels all unlabeled records.",
    )


class LabelExperienceResponse(BaseModel):
    """Response from experience labeling."""

    records_labeled: int
    records_skipped: int  # Already labeled or week not ended
    errors: list[str]


class StoreExperienceRequest(BaseModel):
    """Request to store an experience record with full state."""

    run_id: str
    week_start: str
    week_end: str
    model_type: str  # "ppo" or "sac"
    model_version: str

    # Full state at decision time
    state: ExperienceState | dict[str, Any] = Field(
        ...,
        description="Full state with signals, forecasts, current_weights",
    )

    # Intended action from policy
    intended_action: dict[str, float] = Field(
        default_factory=dict,
        description="Target weights from policy",
    )
    intended_turnover: float = 0.0

    # Legacy fields for backward compatibility
    action: dict[str, float] = Field(
        default_factory=dict,
        description="Deprecated: use intended_action",
    )
    turnover: float = 0.0


class StoreExperienceResponse(BaseModel):
    """Response from storing experience."""

    record_id: str
    stored: bool
    model_type: str


class IntendedOrder(BaseModel):
    """An order that was intended to be submitted."""

    symbol: str
    qty: float
    side: str  # "buy" or "sell"
    client_order_id: str


class ExecutedOrder(BaseModel):
    """An order from Alpaca order history (raw response)."""

    client_order_id: str
    status: str  # "filled", "partially_filled", "canceled", "expired", etc.
    filled_qty: str | None = None
    filled_avg_price: str | None = None


class UpdateExecutionRequest(BaseModel):
    """Request to update experience with execution report after orders settle.

    Can provide EITHER:
    1. Pre-computed execution_report (legacy)
    2. Raw intended_orders + executed_orders (new - matching done internally)
    """

    run_id: str
    model_type: str  # "ppo" or "sac"

    # Option 1: Pre-computed execution report (legacy)
    execution_report: list[dict] | None = Field(
        None,
        description="Pre-computed per-order execution status (legacy)",
    )

    # Option 2: Raw data for internal matching (new)
    intended_orders: list[IntendedOrder] | list[dict] | None = Field(
        None,
        description="Orders we intended to submit (from /orders/generate)",
    )
    executed_orders: list[ExecutedOrder] | list[dict] | None = Field(
        None,
        description="Raw order history from Alpaca (from /alpaca/order-history)",
    )

    actual_weights: dict[str, float] | None = Field(
        None,
        description="Actual portfolio weights after orders settled",
    )


class UpdateExecutionResponse(BaseModel):
    """Response from updating execution report."""

    run_id: str
    updated: bool
    orders_filled: int
    orders_partial: int
    orders_expired: int


# ============================================================================
# Storage helpers
# ============================================================================


class ExperienceStorage:
    """Storage for PPO experience records."""

    def __init__(self, base_path: Path | str | None = None):
        if base_path is None:
            base_path = DEFAULT_DATA_PATH
        self.base_path = Path(base_path)
        self._experience_path = self.base_path / "experience"
        self._experience_path.mkdir(parents=True, exist_ok=True)

    def _record_path(self, run_id: str) -> Path:
        """Get path for a specific run's experience record."""
        # Sanitize run_id for filesystem
        safe_id = run_id.replace(":", "_").replace("/", "_")
        return self._experience_path / f"{safe_id}.json"

    def store(self, record: ExperienceRecord) -> str:
        """Store an experience record.

        Returns:
            Record ID (same as run_id).
        """
        path = self._record_path(record.run_id)
        with open(path, "w") as f:
            json.dump(record.model_dump(), f, indent=2, default=str)
        return record.run_id

    def load(self, run_id: str) -> ExperienceRecord | None:
        """Load an experience record by run_id."""
        path = self._record_path(run_id)
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return ExperienceRecord(**data)

    def list_unlabeled(self) -> list[ExperienceRecord]:
        """List all unlabeled experience records."""
        records = []
        for path in self._experience_path.glob("*.json"):
            with open(path) as f:
                data = json.load(f)
            record = ExperienceRecord(**data)
            if record.reward is None:
                records.append(record)
        return records

    def list_all(self) -> list[ExperienceRecord]:
        """List all experience records."""
        records = []
        for path in self._experience_path.glob("*.json"):
            with open(path) as f:
                data = json.load(f)
            records.append(ExperienceRecord(**data))
        return records

    def update(self, record: ExperienceRecord) -> None:
        """Update an existing experience record."""
        self.store(record)


def get_experience_storage() -> ExperienceStorage:
    """Get experience storage instance."""
    return ExperienceStorage()


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


def compute_realized_reward(
    action: dict[str, float],
    turnover: float,
    symbol_returns: dict[str, float],
    cost_bps: int = 10,
    reward_scale: float = 100.0,
) -> tuple[float, float]:
    """Compute realized reward from actual returns.

    Args:
        action: Target weights at decision time.
        turnover: Portfolio turnover.
        symbol_returns: Realized weekly returns for each symbol.
        cost_bps: Transaction cost in basis points.
        reward_scale: Reward scaling factor.

    Returns:
        Tuple of (reward, realized_return).
    """
    # Compute portfolio return
    portfolio_return = 0.0
    for symbol, weight in action.items():
        if symbol == "CASH":
            continue  # Cash return is 0
        symbol_return = symbol_returns.get(symbol, 0.0)
        portfolio_return += weight * symbol_return

    # Compute transaction cost
    cost_rate = cost_bps / 10_000
    transaction_cost = turnover * cost_rate

    # Compute reward
    net_return = portfolio_return - transaction_cost
    reward = net_return * reward_scale

    return reward, portfolio_return


# ============================================================================
# Endpoints
# ============================================================================


@router.post("/store", response_model=StoreExperienceResponse)
def store_experience(
    request: StoreExperienceRequest,
    storage: ExperienceStorage = Depends(get_experience_storage),
) -> StoreExperienceResponse:
    """Store an experience record with full state.

    This is called after each RL (PPO/SAC) inference to record:
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

    # Create unique run_id including model_type to separate PPO and SAC
    run_id = request.run_id
    if not run_id.endswith(f":{request.model_type}"):
        run_id = f"{request.run_id}:{request.model_type}"

    record = ExperienceRecord(
        run_id=run_id,
        week_start=request.week_start,
        week_end=request.week_end,
        model_type=request.model_type,
        model_version=request.model_version,
        state=request.state,
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
    3. Computes reward = (portfolio_return - transaction_cost) * scale
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
            # If not found, try with model_type suffixes (PPO and SAC)
            # since store_experience appends :{model_type} to run_id
            for model_type in ["ppo", "sac"]:
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

            # Compute weekly returns for each symbol
            symbol_returns = {}
            for symbol in symbols:
                df = prices.get(symbol)
                if df is None or df.empty:
                    symbol_returns[symbol] = 0.0
                    continue

                # Get close prices for the week
                try:
                    # Find closest prices to week start and end
                    start_price = df.loc[df.index >= str(week_start), "close"].iloc[0]
                    end_price = df.loc[df.index <= str(week_end), "close"].iloc[-1]
                    weekly_return = (end_price - start_price) / start_price
                    symbol_returns[symbol] = float(weekly_return)
                except (IndexError, KeyError):
                    symbol_returns[symbol] = 0.0

            # Compute reward
            reward, realized_return = compute_realized_reward(
                action=record.action,
                turnover=record.turnover,
                symbol_returns=symbol_returns,
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
    cost_bps: int = 10,
    reward_scale: float = 100.0,
) -> tuple[float, float]:
    """Compute reward based on ACTUAL portfolio weights (not intended).

    This is the key difference from the legacy labeling - we use what
    actually executed, not what the policy intended.

    Args:
        actual_weights: Actual portfolio weights after orders settled.
        symbol_returns: Realized weekly returns for each symbol.
        cost_bps: Transaction cost in basis points.
        reward_scale: Reward scaling factor.

    Returns:
        Tuple of (reward, portfolio_return).
    """
    import numpy as np

    # Compute portfolio return using ACTUAL weights
    portfolio_return = 0.0
    for symbol, weight in actual_weights.items():
        if symbol == "CASH":
            continue  # Cash return is 0
        symbol_return = symbol_returns.get(symbol, 0.0)
        portfolio_return += weight * symbol_return

    # Use log return for RL (additive across time)
    portfolio_log_return = float(np.log(max(1 + portfolio_return, 1e-10)))

    # Transaction cost is computed from turnover (we use a simpler estimate here)
    # In a more sophisticated version, we could compute actual turnover from
    # execution_report
    cost_rate = cost_bps / 10_000
    estimated_turnover = 0.1  # Estimate 10% turnover per week

    transaction_cost = estimated_turnover * cost_rate
    # Both terms in log space for mathematical consistency:
    # net_return = log(1 + r) - log(1 + tc) = log((1 + r) / (1 + tc))
    net_return = portfolio_log_return - np.log(1 + transaction_cost)
    reward = net_return * reward_scale

    return reward, portfolio_return


def _label_experience_for_account(
    model_type: str,
    run_id: str | None,
    storage: ExperienceStorage,
) -> LabelExperienceResponse:
    """Label experience records for a specific account using actual weights.

    Args:
        model_type: "ppo" or "sac"
        run_id: Specific run to label, or None to label all unlabeled.
        storage: Experience storage instance.

    Returns:
        LabelExperienceResponse with labeling results.
    """
    from datetime import timedelta

    from brain_api.core.alpaca_client import AlpacaAccount, get_alpaca_client
    from brain_api.core.lstm import load_prices_yfinance

    today = date.today()
    records_labeled = 0
    records_skipped = 0
    errors = []

    # Get Alpaca client for this account
    try:
        alpaca_client = get_alpaca_client(AlpacaAccount(model_type))
    except ValueError as e:
        logger.error(f"[Experience] Failed to get Alpaca client for {model_type}: {e}")
        return LabelExperienceResponse(
            records_labeled=0,
            records_skipped=0,
            errors=[str(e)],
        )

    # Get records to label
    if run_id:
        # Add model_type suffix if not present
        if not run_id.endswith(f":{model_type}"):
            run_id = f"{run_id}:{model_type}"
        record = storage.load(run_id)
        records = [record] if record else []
    else:
        # Get all unlabeled records for this model_type
        all_unlabeled = storage.list_unlabeled()
        records = [r for r in all_unlabeled if r.model_type == model_type]

    logger.info(
        f"[Experience] Found {len(records)} {model_type.upper()} records to potentially label"
    )

    for record in records:
        try:
            # Check if week has ended
            week_end = date.fromisoformat(record.week_end)
            if week_end >= today:
                logger.info(
                    f"[Experience] Skipping {record.run_id}: week not ended yet"
                )
                records_skipped += 1
                continue

            # Get ACTUAL weights from Alpaca account
            # If we have actual_weights from update-execution, use those
            # Otherwise, fetch current positions (less accurate but fallback)
            if record.actual_weights:
                actual_weights = record.actual_weights
                logger.info(
                    f"[Experience] Using stored actual_weights for {record.run_id}"
                )
            else:
                try:
                    actual_weights = alpaca_client.get_portfolio_weights()
                    logger.info(
                        f"[Experience] Fetched current weights from Alpaca for {record.run_id}"
                    )
                except Exception as e:
                    logger.warning(
                        f"[Experience] Failed to fetch Alpaca weights: {e}. "
                        f"Falling back to intended action."
                    )
                    # Fallback to intended action if we can't get actual
                    actual_weights = record.intended_action or record.action

            # Get symbols from actual weights
            symbols = [s for s in actual_weights if s != "CASH"]

            if not symbols:
                logger.warning(
                    f"[Experience] No symbols in actual_weights for {record.run_id}"
                )
                records_skipped += 1
                continue

            # Fetch realized returns
            week_start = date.fromisoformat(record.week_start)
            data_start = week_start - timedelta(days=7)
            data_end = week_end + timedelta(days=7)

            prices = load_prices_yfinance(symbols, data_start, data_end)

            # Compute weekly returns for each symbol
            symbol_returns = {}
            for symbol in symbols:
                df = prices.get(symbol)
                if df is None or df.empty:
                    symbol_returns[symbol] = 0.0
                    continue

                try:
                    start_price = df.loc[df.index >= str(week_start), "close"].iloc[0]
                    end_price = df.loc[df.index <= str(week_end), "close"].iloc[-1]
                    weekly_return = (end_price - start_price) / start_price
                    symbol_returns[symbol] = float(weekly_return)
                except (IndexError, KeyError):
                    symbol_returns[symbol] = 0.0

            # Compute reward using ACTUAL weights
            reward, realized_return = _compute_reward_from_actual_weights(
                actual_weights=actual_weights,
                symbol_returns=symbol_returns,
            )

            # Update record
            record.reward = reward
            record.realized_return = realized_return
            record.actual_weights = actual_weights
            record.labeled_at = datetime.now(UTC).isoformat()

            storage.update(record)
            records_labeled += 1

            logger.info(
                f"[Experience] Labeled {model_type.upper()} {record.run_id}: "
                f"reward={reward:.4f}, return={realized_return:.4f}"
            )

        except Exception as e:
            error_msg = f"Error labeling {record.run_id}: {e}"
            logger.error(f"[Experience] {error_msg}")
            errors.append(error_msg)

    logger.info(
        f"[Experience] {model_type.upper()} labeling complete: "
        f"{records_labeled} labeled, {records_skipped} skipped, {len(errors)} errors"
    )

    return LabelExperienceResponse(
        records_labeled=records_labeled,
        records_skipped=records_skipped,
        errors=errors,
    )


@router.post("/label/ppo", response_model=LabelExperienceResponse)
def label_ppo_experience(
    request: LabelExperienceRequest,
    storage: ExperienceStorage = Depends(get_experience_storage),
) -> LabelExperienceResponse:
    """Label PPO experience records using actual execution from PPO Alpaca account.

    This endpoint:
    1. Finds unlabeled PPO experience records where week_end < today
    2. Fetches actual portfolio weights from PPO Alpaca account
    3. Computes reward based on ACTUAL weights (not intended)
    4. Updates the experience record

    The key difference from the generic /label endpoint is that this uses
    the actual executed portfolio, accounting for any orders that expired
    or only partially filled.
    """
    return _label_experience_for_account(
        model_type="ppo",
        run_id=request.run_id,
        storage=storage,
    )


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

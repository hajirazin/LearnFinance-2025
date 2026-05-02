"""Experience buffer and labeling endpoints.

This module provides:
- Storage for RL (SAC) experience tuples (state, action, turnover)
- Labeling endpoint to fill in realized rewards after the week ends
- Reading experience for fine-tuning
"""

import json
import logging
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
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
    """A single experience record for RL (SAC) training.

    Lifecycle:
    1. Store: Called after inference with full state + intended action
    2. Update execution: Called after orders settle with execution_report
    3. Label: Called next week to fill in reward based on actual portfolio
    """

    run_id: str  # e.g., "paper:2026-01-12:sac" (includes model_type)
    week_start: str  # ISO date
    week_end: str  # ISO date
    model_type: str  # "sac"
    model_version: str

    # Bucket discriminator. Optional with ``None`` for backward
    # compatibility with experience JSON files written before this field
    # existed; the labeller falls back to inferring the universe from
    # the run_id prefix in that case (logged at WARNING). New writes
    # MUST set this so the labeller can route to the right Alpaca
    # account without inference.
    universe: str | None = None

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
    # Total portfolio equity (cash + positions market value) at the time of
    # the post-trade snapshot, in USD. Required by the IBKR-SG cost model
    # in the labeller to convert weight deltas into share counts. Optional
    # only for backward compatibility with experience records written
    # before this field existed; the labeller falls back to the IBKR cost
    # config's default NAV anchor (USD 10k) and logs a WARNING in that
    # case.
    nav_usd: float | None = Field(
        None,
        description=(
            "Post-trade total portfolio equity in USD; used by the IBKR "
            "cost model in the labeller. Should be set by "
            "/experience/update-execution from the broker portfolio "
            "snapshot."
        ),
    )

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
    model_type: str  # "sac"
    model_version: str

    # Bucket discriminator. Optional only because legacy callers (and a
    # handful of test fixtures) predate this field; the SAC Temporal
    # workflows always pass it now so /experience/label/sac can route
    # to the right Alpaca account without inferring from run_id.
    universe: str | None = Field(
        None,
        description=(
            "SAC bucket universe (e.g. 'halal_filtered' or 'halal'). "
            "Required for new SAC writes so the labeller can route to "
            "the correct Alpaca account."
        ),
    )

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
    model_type: str  # "sac"

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

    nav_usd: float | None = Field(
        None,
        description=(
            "Post-trade total portfolio equity in USD (cash + positions "
            "market_value). Plumbed in from the broker snapshot so the "
            "labeller's IBKR cost model can size shares correctly."
        ),
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
    """Storage for RL experience records."""

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


def _build_weight_arrays(
    prior_weights: dict[str, float],
    target_weights: dict[str, float],
    symbol_prices: dict[str, float],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Build (symbol_order, prior_w, target_w, prices) arrays for the cost model.

    Aligns ``prior_weights`` and ``target_weights`` on the union of
    their stock symbols (CASH is appended as the last slot). Missing
    weights are treated as 0.0 -- a symbol the policy fully sold (or
    fully bought into from cash) is the canonical use-case for that.

    Per AGENTS.md rule #1, raises if a non-zero-delta symbol is
    missing a price -- silently zero-costing a real trade is the
    failure mode this guard exists to prevent.
    """
    stock_symbols = sorted(
        {s for s in prior_weights if s != "CASH"}
        | {s for s in target_weights if s != "CASH"}
    )
    n_stocks = len(stock_symbols)

    prior_w = np.zeros(n_stocks + 1)
    target_w = np.zeros(n_stocks + 1)
    prices = np.zeros(n_stocks)

    for stock_idx, symbol in enumerate(stock_symbols):
        prior_w[stock_idx] = float(prior_weights.get(symbol, 0.0))
        target_w[stock_idx] = float(target_weights.get(symbol, 0.0))
        delta = abs(target_w[stock_idx] - prior_w[stock_idx])
        if delta > 1e-9:
            price = symbol_prices.get(symbol)
            if price is None or not np.isfinite(price) or price <= 0:
                raise ValueError(
                    f"price for {symbol!r} required to size the rebalance "
                    f"leg (delta_w={delta}); per AGENTS.md rule #1 we "
                    f"refuse to silently zero-cost a real trade. "
                    f"Got price={price!r}."
                )
            prices[stock_idx] = float(price)

    prior_w[-1] = float(prior_weights.get("CASH", 0.0))
    target_w[-1] = float(target_weights.get("CASH", 0.0))

    return stock_symbols, prior_w, target_w, prices


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
    from brain_api.core.portfolio_rl.broker_costs import (
        IBKRSingaporeCostConfig,
        compute_ibkr_rebalance_cost,
    )

    portfolio_return = 0.0
    for symbol, weight in action.items():
        if symbol == "CASH":
            continue  # Cash return is 0
        portfolio_return += weight * symbol_returns.get(symbol, 0.0)

    if prior_weights is None:
        prior_weights = {"CASH": 1.0}
    if symbol_prices is None:
        symbol_prices = {}

    cfg = IBKRSingaporeCostConfig.default()
    if nav_usd is not None:
        cfg = cfg.with_nav(nav_usd)

    symbol_order, prior_w, target_w, prices = _build_weight_arrays(
        prior_weights=prior_weights,
        target_weights=action,
        symbol_prices=symbol_prices,
    )
    rebalance_cost = compute_ibkr_rebalance_cost(
        symbol_order=symbol_order,
        current_weights=prior_w,
        target_weights=target_w,
        prices=prices,
        cfg=cfg,
    )
    tc_fraction = rebalance_cost.total_fraction

    # Log-space reward for mathematical consistency (matches env.step).
    portfolio_log_return = float(np.log(max(1 + portfolio_return, 1e-10)))
    net_return = portfolio_log_return - np.log(1 + tc_fraction)
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

    record = ExperienceRecord(
        run_id=run_id,
        week_start=request.week_start,
        week_end=request.week_end,
        model_type=request.model_type,
        model_version=request.model_version,
        universe=request.universe,
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

            # Compute weekly returns AND end-of-week prices for each
            # symbol. The end_price is what the IBKR-SG cost model uses
            # to convert weight deltas into share counts.
            symbol_returns = {}
            symbol_prices: dict[str, float] = {}
            for symbol in symbols:
                df = prices.get(symbol)
                if df is None or df.empty:
                    symbol_returns[symbol] = 0.0
                    continue

                try:
                    # Find closest prices to week start and end
                    start_price = df.loc[df.index >= str(week_start), "close"].iloc[0]
                    end_price = df.loc[df.index <= str(week_end), "close"].iloc[-1]
                    weekly_return = (end_price - start_price) / start_price
                    symbol_returns[symbol] = float(weekly_return)
                    symbol_prices[symbol] = float(end_price)
                except (IndexError, KeyError):
                    symbol_returns[symbol] = 0.0

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
    """Compute reward based on ACTUAL portfolio weights using IBKR-SG costs.

    Differs from the simulator (env.step) in that the rebalance is
    measured against what actually executed (``actual_weights``) vs
    what the policy intended; the cost formula itself is identical
    -- per-symbol per-leg IBKR Singapore Tiered fees in
    :mod:`brain_api.core.portfolio_rl.broker_costs`.

    The legacy "estimated_turnover = 0.1" hack is gone -- the cost is
    now derived from the **actual** per-symbol weight deltas
    (``actual_weights`` - ``prior_weights``) and the **actual**
    per-symbol prices we have on record. If either side is missing
    for a symbol that traded, we raise per AGENTS.md rule #1.

    Args:
        actual_weights: Actual portfolio weights after orders settled.
        symbol_returns: Realized weekly returns for each symbol.
        prior_weights: Pre-rebalance weights (defaults to all-cash).
        symbol_prices: Per-symbol close prices for the rebalance week.
        nav_usd: Total portfolio equity in USD; defaults to the IBKR
            cost config's USD 10k anchor when None.
        reward_scale: Reward scaling factor.

    Returns:
        Tuple of (reward, portfolio_return).
    """
    from brain_api.core.portfolio_rl.broker_costs import (
        IBKRSingaporeCostConfig,
        compute_ibkr_rebalance_cost,
    )

    portfolio_return = 0.0
    for symbol, weight in actual_weights.items():
        if symbol == "CASH":
            continue  # Cash return is 0
        portfolio_return += weight * symbol_returns.get(symbol, 0.0)

    portfolio_log_return = float(np.log(max(1 + portfolio_return, 1e-10)))

    if prior_weights is None:
        prior_weights = {"CASH": 1.0}
    if symbol_prices is None:
        symbol_prices = {}

    cfg = IBKRSingaporeCostConfig.default()
    if nav_usd is not None:
        cfg = cfg.with_nav(nav_usd)

    symbol_order, prior_w, target_w, prices = _build_weight_arrays(
        prior_weights=prior_weights,
        target_weights=actual_weights,
        symbol_prices=symbol_prices,
    )
    rebalance_cost = compute_ibkr_rebalance_cost(
        symbol_order=symbol_order,
        current_weights=prior_w,
        target_weights=target_w,
        prices=prices,
        cfg=cfg,
    )
    tc_fraction = rebalance_cost.total_fraction

    net_return = portfolio_log_return - np.log(1 + tc_fraction)
    reward = net_return * reward_scale

    return reward, portfolio_return


def _infer_universe_from_run_id(run_id: str) -> str:
    """Infer the SAC universe from a legacy run_id (no ``universe`` field).

    Used as a one-shot migration aid for experience records written
    before the ``universe`` field existed. The two parallel SAC A/B
    workflows have disjoint run_id prefixes by design (per AGENTS.md
    "Run identity & rerun semantics"):

    - ``paper:halal:YYYY-MM-DD[:sac]`` -> ``halal`` (IBKR-routed; the
      Alpaca labeller has no account for this universe and will
      surface an error on resolve_alpaca_account -- see AGENTS.md
      rule #1)
    - everything else                 -> ``halal_filtered`` (sac account)

    Per AGENTS.md rule #1 the inference is intentionally bounded to the
    two known SAC universes -- a future third bucket would need to land
    a ``universe`` field on the record before its experience is
    written, NOT a silent fallback here.
    """
    if run_id.startswith("paper:halal:"):
        return "halal"
    return "halal_filtered"


def _label_experience_for_account(
    model_type: str,
    run_id: str | None,
    storage: ExperienceStorage,
) -> LabelExperienceResponse:
    """Label experience records for a model type using actual weights.

    Routes each record to the correct Alpaca account via
    :func:`resolve_alpaca_account` driven by ``record.universe``. Only
    ``halal_filtered`` is currently Alpaca-routable; ``halal`` records
    are IBKR-routed and MUST carry ``actual_weights`` plumbed in from
    the post-trade IBKR snapshot at write time -- the Alpaca fallback
    cannot serve them and will fail-loud per AGENTS.md rule #1.

    Args:
        model_type: ``"sac"`` (currently the only labeller-supported
            model type; see :func:`resolve_alpaca_account`).
        run_id: Specific run to label, or ``None`` to label every
            unlabeled record for this ``model_type``.
        storage: Experience storage instance.

    Returns:
        LabelExperienceResponse with labeling results.
    """
    from datetime import timedelta

    from brain_api.core.alpaca_client import (
        AlpacaClient,
        get_alpaca_client,
        resolve_alpaca_account,
    )
    from brain_api.core.lstm import load_prices_yfinance

    today = date.today()
    records_labeled = 0
    records_skipped = 0
    errors = []

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

    # Cache one client per resolved account so a mixed-universe batch
    # only constructs each Alpaca client once (and so the labeller does
    # not re-read env vars per record).
    client_cache: dict[str, AlpacaClient] = {}

    def _get_client_for_record(rec: ExperienceRecord) -> AlpacaClient:
        universe = rec.universe
        if universe is None:
            universe = _infer_universe_from_run_id(rec.run_id)
            logger.warning(
                f"[Experience] Record {rec.run_id} has no universe field; "
                f"inferred universe={universe!r} from run_id prefix. "
                f"This path is for legacy records only -- new SAC writes "
                f"set universe explicitly."
            )
        account = resolve_alpaca_account(rec.model_type, universe)
        cached = client_cache.get(account.value)
        if cached is None:
            cached = get_alpaca_client(account)
            client_cache[account.value] = cached
        return cached

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
                    alpaca_client = _get_client_for_record(record)
                    actual_weights = alpaca_client.get_portfolio_weights()
                    logger.info(
                        f"[Experience] Fetched current weights from Alpaca "
                        f"({alpaca_client.account.value}) for {record.run_id}"
                    )
                except ValueError as e:
                    # Unknown (model_type, universe) -> we cannot pick an
                    # account. Per AGENTS.md rule #1, surface as an error
                    # rather than silently labelling against the wrong
                    # account.
                    error_msg = (
                        f"Cannot route {record.run_id} to an Alpaca "
                        f"account (model_type={record.model_type!r}, "
                        f"universe={record.universe!r}): {e}"
                    )
                    logger.error(f"[Experience] {error_msg}")
                    errors.append(error_msg)
                    continue
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

            # Compute weekly returns AND end-of-week prices for each
            # symbol. The end_price feeds the IBKR-SG cost model in
            # _compute_reward_from_actual_weights.
            symbol_returns = {}
            symbol_prices: dict[str, float] = {}
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
                    symbol_prices[symbol] = float(end_price)
                except (IndexError, KeyError):
                    symbol_returns[symbol] = 0.0

            prior_weights = _extract_prior_weights(record)
            if record.nav_usd is None:
                logger.warning(
                    f"[Experience] Record {record.run_id} has no nav_usd; "
                    f"falling back to IBKRSingaporeCostConfig default NAV anchor"
                )

            # Compute reward using ACTUAL weights
            reward, realized_return = _compute_reward_from_actual_weights(
                actual_weights=actual_weights,
                symbol_returns=symbol_returns,
                prior_weights=prior_weights,
                symbol_prices=symbol_prices,
                nav_usd=record.nav_usd,
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

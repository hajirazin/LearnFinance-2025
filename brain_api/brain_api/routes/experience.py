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


class ExperienceRecord(BaseModel):
    """A single experience record for PPO training."""

    run_id: str  # e.g., "paper:2026-01-12"
    week_start: str  # ISO date
    week_end: str  # ISO date
    model_type: str  # "ppo_lstm" or "ppo_patchtst"
    model_version: str

    # State at decision time
    state: dict[str, Any]  # signals, forecasts, current_weights

    # Action taken
    action: dict[str, float]  # target weights
    turnover: float

    # Filled in by labeling job
    reward: float | None = None
    realized_return: float | None = None
    next_state: dict[str, Any] | None = None
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
    """Request to store an experience record."""

    run_id: str
    week_start: str
    week_end: str
    model_type: str
    model_version: str
    state: dict[str, Any]
    action: dict[str, float]
    turnover: float


class StoreExperienceResponse(BaseModel):
    """Response from storing experience."""

    record_id: str
    stored: bool


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
    """Store an experience record.

    This is called after each PPO inference to record the decision
    for later reward labeling and fine-tuning.
    """
    record = ExperienceRecord(
        run_id=request.run_id,
        week_start=request.week_start,
        week_end=request.week_end,
        model_type=request.model_type,
        model_version=request.model_version,
        state=request.state,
        action=request.action,
        turnover=request.turnover,
    )

    record_id = storage.store(record)
    logger.info(f"[Experience] Stored record: {record_id}")

    return StoreExperienceResponse(
        record_id=record_id,
        stored=True,
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
        record = storage.load(request.run_id)
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


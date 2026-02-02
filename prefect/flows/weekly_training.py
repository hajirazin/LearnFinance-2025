"""Weekly training workflow for LearnFinance-2025.

This flow runs every Sunday at 11 AM UTC and executes the full training pipeline:
1. Get halal universe (symbols list)
2. Refresh training data (sentiment gaps + fundamentals)
3. Train LSTM (pure price forecaster)
4. Train PatchTST (multi-signal forecaster)
5. Train PPO (RL allocator)
6. Train SAC (RL allocator)
"""

import os

import httpx
from prefect import flow, task
from prefect.logging import get_run_logger

from flows.models import (
    HalalUniverseResponse,
    RefreshTrainingDataRequest,
    RefreshTrainingDataResponse,
    TrainingResponse,
)

# Configuration
BRAIN_API_URL = os.environ.get("BRAIN_API_URL", "http://localhost:8000")

# Timeout settings (training can take 30+ minutes)
DEFAULT_TIMEOUT = httpx.Timeout(
    connect=30.0,
    read=3600.0,  # 1 hour read timeout for long-running training
    write=30.0,
    pool=30.0,
)


def get_client() -> httpx.Client:
    """Create an HTTP client with appropriate timeouts."""
    return httpx.Client(base_url=BRAIN_API_URL, timeout=DEFAULT_TIMEOUT)


# =============================================================================
# Tasks
# =============================================================================


@task(name="Get Halal Universe", retries=2, retry_delay_seconds=30)
def get_halal_universe() -> HalalUniverseResponse:
    """Fetch the list of halal stock symbols from brain_api."""
    logger = get_run_logger()
    logger.info("Fetching halal universe from brain_api...")

    with get_client() as client:
        response = client.get("/universe/halal")
        response.raise_for_status()
        data = response.json()

    result = HalalUniverseResponse(**data)
    logger.info(f"Got {result.count} halal symbols")
    return result


@task(name="Refresh Training Data", retries=1, retry_delay_seconds=60)
def refresh_training_data(symbols: list[str]) -> RefreshTrainingDataResponse:
    """Refresh sentiment gaps and stale fundamentals for training."""
    logger = get_run_logger()
    logger.info(f"Refreshing training data for {len(symbols)} symbols...")

    request = RefreshTrainingDataRequest(symbols=symbols)

    with get_client() as client:
        response = client.post(
            "/etl/refresh-training-data",
            json=request.model_dump(exclude_none=True),
        )
        response.raise_for_status()
        data = response.json()

    result = RefreshTrainingDataResponse(**data)
    logger.info(
        f"Refresh complete in {result.duration_seconds:.1f}s: "
        f"{result.sentiment_gaps_filled} sentiment gaps filled, "
        f"{len(result.fundamentals_refreshed)} fundamentals refreshed"
    )
    return result


@task(name="Train LSTM", retries=1, retry_delay_seconds=120)
def train_lstm() -> TrainingResponse:
    """Train the LSTM pure-price forecaster model."""
    logger = get_run_logger()
    logger.info("Starting LSTM training...")

    with get_client() as client:
        response = client.post("/train/lstm")
        response.raise_for_status()
        data = response.json()

    result = TrainingResponse(**data)
    logger.info(
        f"LSTM training complete: version={result.version}, promoted={result.promoted}"
    )
    return result


@task(name="Train PatchTST", retries=1, retry_delay_seconds=120)
def train_patchtst() -> TrainingResponse:
    """Train the PatchTST multi-signal forecaster model."""
    logger = get_run_logger()
    logger.info("Starting PatchTST training...")

    with get_client() as client:
        response = client.post("/train/patchtst")
        response.raise_for_status()
        data = response.json()

    result = TrainingResponse(**data)
    logger.info(
        f"PatchTST training complete: version={result.version}, "
        f"promoted={result.promoted}"
    )
    return result


@task(name="Train PPO", retries=1, retry_delay_seconds=120)
def train_ppo() -> TrainingResponse:
    """Train the PPO reinforcement learning allocator."""
    logger = get_run_logger()
    logger.info("Starting PPO training...")

    with get_client() as client:
        response = client.post("/train/ppo/full")
        response.raise_for_status()
        data = response.json()

    result = TrainingResponse(**data)
    logger.info(
        f"PPO training complete: version={result.version}, promoted={result.promoted}"
    )
    return result


@task(name="Train SAC", retries=1, retry_delay_seconds=120)
def train_sac() -> TrainingResponse:
    """Train the SAC reinforcement learning allocator."""
    logger = get_run_logger()
    logger.info("Starting SAC training...")

    with get_client() as client:
        response = client.post("/train/sac/full")
        response.raise_for_status()
        data = response.json()

    result = TrainingResponse(**data)
    logger.info(
        f"SAC training complete: version={result.version}, promoted={result.promoted}"
    )
    return result


# =============================================================================
# Flow
# =============================================================================


@flow(
    name="Weekly Training Pipeline",
    description="Full training pipeline for all models (LSTM, PatchTST, PPO, SAC)",
    retries=0,
    timeout_seconds=14400,  # 4 hours total timeout
)
def weekly_training_flow() -> dict:
    """Execute the full weekly training pipeline.

    Flow diagram (dependencies):
    ```
    get_halal_universe
           │
           ▼
    refresh_training_data
           │
           ├──────────────┐
           ▼              ▼
      train_lstm    train_patchtst
           │              │
           └──────┬───────┘
                  │
           ┌──────┴───────┐
           ▼              ▼
       train_ppo      train_sac
    ```

    Returns:
        dict with training results for each model
    """
    logger = get_run_logger()
    logger.info("Starting weekly training pipeline...")

    # Step 1: Get symbols
    universe = get_halal_universe()

    # Step 2: Refresh training data (uses universe.symbols)
    refresh_result = refresh_training_data(universe.symbols)

    # Step 3 & 4: Train forecasters (can run in parallel after refresh)
    # Using .submit() for concurrent execution and dependency tracking
    lstm_future = train_lstm.submit()
    patchtst_future = train_patchtst.submit()

    # Wait for both forecasters to complete
    lstm_result = lstm_future.result()
    patchtst_result = patchtst_future.result()

    # Step 5 & 6: Train RL allocators (can run in parallel after forecasters)
    ppo_future = train_ppo.submit()
    sac_future = train_sac.submit()

    # Wait for both allocators to complete
    ppo_result = ppo_future.result()
    sac_result = sac_future.result()

    logger.info("Weekly training pipeline complete!")

    return {
        "universe_count": universe.count,
        "refresh": {
            "sentiment_gaps_filled": refresh_result.sentiment_gaps_filled,
            "fundamentals_refreshed": len(refresh_result.fundamentals_refreshed),
        },
        "lstm": {"version": lstm_result.version, "promoted": lstm_result.promoted},
        "patchtst": {
            "version": patchtst_result.version,
            "promoted": patchtst_result.promoted,
        },
        "ppo": {"version": ppo_result.version, "promoted": ppo_result.promoted},
        "sac": {"version": sac_result.version, "promoted": sac_result.promoted},
    }


# =============================================================================
# Deployment (for prefect deploy)
# =============================================================================

if __name__ == "__main__":
    # Run the flow directly for testing
    weekly_training_flow()

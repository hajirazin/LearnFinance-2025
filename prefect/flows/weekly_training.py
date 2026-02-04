"""Weekly training workflow for LearnFinance-2025.

This flow runs every Sunday at 11 AM UTC and executes the full training pipeline:
1. Get halal universe (symbols list)
2. Refresh training data (sentiment gaps + fundamentals)
3. Train LSTM (pure price forecaster)
4. Train PatchTST (multi-signal forecaster)
5. Train PPO (RL allocator)
6. Train SAC (RL allocator)
7. Generate training summary (LLM-powered analysis)
8. Send training summary email
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
    TrainingSummaryEmailResponse,
    TrainingSummaryResponse,
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
    logger.info(f"Got {result.total_stocks} halal symbols")
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


@task(name="Generate Training Summary", retries=1, retry_delay_seconds=30)
def generate_training_summary(
    lstm: TrainingResponse,
    patchtst: TrainingResponse,
    ppo: TrainingResponse,
    sac: TrainingResponse,
) -> TrainingSummaryResponse:
    """Generate LLM summary of all training results.

    Calls POST /llm/training-summary with all 4 training results
    to generate an AI-powered analysis of the training run.

    Args:
        lstm: LSTM training result
        patchtst: PatchTST training result
        ppo: PPO training result
        sac: SAC training result

    Returns:
        TrainingSummaryResponse with LLM-generated summary
    """
    logger = get_run_logger()
    logger.info("Generating training summary via LLM...")

    # Build request payload matching brain_api's TrainingSummaryRequest
    payload = {
        "lstm": {
            "version": lstm.version,
            "data_window_start": lstm.data_window_start,
            "data_window_end": lstm.data_window_end,
            "metrics": lstm.metrics,
            "promoted": lstm.promoted,
        },
        "patchtst": {
            "version": patchtst.version,
            "data_window_start": patchtst.data_window_start,
            "data_window_end": patchtst.data_window_end,
            "metrics": patchtst.metrics,
            "promoted": patchtst.promoted,
            "num_input_channels": patchtst.num_input_channels or 0,
            "signals_used": patchtst.signals_used or [],
        },
        "ppo": {
            "version": ppo.version,
            "data_window_start": ppo.data_window_start,
            "data_window_end": ppo.data_window_end,
            "metrics": ppo.metrics,
            "promoted": ppo.promoted,
            "symbols_used": ppo.symbols_used or [],
        },
        "sac": {
            "version": sac.version,
            "data_window_start": sac.data_window_start,
            "data_window_end": sac.data_window_end,
            "metrics": sac.metrics,
            "promoted": sac.promoted,
            "symbols_used": sac.symbols_used or [],
        },
    }

    with get_client() as client:
        response = client.post("/llm/training-summary", json=payload)
        response.raise_for_status()
        data = response.json()

    result = TrainingSummaryResponse(**data)
    logger.info(
        f"Training summary generated via {result.provider} ({result.model_used}), "
        f"tokens_used={result.tokens_used}"
    )

    # Log the summary content
    logger.info("=== Training Summary ===")
    for key, value in result.summary.items():
        logger.info(f"{key}: {value}")
    logger.info("========================")

    return result


@task(name="Send Training Summary Email", retries=1, retry_delay_seconds=30)
def send_training_summary_email(
    lstm: TrainingResponse,
    patchtst: TrainingResponse,
    ppo: TrainingResponse,
    sac: TrainingResponse,
    summary: TrainingSummaryResponse,
) -> TrainingSummaryEmailResponse:
    """Send training summary via email.

    Calls POST /email/training-summary with all training results and LLM summary
    to send an email notification with side-by-side comparison tables.

    Args:
        lstm: LSTM training result
        patchtst: PatchTST training result
        ppo: PPO training result
        sac: SAC training result
        summary: LLM-generated summary from generate_training_summary

    Returns:
        TrainingSummaryEmailResponse with success status and email details
    """
    logger = get_run_logger()
    logger.info("Sending training summary email...")

    # Build request payload matching brain_api's TrainingSummaryEmailRequest
    payload = {
        "lstm": {
            "version": lstm.version,
            "data_window_start": lstm.data_window_start,
            "data_window_end": lstm.data_window_end,
            "metrics": lstm.metrics,
            "promoted": lstm.promoted,
        },
        "patchtst": {
            "version": patchtst.version,
            "data_window_start": patchtst.data_window_start,
            "data_window_end": patchtst.data_window_end,
            "metrics": patchtst.metrics,
            "promoted": patchtst.promoted,
            "num_input_channels": patchtst.num_input_channels or 0,
            "signals_used": patchtst.signals_used or [],
        },
        "ppo": {
            "version": ppo.version,
            "data_window_start": ppo.data_window_start,
            "data_window_end": ppo.data_window_end,
            "metrics": ppo.metrics,
            "promoted": ppo.promoted,
            "symbols_used": ppo.symbols_used or [],
        },
        "sac": {
            "version": sac.version,
            "data_window_start": sac.data_window_start,
            "data_window_end": sac.data_window_end,
            "metrics": sac.metrics,
            "promoted": sac.promoted,
            "symbols_used": sac.symbols_used or [],
        },
        "summary": summary.summary,
    }

    with get_client() as client:
        response = client.post("/email/training-summary", json=payload)
        response.raise_for_status()
        data = response.json()

    result = TrainingSummaryEmailResponse(**data)
    logger.info(
        f"Training summary email sent: is_success={result.is_success}, "
        f"subject={result.subject}"
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
    persist_result=True,  # Persist task results to allow resume from failure
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
           │              │
           └──────┬───────┘
                  │
                  ▼
    generate_training_summary
                  │
                  ▼
    send_training_summary_email
    ```

    Returns:
        dict with training results for each model, LLM summary, and email status
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

    # Step 7: Generate training summary using LLM
    summary_result = generate_training_summary(
        lstm=lstm_result,
        patchtst=patchtst_result,
        ppo=ppo_result,
        sac=sac_result,
    )

    # Step 8: Send training summary email
    email_result = send_training_summary_email(
        lstm=lstm_result,
        patchtst=patchtst_result,
        ppo=ppo_result,
        sac=sac_result,
        summary=summary_result,
    )

    logger.info("Weekly training pipeline complete!")

    return {
        "universe_count": universe.total_stocks,
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
        "summary": {
            "provider": summary_result.provider,
            "model_used": summary_result.model_used,
            "content": summary_result.summary,
        },
        "email": {
            "is_success": email_result.is_success,
            "subject": email_result.subject,
        },
    }


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        # Run flow once immediately for testing
        weekly_training_flow()
    else:
        # Create deployment and serve with cron schedule
        # Every Sunday at 11 AM UTC: "0 11 * * 0"
        weekly_training_flow.serve(
            name="weekly-training",
            cron="0 11 * * 0",
        )

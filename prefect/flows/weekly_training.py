"""Weekly training workflow for LearnFinance-2025.

This flow runs every Sunday at 11 AM UTC and executes the full training pipeline:
1. Fetch halal_new universe (~410 symbols, fail fast if scraping broken)
2. Train LSTM (pure price forecaster on all ~410 halal_new)
3. Train PatchTST (OHLCV forecaster on all ~410 halal_new)
4. Fetch halal_filtered universe (PatchTST forecast -> top 15)
5. Refresh training data (signals for filtered 15 only)
6. Train PPO (RL allocator on filtered 15)
7. Train SAC (RL allocator on filtered 15)
8. Generate training summary (LLM-powered analysis)
9. Send training summary email
"""

import os
from datetime import timedelta

import httpx
from prefect import flow, task
from prefect.cache_policies import INPUTS
from prefect.logging import get_run_logger

from flows.models import (
    RefreshTrainingDataRequest,
    RefreshTrainingDataResponse,
    TrainingResponse,
    TrainingSummaryEmailResponse,
    TrainingSummaryResponse,
)

# Configuration
BRAIN_API_URL = os.environ.get("BRAIN_API_URL", "http://localhost:8000")

# Timeout settings (training on 400+ symbols can take several hours)
DEFAULT_TIMEOUT = httpx.Timeout(
    connect=30.0,
    read=28800.0,  # 8 hour read timeout for long-running training
    write=30.0,
    pool=30.0,
)


def get_client() -> httpx.Client:
    """Create an HTTP client with appropriate timeouts."""
    return httpx.Client(base_url=BRAIN_API_URL, timeout=DEFAULT_TIMEOUT)


WEEKLY_CACHE_TTL = timedelta(days=7)

# =============================================================================
# Tasks
# =============================================================================


@task(
    name="Refresh Training Data",
    retries=1,
    retry_delay_seconds=60,
    persist_result=True,
    cache_policy=INPUTS,
    cache_expiration=WEEKLY_CACHE_TTL,
)
def refresh_training_data() -> RefreshTrainingDataResponse:
    """Refresh sentiment gaps and stale fundamentals.

    brain_api resolves symbols from its ETL_UNIVERSE config.
    """
    logger = get_run_logger()
    logger.info("Refreshing training data (symbols resolved by brain_api)...")

    request = RefreshTrainingDataRequest()

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


@task(
    name="Fetch Halal New Universe",
    retries=2,
    retry_delay_seconds=30,
    persist_result=True,
    cache_policy=INPUTS,
    cache_expiration=WEEKLY_CACHE_TTL,
)
def fetch_halal_new_universe() -> dict:
    """GET /universe/halal_new -- populate cache, fail fast if scraping broken."""
    logger = get_run_logger()
    logger.info("Fetching halal_new universe (all ~410 symbols)...")

    with get_client() as client:
        response = client.get("/universe/halal_new")
        response.raise_for_status()
        data = response.json()

    total = data.get("total_stocks", len(data.get("stocks", [])))
    logger.info(f"Halal_new universe fetched: {total} stocks")
    return data


@task(
    name="Fetch Halal Filtered Universe",
    retries=1,
    retry_delay_seconds=60,
    persist_result=True,
    cache_policy=INPUTS,
    cache_expiration=WEEKLY_CACHE_TTL,
)
def fetch_halal_filtered_universe() -> dict:
    """GET /universe/halal_filtered -- triggers PatchTST inference on cache miss."""
    logger = get_run_logger()
    logger.info("Fetching halal_filtered universe (PatchTST forecast -> top 15)...")

    with get_client() as client:
        response = client.get("/universe/halal_filtered")
        response.raise_for_status()
        data = response.json()

    stocks = data.get("stocks", [])
    model_version = data.get("model_version", "unknown")
    logger.info(
        f"Halal_filtered universe fetched: {len(stocks)} stocks (model {model_version})"
    )
    return data


@task(
    name="Train LSTM",
    retries=1,
    retry_delay_seconds=120,
    persist_result=True,
    cache_policy=INPUTS,
    cache_expiration=WEEKLY_CACHE_TTL,
)
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


@task(
    name="Train PatchTST",
    retries=1,
    retry_delay_seconds=120,
    persist_result=True,
    cache_policy=INPUTS,
    cache_expiration=WEEKLY_CACHE_TTL,
)
def train_patchtst() -> TrainingResponse:
    """Train the PatchTST OHLCV forecaster model."""
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


@task(
    name="Train PPO",
    retries=1,
    retry_delay_seconds=120,
    persist_result=True,
    cache_policy=INPUTS,
    cache_expiration=WEEKLY_CACHE_TTL,
)
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


@task(
    name="Train SAC",
    retries=1,
    retry_delay_seconds=120,
    persist_result=True,
    cache_policy=INPUTS,
    cache_expiration=WEEKLY_CACHE_TTL,
)
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


@task(
    name="Generate Training Summary",
    retries=1,
    retry_delay_seconds=30,
    persist_result=True,
    cache_policy=INPUTS,
    cache_expiration=WEEKLY_CACHE_TTL,
)
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
    timeout_seconds=115200,  # 32 hours total timeout
    persist_result=True,  # Persist task results to allow resume from failure
)
def weekly_training_flow() -> dict:
    """Execute the full weekly training pipeline.

    Flow diagram:
    ```
    fetch_halal_new_universe
           │
           ▼
      train_lstm → train_patchtst
                        │
                        ▼
              fetch_halal_filtered_universe (PatchTST forecast -> top 15)
                        │
                        ▼
              refresh_training_data (signals for filtered 15)
                        │
                        ▼
                  train_ppo → train_sac
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

    # Step 1: Fetch halal_new universe (ensure ~410 symbols cached, fail fast)
    halal_new_result = fetch_halal_new_universe()

    # Steps 2-3: Train forecasters on all ~410 halal_new symbols
    lstm_result = train_lstm()
    patchtst_result = train_patchtst()

    # Step 4: Fetch halal_filtered (PatchTST forecast -> top 15)
    filtered_result = fetch_halal_filtered_universe()

    # Step 5: Refresh training data (signals for filtered 15 only)
    refresh_result = refresh_training_data()

    # Steps 6-7: Train RL allocators on filtered 15
    ppo_result = train_ppo()
    sac_result = train_sac()

    # Step 8: Generate training summary using LLM
    summary_result = generate_training_summary(
        lstm=lstm_result,
        patchtst=patchtst_result,
        ppo=ppo_result,
        sac=sac_result,
    )

    # Step 9: Send training summary email
    email_result = send_training_summary_email(
        lstm=lstm_result,
        patchtst=patchtst_result,
        ppo=ppo_result,
        sac=sac_result,
        summary=summary_result,
    )

    logger.info("Weekly training pipeline complete!")

    return {
        "halal_new": {
            "total_stocks": halal_new_result.get(
                "total_stocks", len(halal_new_result.get("stocks", []))
            ),
        },
        "lstm": {"version": lstm_result.version, "promoted": lstm_result.promoted},
        "patchtst": {
            "version": patchtst_result.version,
            "promoted": patchtst_result.promoted,
        },
        "filtered": {
            "stocks": len(filtered_result.get("stocks", [])),
            "model_version": filtered_result.get("model_version"),
            "selection_method": filtered_result.get("selection_method"),
        },
        "refresh": {
            "sentiment_gaps_filled": refresh_result.sentiment_gaps_filled,
            "fundamentals_refreshed": len(refresh_result.fundamentals_refreshed),
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

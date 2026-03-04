"""India weekly training workflow for LearnFinance-2025.

This flow runs every Sunday at 04:30 UTC (10:00 AM IST) and trains
the India PatchTST model on NiftyShariah500 stocks:

1. Fetch NiftyShariah500 universe (~210 NSE India stocks)
2. Train India PatchTST (OHLCV forecaster on all ~210 stocks)
3. Fetch halal_india universe (PatchTST forecast -> top 15)
4. Generate India training summary (LLM-powered analysis)
5. Send India training summary email
"""

import os
from datetime import timedelta

import httpx
from prefect import flow, task
from prefect.cache_policies import INPUTS
from prefect.logging import get_run_logger

from flows.models import (
    TrainingResponse,
    TrainingSummaryEmailResponse,
    TrainingSummaryResponse,
)

BRAIN_API_URL = os.environ.get("BRAIN_API_URL", "http://localhost:8000")

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
    name="Fetch NiftyShariah500 Universe",
    retries=2,
    retry_delay_seconds=30,
    persist_result=True,
    cache_policy=INPUTS,
    cache_expiration=WEEKLY_CACHE_TTL,
)
def fetch_nifty_shariah_500_universe() -> dict:
    """GET /universe/nifty_shariah_500 -- fail fast if NSE broken."""
    logger = get_run_logger()
    logger.info("Fetching NiftyShariah500 universe (all ~210 symbols)...")

    with get_client() as client:
        response = client.get("/universe/nifty_shariah_500")
        response.raise_for_status()
        data = response.json()

    total = data.get("total_stocks", len(data.get("stocks", [])))
    logger.info(f"NiftyShariah500 universe fetched: {total} stocks")
    return data


@task(
    name="Train India PatchTST",
    retries=1,
    retry_delay_seconds=120,
    persist_result=True,
    cache_policy=INPUTS,
    cache_expiration=WEEKLY_CACHE_TTL,
)
def train_india_patchtst() -> TrainingResponse:
    """Train the India PatchTST OHLCV forecaster model on NiftyShariah500."""
    logger = get_run_logger()
    logger.info("Starting India PatchTST training...")

    with get_client() as client:
        response = client.post("/train/patchtst/india")
        response.raise_for_status()
        data = response.json()

    result = TrainingResponse(**data)
    logger.info(
        f"India PatchTST training complete: version={result.version}, "
        f"promoted={result.promoted}"
    )
    return result


@task(
    name="Fetch Halal India Universe",
    retries=1,
    retry_delay_seconds=60,
    persist_result=True,
    cache_policy=INPUTS,
    cache_expiration=WEEKLY_CACHE_TTL,
)
def fetch_halal_india_universe() -> dict:
    """GET /universe/halal_india -- triggers India PatchTST inference on cache miss."""
    logger = get_run_logger()
    logger.info("Fetching halal_india universe (India PatchTST forecast -> top 15)...")

    with get_client() as client:
        response = client.get("/universe/halal_india")
        response.raise_for_status()
        data = response.json()

    stocks = data.get("stocks", [])
    model_version = data.get("model_version", "unknown")
    logger.info(
        f"Halal_india universe fetched: {len(stocks)} stocks (model {model_version})"
    )
    return data


@task(
    name="Generate India Training Summary",
    retries=1,
    retry_delay_seconds=30,
    persist_result=True,
    cache_policy=INPUTS,
    cache_expiration=WEEKLY_CACHE_TTL,
)
def generate_india_training_summary(
    patchtst: TrainingResponse,
) -> TrainingSummaryResponse:
    """Generate LLM summary of India PatchTST training results.

    Calls POST /llm/india-training-summary with PatchTST result only
    (India has no LSTM, PPO, or SAC).
    """
    logger = get_run_logger()
    logger.info("Generating India training summary via LLM...")

    payload = {
        "patchtst": {
            "version": patchtst.version,
            "data_window_start": patchtst.data_window_start,
            "data_window_end": patchtst.data_window_end,
            "metrics": patchtst.metrics,
            "promoted": patchtst.promoted,
            "num_input_channels": patchtst.num_input_channels or 0,
            "signals_used": patchtst.signals_used or [],
        },
    }

    with get_client() as client:
        response = client.post("/llm/india-training-summary", json=payload)
        response.raise_for_status()
        data = response.json()

    result = TrainingSummaryResponse(**data)
    logger.info(
        f"India training summary generated via {result.provider} ({result.model_used})"
    )
    return result


@task(name="Send India Training Summary Email", retries=1, retry_delay_seconds=30)
def send_india_training_email(
    patchtst: TrainingResponse,
    summary: TrainingSummaryResponse,
) -> TrainingSummaryEmailResponse:
    """Send India training summary via email.

    Calls POST /email/india-training-summary with PatchTST result and LLM summary.
    """
    logger = get_run_logger()
    logger.info("Sending India training summary email...")

    payload = {
        "patchtst": {
            "version": patchtst.version,
            "data_window_start": patchtst.data_window_start,
            "data_window_end": patchtst.data_window_end,
            "metrics": patchtst.metrics,
            "promoted": patchtst.promoted,
            "num_input_channels": patchtst.num_input_channels or 0,
            "signals_used": patchtst.signals_used or [],
        },
        "summary": summary.summary,
    }

    with get_client() as client:
        response = client.post("/email/india-training-summary", json=payload)
        response.raise_for_status()
        data = response.json()

    result = TrainingSummaryEmailResponse(**data)
    logger.info(
        f"India training email sent: is_success={result.is_success}, "
        f"subject={result.subject}"
    )
    return result


# =============================================================================
# Flow
# =============================================================================


@flow(
    name="India Weekly Training Pipeline",
    description="Training pipeline for India PatchTST model (NiftyShariah500)",
    retries=0,
    timeout_seconds=115200,  # 32 hours total timeout
    persist_result=True,
)
def india_weekly_training_flow() -> dict:
    """Execute the India weekly training pipeline.

    Flow diagram:
    ```
    fetch_nifty_shariah_500_universe
               |
               v
       train_india_patchtst
               |
               v
       fetch_halal_india_universe
               |
               v
    generate_india_training_summary
               |
               v
     send_india_training_email
    ```

    Returns:
        dict with training results, LLM summary, and email status
    """
    logger = get_run_logger()
    logger.info("Starting India weekly training pipeline...")

    nifty_result = fetch_nifty_shariah_500_universe()

    patchtst_result = train_india_patchtst()

    india_filtered = fetch_halal_india_universe()

    summary_result = generate_india_training_summary(patchtst=patchtst_result)

    email_result = send_india_training_email(
        patchtst=patchtst_result,
        summary=summary_result,
    )

    logger.info("India weekly training pipeline complete!")

    return {
        "nifty_shariah_500": {
            "total_stocks": nifty_result.get(
                "total_stocks", len(nifty_result.get("stocks", []))
            ),
        },
        "patchtst": {
            "version": patchtst_result.version,
            "promoted": patchtst_result.promoted,
        },
        "halal_india": {
            "stocks": len(india_filtered.get("stocks", [])),
            "model_version": india_filtered.get("model_version"),
            "selection_method": india_filtered.get("selection_method"),
        },
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
        india_weekly_training_flow()
    else:
        # Every Sunday at 04:30 UTC (10:00 AM IST): "30 4 * * 0"
        india_weekly_training_flow.serve(
            name="india-weekly-training",
            cron="30 4 * * 0",
        )

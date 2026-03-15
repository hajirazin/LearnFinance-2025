"""Training activities for US and India pipelines.

These activities have long timeouts (8+ hours) and use heartbeating
to keep the Temporal server informed during long-running training.
"""

import logging
import time

from temporalio import activity
from temporalio.exceptions import ApplicationError

from activities.client import get_training_client
from models import (
    RefreshTrainingDataRequest,
    RefreshTrainingDataResponse,
    TrainingResponse,
    TrainingSummaryEmailResponse,
    TrainingSummaryResponse,
)

logger = logging.getLogger(__name__)


@activity.defn
def refresh_training_data() -> RefreshTrainingDataResponse:
    """Refresh sentiment gaps and stale fundamentals."""
    logger.info("Refreshing training data (symbols resolved by brain_api)...")
    request = RefreshTrainingDataRequest()
    with get_training_client() as client:
        response = client.post(
            "/etl/refresh-training-data",
            json=request.model_dump(exclude_none=True),
        )
        response.raise_for_status()
    result = RefreshTrainingDataResponse(**response.json())
    logger.info(
        f"Refresh complete in {result.duration_seconds:.1f}s: "
        f"{result.sentiment_gaps_filled} sentiment gaps filled, "
        f"{len(result.fundamentals_refreshed)} fundamentals refreshed"
    )
    return result


@activity.defn
def fetch_halal_new_universe() -> dict:
    """GET /universe/halal_new -- populate cache, fail fast if scraping broken."""
    logger.info("Fetching halal_new universe (all ~410 symbols)...")
    with get_training_client() as client:
        response = client.get("/universe/halal_new")
        response.raise_for_status()
    data = response.json()
    total = data.get("total_stocks", len(data.get("stocks", [])))
    logger.info(f"Halal_new universe fetched: {total} stocks")
    return data


@activity.defn
def fetch_halal_filtered_universe() -> dict:
    """GET /universe/halal_filtered -- triggers PatchTST inference on cache miss."""
    logger.info("Fetching halal_filtered universe (PatchTST forecast -> top 15)...")
    with get_training_client() as client:
        response = client.get("/universe/halal_filtered")
        response.raise_for_status()
    data = response.json()
    stocks = data.get("stocks", [])
    model_version = data.get("model_version", "unknown")
    logger.info(
        f"Halal_filtered universe fetched: {len(stocks)} stocks (model {model_version})"
    )
    return data


@activity.defn
def fetch_nifty_shariah_500_universe() -> dict:
    """GET /universe/nifty_shariah_500 -- fail fast if NSE broken."""
    logger.info("Fetching NiftyShariah500 universe (all ~210 symbols)...")
    with get_training_client() as client:
        response = client.get("/universe/nifty_shariah_500")
        response.raise_for_status()
    data = response.json()
    total = data.get("total_stocks", len(data.get("stocks", [])))
    logger.info(f"NiftyShariah500 universe fetched: {total} stocks")
    return data


@activity.defn
def fetch_halal_india_universe() -> dict:
    """GET /universe/halal_india -- triggers India PatchTST inference on cache miss."""
    logger.info("Fetching halal_india universe (India PatchTST forecast -> top 15)...")
    with get_training_client() as client:
        response = client.get("/universe/halal_india")
        response.raise_for_status()
    data = response.json()
    stocks = data.get("stocks", [])
    model_version = data.get("model_version", "unknown")
    logger.info(
        f"Halal_india universe fetched: {len(stocks)} stocks (model {model_version})"
    )
    return data


def _poll_training_job(
    endpoint: str,
    poll_interval: float = 60.0,
) -> TrainingResponse:
    """Start a training job via POST and poll until completion.

    1. POST to endpoint: if 200, return result (idempotent cache hit)
    2. If 202, extract job_id and poll GET /train/status/{job_id}
    3. Heartbeat on each poll cycle to keep Temporal informed
    4. Return TrainingResponse on completion, raise on failure/cancel
    """
    with get_training_client() as client:
        response = client.post(endpoint)
        response.raise_for_status()

        if response.status_code == 200:
            return TrainingResponse(**response.json())

        job_data = response.json()
        job_id = job_data["job_id"]
        logger.info(f"Training job started: {job_id}")

        while True:
            activity.heartbeat(job_id)
            time.sleep(poll_interval)

            status_resp = client.get(f"/train/status/{job_id}")
            status_resp.raise_for_status()
            status = status_resp.json()

            logger.info(
                f"Job {job_id}: status={status['status']}, "
                f"progress={status.get('progress', {})}"
            )

            if status["status"] == "completed":
                return TrainingResponse(**status["result"])
            elif status["status"] in ("failed", "cancelled"):
                raise ApplicationError(
                    f"Training {status['status']}: {status.get('error', 'unknown')}"
                )


@activity.defn
def train_lstm() -> TrainingResponse:
    """Train the LSTM pure-price forecaster model."""
    logger.info("Starting LSTM training...")
    return _poll_training_job("/train/lstm")


@activity.defn
def train_patchtst() -> TrainingResponse:
    """Train the PatchTST OHLCV forecaster model."""
    logger.info("Starting PatchTST training...")
    return _poll_training_job("/train/patchtst")


@activity.defn
def train_sac() -> TrainingResponse:
    """Train the SAC reinforcement learning allocator."""
    logger.info("Starting SAC training...")
    return _poll_training_job("/train/sac/full")


@activity.defn
def train_india_patchtst() -> TrainingResponse:
    """Train the India PatchTST OHLCV forecaster model on NiftyShariah500."""
    logger.info("Starting India PatchTST training...")
    return _poll_training_job("/train/patchtst/india")


@activity.defn
def generate_training_summary(
    lstm: TrainingResponse,
    patchtst: TrainingResponse,
    sac: TrainingResponse,
) -> TrainingSummaryResponse:
    """Generate LLM summary of all training results."""
    logger.info("Generating training summary via LLM...")
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
        "sac": {
            "version": sac.version,
            "data_window_start": sac.data_window_start,
            "data_window_end": sac.data_window_end,
            "metrics": sac.metrics,
            "promoted": sac.promoted,
            "symbols_used": sac.symbols_used or [],
        },
    }
    with get_training_client() as client:
        response = client.post("/llm/training-summary", json=payload)
        response.raise_for_status()
    result = TrainingSummaryResponse(**response.json())
    logger.info(
        f"Training summary generated via {result.provider} ({result.model_used}), "
        f"tokens_used={result.tokens_used}"
    )
    return result


@activity.defn
def send_training_summary_email(
    lstm: TrainingResponse,
    patchtst: TrainingResponse,
    sac: TrainingResponse,
    summary: TrainingSummaryResponse,
) -> TrainingSummaryEmailResponse:
    """Send training summary via email."""
    logger.info("Sending training summary email...")
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
    with get_training_client() as client:
        response = client.post("/email/training-summary", json=payload)
        response.raise_for_status()
    result = TrainingSummaryEmailResponse(**response.json())
    logger.info(
        f"Training summary email sent: is_success={result.is_success}, "
        f"subject={result.subject}"
    )
    return result


@activity.defn
def generate_india_training_summary(
    patchtst: TrainingResponse,
) -> TrainingSummaryResponse:
    """Generate LLM summary of India PatchTST training results."""
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
    with get_training_client() as client:
        response = client.post("/llm/india-training-summary", json=payload)
        response.raise_for_status()
    result = TrainingSummaryResponse(**response.json())
    logger.info(
        f"India training summary generated via {result.provider} ({result.model_used})"
    )
    return result


@activity.defn
def send_india_training_email(
    patchtst: TrainingResponse,
    summary: TrainingSummaryResponse,
) -> TrainingSummaryEmailResponse:
    """Send India training summary via email."""
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
    with get_training_client() as client:
        response = client.post("/email/india-training-summary", json=payload)
        response.raise_for_status()
    result = TrainingSummaryEmailResponse(**response.json())
    logger.info(
        f"India training email sent: is_success={result.is_success}, "
        f"subject={result.subject}"
    )
    return result

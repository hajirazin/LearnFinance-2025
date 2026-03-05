"""Training activities for US and India pipelines.

These activities have long timeouts (8+ hours) and use heartbeating
to keep the Temporal server informed during long-running training.
"""

import logging

from temporalio import activity

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


@activity.defn
def train_lstm() -> TrainingResponse:
    """Train the LSTM pure-price forecaster model."""
    logger.info("Starting LSTM training...")
    activity.heartbeat()
    with get_training_client() as client:
        response = client.post("/train/lstm")
        response.raise_for_status()
    result = TrainingResponse(**response.json())
    logger.info(
        f"LSTM training complete: version={result.version}, promoted={result.promoted}"
    )
    return result


@activity.defn
def train_patchtst() -> TrainingResponse:
    """Train the PatchTST OHLCV forecaster model."""
    logger.info("Starting PatchTST training...")
    activity.heartbeat()
    with get_training_client() as client:
        response = client.post("/train/patchtst")
        response.raise_for_status()
    result = TrainingResponse(**response.json())
    logger.info(
        f"PatchTST training complete: version={result.version}, "
        f"promoted={result.promoted}"
    )
    return result


@activity.defn
def train_ppo() -> TrainingResponse:
    """Train the PPO reinforcement learning allocator."""
    logger.info("Starting PPO training...")
    activity.heartbeat()
    with get_training_client() as client:
        response = client.post("/train/ppo/full")
        response.raise_for_status()
    result = TrainingResponse(**response.json())
    logger.info(
        f"PPO training complete: version={result.version}, promoted={result.promoted}"
    )
    return result


@activity.defn
def train_sac() -> TrainingResponse:
    """Train the SAC reinforcement learning allocator."""
    logger.info("Starting SAC training...")
    activity.heartbeat()
    with get_training_client() as client:
        response = client.post("/train/sac/full")
        response.raise_for_status()
    result = TrainingResponse(**response.json())
    logger.info(
        f"SAC training complete: version={result.version}, promoted={result.promoted}"
    )
    return result


@activity.defn
def train_india_patchtst() -> TrainingResponse:
    """Train the India PatchTST OHLCV forecaster model on NiftyShariah500."""
    logger.info("Starting India PatchTST training...")
    activity.heartbeat()
    with get_training_client() as client:
        response = client.post("/train/patchtst/india")
        response.raise_for_status()
    result = TrainingResponse(**response.json())
    logger.info(
        f"India PatchTST training complete: version={result.version}, "
        f"promoted={result.promoted}"
    )
    return result


@activity.defn
def generate_training_summary(
    lstm: TrainingResponse,
    patchtst: TrainingResponse,
    ppo: TrainingResponse,
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
    ppo: TrainingResponse,
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

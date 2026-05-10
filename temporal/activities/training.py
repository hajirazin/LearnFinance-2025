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
def refresh_training_data(universe: str) -> RefreshTrainingDataResponse:
    """Refresh sentiment gaps and stale fundamentals for ``universe``.

    ``universe`` selects the registered ETL universe at the brain_api
    side so two parallel SAC workflows (e.g. ``halal_filtered`` and a
    future ``halal``) can each refresh their own slate.
    """
    logger.info("Refreshing training data for universe=%s ...", universe)
    request = RefreshTrainingDataRequest(universe=universe)
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
def fetch_halal_universe() -> dict:
    """GET /universe/halal -- legacy yfinance ETF top-holdings.

    The ``halal`` universe is the parallel A/B sibling of
    ``halal_filtered`` for the SAC bucket comparison. yfinance's
    ETF top-holdings can fluctuate month-to-month, so the slate size
    is variable (typical range 12-15 after dedup + US filter); SAC's
    actor/critic dim is resized at training time via the bucket-level
    config factory in brain_api.
    """
    logger.info("Fetching halal universe (yfinance ETF top-holdings)...")
    with get_training_client() as client:
        response = client.get("/universe/halal")
        response.raise_for_status()
    data = response.json()
    stocks = data.get("stocks", [])
    total = data.get("total_stocks", len(stocks))
    logger.info(f"Halal universe fetched: {total} stocks")
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
    json_body: dict | None = None,
    *,
    params: dict[str, str] | None = None,
    poll_interval: float = 60.0,
) -> TrainingResponse:
    """Start a training job via POST and poll until completion.

    1. POST to endpoint with ``json_body`` (carries the ``universe``
       selector for universe-keyed buckets): if 200, return result
       (idempotent cache hit)
    2. If 202, extract job_id and poll GET /train/status/{job_id}
    3. Heartbeat on each poll cycle to keep Temporal informed
    4. Return TrainingResponse on completion, raise on failure/cancel

    ``params`` are forwarded as query parameters on the initial POST
    (e.g. ``skip_snapshot`` for India PatchTST).
    """
    with get_training_client() as client:
        response = client.post(endpoint, json=json_body, params=params)
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
def train_lstm(universe: str) -> TrainingResponse:
    """Train the LSTM pure-price forecaster model on the given universe.

    ``universe`` selects the universe-keyed bucket (e.g. ``halal_new``)
    so that two parallel workflows can hit the same endpoint with
    different universes without colliding on storage.
    """
    logger.info(f"Starting LSTM training on universe={universe}...")
    return _poll_training_job("/train/lstm", json_body={"universe": universe})


@activity.defn
def train_patchtst(universe: str) -> TrainingResponse:
    """Train the US PatchTST OHLCV forecaster on the given universe."""
    logger.info(f"Starting PatchTST training on universe={universe}...")
    return _poll_training_job("/train/patchtst", json_body={"universe": universe})


@activity.defn
def train_sac(universe: str) -> TrainingResponse:
    """Train the SAC reinforcement-learning allocator on the given universe.

    Two registered SAC buckets accept this call:
    ``halal_filtered`` (slate fixed at 15 by the bucket validator) and
    ``halal`` (variable-size yfinance ETF top-holdings; SAC's actor /
    critic dim is resized at training time by the bucket-level config
    factory in brain_api). The API enforces per-bucket symbol-count
    invariants and returns 422 if violated.
    """
    logger.info(f"Starting SAC training on universe={universe}...")
    return _poll_training_job("/train/sac/full", json_body={"universe": universe})


@activity.defn
def train_india_patchtst(universe: str) -> TrainingResponse:
    """Train the India PatchTST OHLCV forecaster on the given universe.

    Snapshots are skipped: India PatchTST has no SAC consumer that
    needs walk-forward snapshots, so producing them only wastes time
    and disk. (US PatchTST keeps snapshots for SAC training.)
    """
    logger.info(f"Starting India PatchTST training on universe={universe}...")
    return _poll_training_job(
        "/train/patchtst/india",
        json_body={"universe": universe},
        params={"skip_snapshot": "true"},
    )


def _lstm_payload(lstm: TrainingResponse) -> dict:
    """Serialise an LSTM ``TrainingResponse`` for brain_api JSON bodies.

    Mirrors the ``LSTMTrainResponse`` Pydantic schema so the same dict
    is reusable across the ``/llm/forecasters-training-summary`` and
    ``/email/forecasters-training-summary`` payloads.
    """
    return {
        "version": lstm.version,
        "data_window_start": lstm.data_window_start,
        "data_window_end": lstm.data_window_end,
        "metrics": lstm.metrics,
        "promoted": lstm.promoted,
        "failure_reasons": lstm.failure_reasons,
    }


def _patchtst_payload(patchtst: TrainingResponse) -> dict:
    """Serialise a PatchTST ``TrainingResponse`` for brain_api JSON bodies.

    Mirrors the ``PatchTSTTrainResponse`` Pydantic schema so the same
    dict is reusable across forecaster summary + email payloads (US and
    India).
    """
    return {
        "version": patchtst.version,
        "data_window_start": patchtst.data_window_start,
        "data_window_end": patchtst.data_window_end,
        "metrics": patchtst.metrics,
        "promoted": patchtst.promoted,
        "failure_reasons": patchtst.failure_reasons,
        "num_input_channels": patchtst.num_input_channels or 0,
        "signals_used": patchtst.signals_used or [],
    }


def _sac_payload(sac: TrainingResponse) -> dict:
    """Serialise a SAC ``TrainingResponse`` for brain_api JSON bodies.

    Mirrors the ``SACTrainResponse`` Pydantic schema for the
    ``/llm/sac-training-summary`` and ``/email/sac-training-summary``
    payloads.
    """
    return {
        "version": sac.version,
        "data_window_start": sac.data_window_start,
        "data_window_end": sac.data_window_end,
        "metrics": sac.metrics,
        "promoted": sac.promoted,
        "failure_reasons": sac.failure_reasons,
        "symbols_used": sac.symbols_used or [],
    }


@activity.defn
def generate_forecasters_training_summary(
    lstm: TrainingResponse,
    patchtst: TrainingResponse,
) -> TrainingSummaryResponse:
    """Generate LLM summary for the US LSTM + PatchTST training run.

    Called by ``USForecastersTrainingWorkflow`` after both forecasters
    finish training serially. SAC has its own summary endpoint and runs
    on a separate workflow, so it is intentionally not included here.
    """
    logger.info("Generating forecasters (LSTM + PatchTST) training summary via LLM...")
    payload = {
        "lstm": _lstm_payload(lstm),
        "patchtst": _patchtst_payload(patchtst),
    }
    with get_training_client() as client:
        response = client.post("/llm/forecasters-training-summary", json=payload)
        response.raise_for_status()
    result = TrainingSummaryResponse(**response.json())
    logger.info(
        f"Forecasters training summary generated via {result.provider} "
        f"({result.model_used}), tokens_used={result.tokens_used}"
    )
    return result


@activity.defn
def send_forecasters_training_email(
    lstm: TrainingResponse,
    patchtst: TrainingResponse,
    summary: TrainingSummaryResponse,
) -> TrainingSummaryEmailResponse:
    """Send the US Forecasters (LSTM + PatchTST) training summary email."""
    logger.info("Sending forecasters training summary email...")
    payload = {
        "lstm": _lstm_payload(lstm),
        "patchtst": _patchtst_payload(patchtst),
        "summary": summary.summary,
    }
    with get_training_client() as client:
        response = client.post("/email/forecasters-training-summary", json=payload)
        response.raise_for_status()
    result = TrainingSummaryEmailResponse(**response.json())
    logger.info(
        f"Forecasters training summary email sent: is_success={result.is_success}, "
        f"subject={result.subject}"
    )
    return result


@activity.defn
def generate_sac_training_summary(
    sac: TrainingResponse,
    universe: str,
) -> TrainingSummaryResponse:
    """Generate LLM summary for a US SAC training run.

    Called by either US SAC workflow:
    ``USSACTrainingWorkflow`` (universe=``halal_filtered``) and the
    parallel A/B sibling ``USSACHalalTrainingWorkflow``
    (universe=``halal``). SAC consumes whatever PatchTST ``current``
    pointer is live at trigger time, so forecaster metrics are
    summarised separately by ``USForecastersTrainingWorkflow``.

    The ``universe`` argument is forwarded to brain_api so the
    resulting summary identifies which bucket the metrics describe.
    It is required (no default) so every workflow call site sends a
    matching arg count -- Temporal's activity decoder silently drops
    Pydantic type hints when the workflow's positional-arg count
    differs from the activity signature, which would turn
    ``sac: TrainingResponse`` into a plain ``dict`` and break
    ``_sac_payload``.
    """
    logger.info(f"Generating SAC training summary via LLM (universe={universe})...")
    payload = {"sac": _sac_payload(sac), "universe": universe}
    with get_training_client() as client:
        response = client.post("/llm/sac-training-summary", json=payload)
        response.raise_for_status()
    result = TrainingSummaryResponse(**response.json())
    logger.info(
        f"SAC training summary generated via {result.provider} "
        f"({result.model_used}), tokens_used={result.tokens_used}"
    )
    return result


@activity.defn
def send_sac_training_email(
    sac: TrainingResponse,
    summary: TrainingSummaryResponse,
    universe: str,
) -> TrainingSummaryEmailResponse:
    """Send the US SAC training summary email.

    Shared by both SAC workflows. ``universe`` is forwarded to
    brain_api so the email subject is bucket-specific (e.g.
    "US SAC (halal) Training: ..."), letting a human inbox reader
    distinguish the two parallel reports without opening them. It is
    required (no default) for the same reason as
    ``generate_sac_training_summary`` -- arg-count mismatch silently
    strips Pydantic type hints from the activity decoder.
    """
    logger.info(f"Sending SAC training summary email (universe={universe})...")
    payload = {
        "sac": _sac_payload(sac),
        "summary": summary.summary,
        "universe": universe,
    }
    with get_training_client() as client:
        response = client.post("/email/sac-training-summary", json=payload)
        response.raise_for_status()
    result = TrainingSummaryEmailResponse(**response.json())
    logger.info(
        f"SAC training summary email sent: is_success={result.is_success}, "
        f"subject={result.subject}"
    )
    return result


@activity.defn
def generate_india_training_summary(
    patchtst: TrainingResponse,
) -> TrainingSummaryResponse:
    """Generate LLM summary of India PatchTST training results."""
    logger.info("Generating India training summary via LLM...")
    payload = {"patchtst": _patchtst_payload(patchtst)}
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
        "patchtst": _patchtst_payload(patchtst),
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

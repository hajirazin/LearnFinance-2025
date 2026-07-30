"""Models for training endpoints."""

from typing import Any

from pydantic import BaseModel


class TrainingResponse(BaseModel):
    """Common response model for training endpoints.

    Works for LSTM, PatchTST, and SAC training responses.
    Uses flexible types to accommodate different metric structures.
    """

    version: str
    data_window_start: str
    data_window_end: str
    metrics: dict[str, Any]
    promoted: bool
    prior_version: str | None = None
    # Empty when ``promoted`` is True; otherwise lists every guardrail
    # the new artifact failed (finite metrics, files-on-disk, SAC CAGR
    # floor, SAC finetune symbol-order). Threaded through the LLM
    # prompt + email template so the operator sees what to investigate.
    failure_reasons: list[str] = []
    # Optional fields that vary by model type
    hf_repo: str | None = None
    hf_url: str | None = None
    symbols_used: list[str] | None = None
    num_input_channels: int | None = None
    signals_used: list[str] | None = None


class TrainingJobResponse(BaseModel):
    """202 response when a training job is started or already running."""

    job_id: str
    status: str
    message: str


class SACTrainingWorkflowInput(BaseModel):
    """Operator inputs for a scheduled or manually started SAC training run."""

    force: bool = False


class SACReadinessIssue(BaseModel):
    """Exact condition preventing SAC training from starting."""

    source: str
    detail: str
    symbol: str | None = None
    retryable: bool


class SACTrainingReadiness(BaseModel):
    """Result of the Brain-owned SAC training preflight."""

    universe: str
    symbols: list[str]
    ready: bool
    missing: list[SACReadinessIssue] = []
    errors: list[SACReadinessIssue] = []

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

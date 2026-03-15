"""Shared response models for training endpoints."""

from typing import Any

from pydantic import BaseModel


class TrainingJobResponse(BaseModel):
    """202 response when a training job is started or already running."""

    job_id: str
    status: str
    message: str


class TrainingJobStatusResponse(BaseModel):
    """Response from GET /train/status/{job_id}."""

    job_id: str
    model_type: str
    status: str
    started_at: str
    completed_at: str | None = None
    progress: dict[str, Any] = {}
    error: str | None = None
    result: dict[str, Any] | None = None


class LSTMTrainResponse(BaseModel):
    """Response model for LSTM training endpoint."""

    version: str
    data_window_start: str  # YYYY-MM-DD
    data_window_end: str  # YYYY-MM-DD
    metrics: dict[str, Any]
    promoted: bool
    prior_version: str | None = None
    hf_repo: str | None = None  # HuggingFace repo if uploaded
    hf_url: str | None = None  # URL to model on HuggingFace


class PatchTSTTrainResponse(BaseModel):
    """Response model for PatchTST training endpoint."""

    version: str
    data_window_start: str  # YYYY-MM-DD
    data_window_end: str  # YYYY-MM-DD
    metrics: dict[str, Any]
    promoted: bool
    prior_version: str | None = None
    hf_repo: str | None = None  # HuggingFace repo if uploaded
    hf_url: str | None = None  # URL to model on HuggingFace
    # PatchTST-specific fields
    num_input_channels: int  # Number of feature channels used
    signals_used: list[str]  # List of signal types included


class SACTrainResponse(BaseModel):
    """Response model for SAC training endpoint (unified with dual forecasts)."""

    version: str
    data_window_start: str
    data_window_end: str
    metrics: dict[str, float]
    promoted: bool
    prior_version: str | None = None
    symbols_used: list[str]
    hf_repo: str | None = None
    hf_url: str | None = None

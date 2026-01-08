"""Shared response models for training endpoints."""

from typing import Any

from pydantic import BaseModel


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


class PPOLSTMTrainResponse(BaseModel):
    """Response model for PPO + LSTM training endpoint."""

    version: str
    data_window_start: str  # YYYY-MM-DD
    data_window_end: str  # YYYY-MM-DD
    metrics: dict[str, Any]
    promoted: bool
    prior_version: str | None = None
    symbols_used: list[str]
    hf_repo: str | None = None  # HuggingFace repo if uploaded
    hf_url: str | None = None  # URL to model on HuggingFace


class PPOPatchTSTTrainResponse(BaseModel):
    """Response model for PPO + PatchTST training endpoint."""

    version: str
    data_window_start: str  # YYYY-MM-DD
    data_window_end: str  # YYYY-MM-DD
    metrics: dict[str, Any]
    promoted: bool
    prior_version: str | None = None
    symbols_used: list[str]
    hf_repo: str | None = None  # HuggingFace repo if uploaded
    hf_url: str | None = None  # URL to model on HuggingFace


class SACLSTMTrainResponse(BaseModel):
    """Response model for SAC + LSTM training endpoint."""

    version: str
    data_window_start: str
    data_window_end: str
    metrics: dict[str, float]
    promoted: bool
    prior_version: str | None
    symbols_used: list[str]
    hf_repo: str | None = None
    hf_url: str | None = None


class SACPatchTSTTrainResponse(BaseModel):
    """Response model for SAC + PatchTST training endpoint."""

    version: str
    data_window_start: str
    data_window_end: str
    metrics: dict[str, float]
    promoted: bool
    prior_version: str | None
    symbols_used: list[str]
    hf_repo: str | None = None
    hf_url: str | None = None


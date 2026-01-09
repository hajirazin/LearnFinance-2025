"""Shared type definitions and protocols for model components.

This module defines common interfaces and type aliases used across
different model implementations (LSTM, PatchTST, etc.).
"""

from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol, TypeVar

import numpy as np
from sklearn.preprocessing import StandardScaler

# Type variable for model types
ModelT = TypeVar("ModelT")
ConfigT = TypeVar("ConfigT")


class ModelConfig(Protocol):
    """Protocol for model configuration classes."""

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for serialization."""
        ...


@dataclass
class BaseInferenceFeatures:
    """Base features prepared for inference for a single symbol.

    Model-specific implementations may add additional fields.
    """

    symbol: str
    features: np.ndarray | None  # Shape varies by model, None if insufficient data
    has_enough_history: bool
    history_days_used: int
    data_end_date: date | None


@dataclass
class BasePrediction:
    """Base prediction result for a single symbol.

    Model-specific implementations may add additional fields.
    """

    symbol: str
    predicted_weekly_return_pct: float | None  # Percentage, e.g., 2.5 for +2.5%
    direction: str  # "UP", "DOWN", or "FLAT"
    has_enough_history: bool
    history_days_used: int
    data_end_date: str | None  # ISO format
    target_week_start: str  # ISO format
    target_week_end: str  # ISO format


@dataclass
class BaseDatasetResult:
    """Base result of dataset building for weekly return prediction."""

    X: np.ndarray  # Input sequences, shape varies by model
    y: np.ndarray  # Targets (weekly returns)
    feature_scaler: StandardScaler


@dataclass
class BaseTrainingResult:
    """Base result of model training."""

    train_loss: float
    val_loss: float
    baseline_loss: float  # Baseline: predict 0 return (no change)


def classify_direction(weekly_return: float, threshold: float = 0.001) -> str:
    """Classify predicted return into direction.

    Args:
        weekly_return: Predicted weekly return (decimal, e.g., 0.025 for 2.5%)
        threshold: Minimum absolute return for UP/DOWN classification

    Returns:
        "UP" if return > threshold
        "DOWN" if return < -threshold
        "FLAT" otherwise
    """
    if weekly_return > threshold:
        return "UP"
    elif weekly_return < -threshold:
        return "DOWN"
    return "FLAT"


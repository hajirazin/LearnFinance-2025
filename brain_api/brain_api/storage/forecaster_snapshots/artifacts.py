"""Dataclasses for loaded forecaster snapshot artifacts."""

from dataclasses import dataclass
from datetime import date
from typing import Any

from sklearn.preprocessing import StandardScaler


@dataclass
class LSTMSnapshotArtifacts:
    """Loaded LSTM snapshot artifacts for inference.

    Contains everything needed to run inference:
    - config: model hyperparameters
    - feature_scaler: fitted StandardScaler for input normalization
    - model: PyTorch LSTM model with loaded weights
    - cutoff_date: the data cutoff date for this snapshot
    """

    config: Any  # LSTMConfig
    feature_scaler: StandardScaler
    model: Any  # LSTMModel
    cutoff_date: date


@dataclass
class PatchTSTSnapshotArtifacts:
    """Loaded PatchTST snapshot artifacts for inference.

    Contains everything needed to run inference:
    - config: model hyperparameters
    - feature_scaler: fitted StandardScaler for input normalization
    - model: HuggingFace PatchTSTForPrediction model
    - cutoff_date: the data cutoff date for this snapshot
    """

    config: Any  # PatchTSTConfig
    feature_scaler: StandardScaler
    model: Any  # PatchTSTForPrediction
    cutoff_date: date

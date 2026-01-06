"""Local filesystem storage for LSTM model artifacts."""

from dataclasses import dataclass
from typing import Any

from sklearn.preprocessing import StandardScaler

from brain_api.core.lstm import LSTMConfig, LSTMModel
from brain_api.storage.base_local import BaseLocalModelStorage


@dataclass
class LSTMArtifacts:
    """Loaded LSTM model artifacts for inference.

    Contains everything needed to run inference:
    - config: model hyperparameters (sequence_length, input_size, etc.)
    - feature_scaler: fitted StandardScaler for input normalization
    - model: PyTorch LSTM model with loaded weights
    - version: the version string these artifacts came from
    """

    config: LSTMConfig
    feature_scaler: StandardScaler
    model: LSTMModel
    version: str


class LocalModelStorage(
    BaseLocalModelStorage[LSTMConfig, LSTMModel, LSTMArtifacts]
):
    """Local filesystem storage for LSTM model artifacts.

    Artifacts are stored under:
        {base_path}/models/lstm/{version}/
            - weights.pt            (PyTorch model weights)
            - feature_scaler.pkl    (sklearn StandardScaler for input features)
            - config.json           (model hyperparameters)
            - metadata.json         (training info, metrics, data window)

    The current version pointer is stored at:
        {base_path}/models/lstm/current

    Note: The model predicts weekly returns directly, so no price_scaler
    is needed for denormalization.
    """

    @property
    def model_type(self) -> str:
        return "lstm"

    def _load_config(self, config_dict: dict[str, Any]) -> LSTMConfig:
        return LSTMConfig(**config_dict)

    def _create_model(self, config: LSTMConfig) -> LSTMModel:
        return LSTMModel(config)

    def _create_artifacts(
        self,
        config: LSTMConfig,
        feature_scaler: StandardScaler,
        model: LSTMModel,
        version: str,
    ) -> LSTMArtifacts:
        return LSTMArtifacts(
            config=config,
            feature_scaler=feature_scaler,
            model=model,
            version=version,
        )

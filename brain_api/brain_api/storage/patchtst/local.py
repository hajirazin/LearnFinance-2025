"""Local filesystem storage for PatchTST model artifacts."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sklearn.preprocessing import StandardScaler

from brain_api.storage.base_local import BaseLocalModelStorage

if TYPE_CHECKING:
    from transformers import PatchTSTForPrediction

    from brain_api.core.patchtst import PatchTSTConfig


@dataclass
class PatchTSTArtifacts:
    """Loaded PatchTST model artifacts for inference.

    Contains everything needed to run inference:
    - config: model hyperparameters
    - feature_scaler: fitted StandardScaler for input normalization
    - model: HuggingFace PatchTSTForPrediction model with loaded weights
    - version: the version string these artifacts came from
    """

    config: "PatchTSTConfig"
    feature_scaler: StandardScaler
    model: "PatchTSTForPrediction"
    version: str


class PatchTSTModelStorage(
    BaseLocalModelStorage["PatchTSTConfig", "PatchTSTForPrediction", PatchTSTArtifacts]
):
    """Local filesystem storage for PatchTST model artifacts.

    Artifacts are stored under:
        {base_path}/models/patchtst/{version}/
            - weights.pt            (PyTorch model weights)
            - feature_scaler.pkl    (sklearn StandardScaler for input features)
            - config.json           (model hyperparameters)
            - metadata.json         (training info, metrics, data window)

    The current version pointer is stored at:
        {base_path}/models/patchtst/current
    """

    @property
    def model_type(self) -> str:
        return "patchtst"

    def _load_config(self, config_dict: dict[str, Any]) -> "PatchTSTConfig":
        from brain_api.core.patchtst import PatchTSTConfig

        return PatchTSTConfig(**config_dict)

    def _create_model(self, config: "PatchTSTConfig") -> "PatchTSTForPrediction":
        from transformers import PatchTSTForPrediction

        hf_config = config.to_hf_config()
        return PatchTSTForPrediction(hf_config)

    def _create_artifacts(
        self,
        config: "PatchTSTConfig",
        feature_scaler: StandardScaler,
        model: "PatchTSTForPrediction",
        version: str,
    ) -> PatchTSTArtifacts:
        return PatchTSTArtifacts(
            config=config,
            feature_scaler=feature_scaler,
            model=model,
            version=version,
        )

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


class PatchTSTHalalNewModelStorage(
    BaseLocalModelStorage["PatchTSTConfig", "PatchTSTForPrediction", PatchTSTArtifacts]
):
    """Local filesystem storage for PatchTST trained on the halal_new universe.

    Artifacts are stored under:
        {base_path}/models/patchtst_halal_new/{version}/
            - weights.pt            (PyTorch model weights)
            - feature_scaler.pkl    (sklearn StandardScaler for input features)
            - config.json           (model hyperparameters)
            - metadata.json         (training info, metrics, data window)

    The current version pointer is stored at:
        {base_path}/models/patchtst_halal_new/current

    The bucket name encodes ``(model, universe)``; adding a new
    PatchTST universe (e.g. ``halal_filtered`` for an A/B comparison)
    means a new sibling subclass with its own ``model_type``. See
    ``brain_api.core.model_buckets``.
    """

    @property
    def model_type(self) -> str:
        return "patchtst_halal_new"

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


class PatchTSTNiftyShariah500ModelStorage(PatchTSTHalalNewModelStorage):
    """PatchTST storage for the Indian ``nifty_shariah_500`` universe.

    Artifacts live under ``data/models/patchtst_nifty_shariah_500/``
    with an independent ``current`` pointer. India PatchTST is trained
    on ~210 ``.NS``-suffixed Nifty 500 Shariah constituents (the broad
    forecaster universe), distinct from the sticky-15 ``halal_india``
    bucket that future India SAC training will use.
    """

    @property
    def model_type(self) -> str:
        return "patchtst_nifty_shariah_500"


# Backward compatibility aliases. Existing callers that pre-date the
# {model}_{universe} naming continue to work; new code should use the
# explicit names so the bucket identity is visible at the import site.
PatchTSTModelStorage = PatchTSTHalalNewModelStorage
PatchTSTIndiaModelStorage = PatchTSTNiftyShariah500ModelStorage

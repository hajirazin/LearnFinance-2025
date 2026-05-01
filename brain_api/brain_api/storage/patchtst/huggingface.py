"""HuggingFace Hub storage for PatchTST model artifacts."""

from typing import TYPE_CHECKING, Any

from brain_api.core.config import (
    get_hf_patchtst_halal_new_model_repo,
    get_hf_patchtst_nifty_shariah_500_model_repo,
)
from brain_api.storage.base_huggingface import BaseHuggingFaceModelStorage, HFModelInfo
from brain_api.storage.patchtst.local import (
    PatchTSTArtifacts,
    PatchTSTHalalNewModelStorage,
    PatchTSTNiftyShariah500ModelStorage,
)

if TYPE_CHECKING:
    from transformers import PatchTSTForPrediction

    from brain_api.core.patchtst import PatchTSTConfig

# Re-export HFModelInfo for backward compatibility
__all__ = [
    "HFModelInfo",
    "PatchTSTHalalNewHuggingFaceModelStorage",
    "PatchTSTHuggingFaceModelStorage",
    "PatchTSTIndiaHuggingFaceModelStorage",
    "PatchTSTNiftyShariah500HuggingFaceModelStorage",
]


class PatchTSTHalalNewHuggingFaceModelStorage(
    BaseHuggingFaceModelStorage[
        "PatchTSTConfig",
        "PatchTSTForPrediction",
        PatchTSTArtifacts,
        PatchTSTHalalNewModelStorage,
    ]
):
    """HuggingFace Hub storage for PatchTST trained on the halal_new universe.

    Stores model artifacts as files in a HuggingFace Model repository:
        - weights.pt            (PyTorch model weights)
        - feature_scaler.pkl    (sklearn StandardScaler for input features)
        - config.json           (model hyperparameters)
        - metadata.json         (training info, metrics, data window)

    Versions are managed as git tags/branches on the HF repo.
    The 'main' branch typically points to the current promoted version.
    """

    def __init__(
        self,
        repo_id: str | None = None,
        token: str | None = None,
        local_cache: PatchTSTHalalNewModelStorage | None = None,
    ):
        """Initialize PatchTST halal_new HuggingFace storage.

        Args:
            repo_id: HuggingFace repo ID. Defaults to
                ``HF_PATCHTST_HALAL_NEW_MODEL_REPO`` env var.
            token: HuggingFace API token.
            local_cache: Optional local storage for caching downloads.
        """
        if repo_id is None:
            repo_id = get_hf_patchtst_halal_new_model_repo()
        super().__init__(repo_id=repo_id, token=token, local_cache=local_cache)

    @property
    def model_type(self) -> str:
        return "patchtst_halal_new"

    def _create_local_storage(self) -> PatchTSTHalalNewModelStorage:
        return PatchTSTHalalNewModelStorage()

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
        feature_scaler: Any,
        model: "PatchTSTForPrediction",
        version: str,
    ) -> PatchTSTArtifacts:
        return PatchTSTArtifacts(
            config=config,
            feature_scaler=feature_scaler,
            model=model,
            version=version,
        )

    def _generate_readme(self, version: str, metadata: dict[str, Any]) -> str:
        return f"""---
tags:
- patchtst
- transformer
- finance
- weekly-returns
- learnfinance
- time-series
---

# LearnFinance PatchTST Model - {version}

OHLCV 5-channel PatchTST transformer model for predicting weekly stock returns.

## Model Details

- **Version**: {version}
- **Model Type**: PatchTST (Patch Time Series Transformer)
- **Training Window**: {metadata.get("data_window", {}).get("start", "N/A")} to {metadata.get("data_window", {}).get("end", "N/A")}
- **Symbols**: {len(metadata.get("symbols", []))} stocks

## Input Channels (11 total)

- OHLCV log returns (5): open, high, low, close, volume
- News sentiment (1)
- Fundamentals (5): gross_margin, operating_margin, net_margin, current_ratio, debt_to_equity

## Metrics

- Train Loss: {metadata.get("metrics", {}).get("train_loss", "N/A")}
- Validation Loss: {metadata.get("metrics", {}).get("val_loss", "N/A")}
- Baseline Loss: {metadata.get("metrics", {}).get("baseline_loss", "N/A")}

## Usage

```python
from brain_api.storage.patchtst.huggingface import PatchTSTHalalNewHuggingFaceModelStorage
from brain_api.storage.patchtst.local import PatchTSTHalalNewModelStorage

storage = PatchTSTHalalNewHuggingFaceModelStorage(
    repo_id="{self.repo_id}",
    local_cache=PatchTSTHalalNewModelStorage(),
)
artifacts = storage.download_model(version="{version}")
```
"""


class PatchTSTNiftyShariah500HuggingFaceModelStorage(
    PatchTSTHalalNewHuggingFaceModelStorage,
):
    """HuggingFace Hub storage for India PatchTST models.

    Uses the India-specific HF repo
    (``HF_PATCHTST_NIFTY_SHARIAH_500_MODEL_REPO``) and the
    ``data/models/patchtst_nifty_shariah_500/`` local cache directory.
    """

    def __init__(
        self,
        repo_id: str | None = None,
        token: str | None = None,
        local_cache: PatchTSTNiftyShariah500ModelStorage | None = None,
    ):
        if repo_id is None:
            repo_id = get_hf_patchtst_nifty_shariah_500_model_repo()
        if local_cache is None:
            local_cache = PatchTSTNiftyShariah500ModelStorage()
        super().__init__(repo_id=repo_id, token=token, local_cache=local_cache)

    @property
    def model_type(self) -> str:
        return "patchtst_nifty_shariah_500"

    def _create_local_storage(self) -> PatchTSTNiftyShariah500ModelStorage:
        return PatchTSTNiftyShariah500ModelStorage()


# Backward compatibility aliases. Existing callers and the snapshot
# subsystem continue to import the old names while we migrate.
PatchTSTHuggingFaceModelStorage = PatchTSTHalalNewHuggingFaceModelStorage
PatchTSTIndiaHuggingFaceModelStorage = PatchTSTNiftyShariah500HuggingFaceModelStorage

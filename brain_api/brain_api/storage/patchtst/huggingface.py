"""HuggingFace Hub storage for PatchTST model artifacts."""

from typing import TYPE_CHECKING, Any

from brain_api.storage.base_huggingface import BaseHuggingFaceModelStorage, HFModelInfo
from brain_api.storage.patchtst.local import PatchTSTArtifacts, PatchTSTModelStorage

if TYPE_CHECKING:
    from transformers import PatchTSTForPrediction

    from brain_api.core.patchtst import PatchTSTConfig

# Re-export HFModelInfo for backward compatibility
__all__ = ["PatchTSTHuggingFaceModelStorage", "HFModelInfo"]


class PatchTSTHuggingFaceModelStorage(
    BaseHuggingFaceModelStorage[
        "PatchTSTConfig", "PatchTSTForPrediction", PatchTSTArtifacts, PatchTSTModelStorage
    ]
):
    """HuggingFace Hub storage for PatchTST model artifacts.

    Stores model artifacts as files in a HuggingFace Model repository:
        - weights.pt            (PyTorch model weights)
        - feature_scaler.pkl    (sklearn StandardScaler for input features)
        - config.json           (model hyperparameters)
        - metadata.json         (training info, metrics, data window)

    Versions are managed as git tags/branches on the HF repo.
    The 'main' branch typically points to the current promoted version.
    """

    @property
    def model_type(self) -> str:
        return "patchtst"

    def _create_local_storage(self) -> PatchTSTModelStorage:
        return PatchTSTModelStorage()

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

Multi-signal PatchTST transformer model for predicting weekly stock returns.

## Model Details

- **Version**: {version}
- **Model Type**: PatchTST (Patch Time Series Transformer)
- **Training Window**: {metadata.get('data_window', {}).get('start', 'N/A')} to {metadata.get('data_window', {}).get('end', 'N/A')}
- **Symbols**: {len(metadata.get('symbols', []))} stocks

## Input Channels (11 total)

- OHLCV log returns (5): open, high, low, close, volume
- News sentiment (1)
- Fundamentals (5): gross_margin, operating_margin, net_margin, current_ratio, debt_to_equity

## Metrics

- Train Loss: {metadata.get('metrics', {}).get('train_loss', 'N/A')}
- Validation Loss: {metadata.get('metrics', {}).get('val_loss', 'N/A')}
- Baseline Loss: {metadata.get('metrics', {}).get('baseline_loss', 'N/A')}

## Usage

```python
from brain_api.storage.huggingface import PatchTSTHuggingFaceModelStorage

storage = PatchTSTHuggingFaceModelStorage(repo_id="{self.repo_id}")
artifacts = storage.download_model(version="{version}")
```
"""

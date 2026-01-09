"""HuggingFace Hub storage for LSTM model artifacts."""

from typing import Any

from brain_api.core.lstm import LSTMConfig, LSTMModel
from brain_api.storage.base_huggingface import BaseHuggingFaceModelStorage, HFModelInfo
from brain_api.storage.lstm.local import LocalModelStorage, LSTMArtifacts

# Re-export HFModelInfo for backward compatibility
__all__ = ["HFModelInfo", "HuggingFaceModelStorage"]


class HuggingFaceModelStorage(
    BaseHuggingFaceModelStorage[LSTMConfig, LSTMModel, LSTMArtifacts, LocalModelStorage]
):
    """HuggingFace Hub storage for LSTM model artifacts.

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
        return "lstm"

    def _create_local_storage(self) -> LocalModelStorage:
        return LocalModelStorage()

    def _load_config(self, config_dict: dict[str, Any]) -> LSTMConfig:
        return LSTMConfig(**config_dict)

    def _create_model(self, config: LSTMConfig) -> LSTMModel:
        return LSTMModel(config)

    def _create_artifacts(
        self,
        config: LSTMConfig,
        feature_scaler: Any,
        model: LSTMModel,
        version: str,
    ) -> LSTMArtifacts:
        return LSTMArtifacts(
            config=config,
            feature_scaler=feature_scaler,
            model=model,
            version=version,
        )

    def _generate_readme(self, version: str, metadata: dict[str, Any]) -> str:
        return f"""---
tags:
- lstm
- finance
- weekly-returns
- learnfinance
---

# LearnFinance LSTM Model - {version}

LSTM model for predicting weekly stock returns.

## Model Details

- **Version**: {version}
- **Training Window**: {metadata.get('data_window', {}).get('start', 'N/A')} to {metadata.get('data_window', {}).get('end', 'N/A')}
- **Symbols**: {len(metadata.get('symbols', []))} stocks

## Metrics

- Train Loss: {metadata.get('metrics', {}).get('train_loss', 'N/A')}
- Validation Loss: {metadata.get('metrics', {}).get('val_loss', 'N/A')}
- Baseline Loss: {metadata.get('metrics', {}).get('baseline_loss', 'N/A')}

## Usage

```python
from brain_api.storage.huggingface import HuggingFaceModelStorage

storage = HuggingFaceModelStorage(repo_id="{self.repo_id}")
artifacts = storage.download_model(version="{version}")
```
"""

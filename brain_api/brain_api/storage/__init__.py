"""Storage abstractions for model artifacts and datasets.

This module provides a clean API for storage operations:
- Local filesystem storage for LSTM and PatchTST models
- HuggingFace Hub storage for models and datasets

The module is organized into submodules:
- storage/lstm/       - LSTM model storage (local + HuggingFace)
- storage/patchtst/   - PatchTST model storage (local + HuggingFace stub)
- storage/datasets/   - Dataset storage (HuggingFace)

For convenience, commonly used classes are re-exported from:
- storage/local.py    - All local storage classes
- storage/huggingface.py - All HuggingFace storage classes
"""

# Re-export from wrapper modules for backward compatibility
from brain_api.storage.huggingface import (
    HFModelInfo,
    HuggingFaceDatasetStorage,
    HuggingFaceModelStorage,
)
from brain_api.storage.local import (
    DEFAULT_DATA_PATH,
    LocalModelStorage,
    LSTMArtifacts,
    PatchTSTArtifacts,
    PatchTSTModelStorage,
    create_metadata,
    create_patchtst_metadata,
)

__all__ = [
    # Local storage
    "DEFAULT_DATA_PATH",
    # HuggingFace storage
    "HFModelInfo",
    "HuggingFaceDatasetStorage",
    "HuggingFaceModelStorage",
    "LSTMArtifacts",
    "LocalModelStorage",
    "PatchTSTArtifacts",
    "PatchTSTModelStorage",
    "create_metadata",
    "create_patchtst_metadata",
]

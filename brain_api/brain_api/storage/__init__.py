"""Storage abstractions for model artifacts."""

from brain_api.storage.local import (
    LSTMArtifacts,
    LocalModelStorage,
    create_metadata,
)
from brain_api.storage.huggingface import (
    HuggingFaceDatasetStorage,
    HuggingFaceModelStorage,
    HFModelInfo,
)

__all__ = [
    # Local storage
    "LSTMArtifacts",
    "LocalModelStorage",
    "create_metadata",
    # HuggingFace storage
    "HuggingFaceDatasetStorage",
    "HuggingFaceModelStorage",
    "HFModelInfo",
]
